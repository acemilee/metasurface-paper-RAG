from __future__ import annotations

import json
import multiprocessing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import fitz

from paper_rag.services.captions import extract_figure_captions
from paper_rag.services.ocr import ocr_page, render_page_for_ocr, should_use_ocr
from paper_rag.services.tables import extract_table_candidates, table_to_markdown

if TYPE_CHECKING:
    from paper_rag.config import Settings


@dataclass(frozen=True)
class ParsedTextBlock:
    page_number: int
    reading_order: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source: str = "digital_text"
    confidence: float = 1.0


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    blocks: list[ParsedTextBlock]
    extraction_method: str = "digital_text"
    quality_score: float = 1.0
    ocr_confidence: float | None = None


@dataclass(frozen=True)
class ParsedDocument:
    document_id: UUID
    page_count: int
    pages: list[ParsedPage]


def extract_page_blocks(page: fitz.Page, page_number: int) -> list[ParsedTextBlock]:
    raw_blocks = [block for block in page.get_text("blocks") if block[4].strip()]
    raw_blocks.sort(key=lambda block: (round(block[1], 1), round(block[0], 1)))
    return [
        ParsedTextBlock(page_number, index, block[4].strip(), float(block[0]), float(block[1]), float(block[2]), float(block[3]))
        for index, block in enumerate(raw_blocks)
    ]


def normalize_reading_order(blocks: list[ParsedTextBlock]) -> list[ParsedTextBlock]:
    return sorted(blocks, key=lambda block: (block.reading_order, block.y0, block.x0))


def parse_text_pdf(path: Path, document_id: UUID) -> ParsedDocument:
    document = fitz.open(path)
    with document:
        pages = [
            ParsedPage(page_number=index + 1, text=page.get_text("text").strip(), blocks=normalize_reading_order(extract_page_blocks(page, index + 1)))
            for index, page in enumerate(document)
        ]
    return ParsedDocument(document_id=document_id, page_count=len(pages), pages=pages)


def _ocr_worker(image_path: str, render_dpi: int, cpu_threads: int):
    return ocr_page(Path(image_path), render_dpi, cpu_threads)


class OcrProcessRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool = None

    def __enter__(self):
        return self

    def run(self, image_path: Path):
        if self._pool is None:
            self._pool = multiprocessing.get_context("spawn").Pool(processes=1)
        pending = self._pool.apply_async(
            _ocr_worker,
            (str(image_path), self.settings.ocr_render_dpi, self.settings.ocr_cpu_threads),
        )
        try:
            return pending.get(timeout=self.settings.ocr_page_timeout_seconds)
        except multiprocessing.TimeoutError as exc:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            raise TimeoutError("OCR page exceeded configured timeout") from exc

    def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        self._pool.join()
        self._pool = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


def parse_pdf(path: Path, document_id: UUID, settings: Settings) -> ParsedDocument:
    pages = []
    ocr_page_count = 0
    with OcrProcessRunner(settings) as ocr_runner:
        with fitz.open(path) as document:
            for index, page in enumerate(document):
                page_number = index + 1
                digital_text = page.get_text("text").strip()
                blocks = normalize_reading_order(extract_page_blocks(page, page_number))
                extraction_method = "digital_text"
                ocr_confidence = None
                if settings.ocr_enabled and should_use_ocr(
                    digital_text, len(blocks), settings.ocr_min_page_chars
                ):
                    if ocr_page_count >= settings.ocr_max_pages_per_document:
                        raise ValueError("PDF exceeds configured OCR page limit")
                    image_path = render_page_for_ocr(
                        path, page_number, settings.ocr_render_dpi
                    )
                    try:
                        result = ocr_runner.run(image_path)
                    finally:
                        image_path.unlink(missing_ok=True)
                    blocks = [
                        ParsedTextBlock(
                            page_number,
                            order,
                            item.text,
                            *item.bbox,
                            source="ocr",
                            confidence=item.confidence,
                        )
                        for order, item in enumerate(result.blocks)
                    ]
                    digital_text = result.text
                    extraction_method = "ocr"
                    ocr_confidence = result.mean_confidence
                    ocr_page_count += 1
                captions = extract_figure_captions(blocks)
                caption_texts = {caption.text for caption in captions}
                blocks = [
                    replace(block, source="caption")
                    if block.text in caption_texts
                    else block
                    for block in blocks
                ]
                try:
                    tables = extract_table_candidates(path, page_number)
                except Exception:
                    tables = []
                for table in tables:
                    table_bbox = table.bbox or (
                        0.0, 0.0, float(page.rect.width), float(page.rect.height)
                    )
                    blocks.append(
                        ParsedTextBlock(
                            page_number,
                            len(blocks),
                            table_to_markdown(table),
                            *table_bbox,
                            source="table",
                            confidence=table.confidence,
                        )
                    )
                quality = (
                    sum(block.confidence for block in blocks) / len(blocks)
                    if blocks
                    else 0.0
                )
                combined_text = digital_text
                if tables:
                    combined_text += "\n\n" + "\n\n".join(
                        table_to_markdown(table) for table in tables
                    )
                pages.append(
                    ParsedPage(
                        page_number,
                        combined_text.strip(),
                        normalize_reading_order(blocks),
                        extraction_method,
                        quality,
                        ocr_confidence,
                    )
                )
    return ParsedDocument(document_id, len(pages), pages)


def write_page_jsonl(parsed: ParsedDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for page in parsed.pages:
            record = {"document_id": str(parsed.document_id), "page_number": page.page_number, "text": page.text, "blocks": [asdict(block) for block in page.blocks]}
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
