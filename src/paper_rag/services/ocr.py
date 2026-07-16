from __future__ import annotations

import os
import tempfile
import gc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fitz


@dataclass(frozen=True)
class OcrTextBlock:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class OcrPageResult:
    blocks: list[OcrTextBlock]
    mean_confidence: float

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


def should_use_ocr(text: str, block_count: int, min_page_chars: int = 80) -> bool:
    return len("".join(text.split())) < min_page_chars or block_count == 0


def render_page_for_ocr(pdf_path: Path, page_number: int, dpi: int = 216) -> Path:
    scale = dpi / 72.0
    with fitz.open(pdf_path) as document:
        pixmap = document[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(scale, scale), alpha=False
        )
    handle, output_name = tempfile.mkstemp(
        prefix=f"paper-rag-p{page_number}-", suffix=".png"
    )
    os.close(handle)
    output_path = Path(output_name)
    pixmap.save(output_path)
    return output_path


@lru_cache(maxsize=1)
def _get_ocr_engine(cpu_threads: int):
    from paddleocr import PaddleOCR

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=cpu_threads,
    )


def ocr_page(
    image_path: Path, render_dpi: int = 216, cpu_threads: int = 4
) -> OcrPageResult:
    results = list(_get_ocr_engine(cpu_threads).predict(str(image_path)))
    if not results:
        return OcrPageResult([], 0.0)
    result = results[0]
    scale = render_dpi / 72.0
    blocks = []
    for text, score, box in zip(
        result.get("rec_texts", []),
        result.get("rec_scores", []),
        result.get("rec_boxes", []),
    ):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        x0, y0, x1, y1 = (float(value) / scale for value in box)
        blocks.append(OcrTextBlock(cleaned, (x0, y0, x1, y1), float(score)))
    blocks.sort(key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))
    mean = sum(block.confidence for block in blocks) / len(blocks) if blocks else 0.0
    return OcrPageResult(blocks, mean)


def release_ocr_engine() -> None:
    _get_ocr_engine.cache_clear()
    gc.collect()
