from __future__ import annotations

import json
import re
from dataclasses import dataclass
from uuid import UUID

from paper_rag.models.formula import Formula
from paper_rag.services.pdf_parser import ParsedDocument


SECTION_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")


@dataclass(frozen=True)
class ChunkDraft:
    document_id: UUID
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    section_path: str | None
    content_type: str
    formula_ids: list[UUID]
    quality_score: float = 1.0
    has_low_confidence_ocr: bool = False


def make_vector_id(document_id: UUID, chunking_version: str, chunk_index: int) -> str:
    return f"{document_id}:{chunking_version}:{chunk_index}"


def _section_for_text(text: str, current: str | None) -> str | None:
    for line in text.splitlines():
        if SECTION_HEADING.match(line.strip()):
            return line.strip()
    return current


def build_chunks(document: ParsedDocument, formulas: list[Formula], target_chars: int, overlap_chars: int, ocr_numeric_min_confidence: float = 0.85) -> list[ChunkDraft]:
    formulas_by_page: dict[int, list[Formula]] = {}
    formula_pages: dict[UUID, int] = {}
    for formula in formulas:
        formulas_by_page.setdefault(formula.page_number, []).append(formula)
        formula_pages[formula.id] = formula.page_number
    chunks: list[ChunkDraft] = []
    section: str | None = None
    buffer = ""
    page_start = 1
    page_end = 1
    formula_ids: list[UUID] = []
    quality_scores: list[float] = []
    has_low_confidence_ocr = False

    def formula_ids_in_text(text: str) -> list[UUID]:
        return [formula.id for formula in formulas if formula.placeholder in text]

    for page in document.pages:
        quality_scores.append(page.quality_score)
        has_low_confidence_ocr = has_low_confidence_ocr or (
            page.extraction_method == "ocr" and page.quality_score < ocr_numeric_min_confidence
        )
        section = _section_for_text(page.text, section)
        page_text = page.text
        page_formulas = formulas_by_page.get(page.page_number, [])
        for formula in page_formulas:
            if formula.raw_text:
                page_text = page_text.replace(formula.raw_text, formula.placeholder, 1)
                meaning = formula.physical_meaning or "物理含义证据不足"
                page_text += f"\n[{formula.placeholder}: {meaning}]"
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page_text) if part.strip()]
        for paragraph in paragraphs:
            candidate = (buffer + "\n\n" + paragraph).strip() if buffer else paragraph
            if buffer and len(candidate) > target_chars:
                chunks.append(ChunkDraft(document.document_id, len(chunks), buffer, page_start, page_end, section, "paragraph", formula_ids.copy(), min(quality_scores), has_low_confidence_ocr))
                buffer = buffer[-overlap_chars:] + "\n\n" + paragraph if overlap_chars else paragraph
                page_start = page.page_number
                quality_scores = [page.quality_score]
                has_low_confidence_ocr = page.extraction_method == "ocr" and page.quality_score < ocr_numeric_min_confidence
            else:
                buffer = candidate
            formula_ids = formula_ids_in_text(buffer)
            if formula_ids:
                page_start = min(page_start, *(formula_pages[item] for item in formula_ids))
            page_end = page.page_number
    if buffer:
        chunks.append(ChunkDraft(document.document_id, len(chunks), buffer, page_start, page_end, section, "paragraph", formula_ids.copy(), min(quality_scores), has_low_confidence_ocr))
    return chunks
