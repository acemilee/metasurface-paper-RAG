from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import fitz


class PdfType(StrEnum):
    DIGITAL_TEXT = "digital_text_pdf"
    SCANNED = "scanned_pdf"
    HYBRID = "hybrid_pdf"
    ENCRYPTED_OR_INVALID = "encrypted_or_invalid"


@dataclass(frozen=True)
class PdfClassification:
    pdf_type: PdfType
    page_count: int
    text_page_count: int


def classify_pdf(path: Path) -> PdfClassification:
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError):
        return PdfClassification(PdfType.ENCRYPTED_OR_INVALID, 0, 0)
    with document:
        if document.is_encrypted:
            return PdfClassification(PdfType.ENCRYPTED_OR_INVALID, document.page_count, 0)
        text_pages = sum(1 for page in document if len(page.get_text("text").strip()) >= 80)
        if text_pages == document.page_count:
            pdf_type = PdfType.DIGITAL_TEXT
        elif text_pages == 0:
            pdf_type = PdfType.SCANNED
        else:
            pdf_type = PdfType.HYBRID
        return PdfClassification(pdf_type, document.page_count, text_pages)
