from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import fitz
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.document import Document
from paper_rag.models.formula import Formula


class FormulaSourceRegionError(ValueError):
    pass


@dataclass(frozen=True)
class FormulaCropHashReport:
    document_id: UUID
    hashed_formula_ids: tuple[UUID, ...]
    invalid_formula_ids: tuple[UUID, ...]


def parse_formula_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise FormulaSourceRegionError("invalid bbox")
        bbox = tuple(float(item) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FormulaSourceRegionError("invalid bbox") from exc
    if not all(math.isfinite(item) for item in bbox):
        raise FormulaSourceRegionError("non-finite bbox")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise FormulaSourceRegionError("empty bbox")
    return bbox


def _render_with_pdf(pdf: fitz.Document, formula: Formula) -> bytes:
    if not 1 <= formula.page_number <= len(pdf):
        raise FormulaSourceRegionError("invalid page")
    page = pdf[formula.page_number - 1]
    clip = fitz.Rect(*parse_formula_bbox(formula.bbox_json)) & page.rect
    if clip.is_empty or clip.width <= 1 or clip.height <= 1:
        raise FormulaSourceRegionError("empty bbox")
    clip = fitz.Rect(
        max(page.rect.x0, clip.x0 - 4),
        max(page.rect.y0, clip.y0 - 4),
        min(page.rect.x1, clip.x1 + 4),
        min(page.rect.y1, clip.y1 + 4),
    )
    return page.get_pixmap(
        matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False
    ).tobytes("png")


def render_formula_crop_png(document: Document, formula: Formula) -> bytes:
    path = Path(document.stored_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with fitz.open(path) as pdf:
            return _render_with_pdf(pdf, formula)
    except fitz.FileDataError as exc:
        raise FormulaSourceRegionError("invalid source PDF") from exc


def refresh_formula_source_crop_hashes(
    session: Session,
    document_id: UUID,
    pages: Collection[int] | None = None,
) -> FormulaCropHashReport:
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    statement = select(Formula).where(Formula.document_id == document_id)
    if pages is not None:
        statement = statement.where(Formula.page_number.in_(sorted(set(pages))))
    formulas = list(session.scalars(statement.order_by(Formula.page_number, Formula.id)))
    hashed: list[UUID] = []
    invalid: list[UUID] = []
    path = Path(document.stored_path)
    if path.is_file():
        try:
            with fitz.open(path) as pdf:
                for formula in formulas:
                    try:
                        png = _render_with_pdf(pdf, formula)
                    except FormulaSourceRegionError:
                        formula.source_crop_sha256 = None
                        formula.fidelity_status = "unusable"
                        invalid.append(formula.id)
                    else:
                        formula.source_crop_sha256 = hashlib.sha256(png).hexdigest()
                        hashed.append(formula.id)
        except fitz.FileDataError:
            invalid.extend(item.id for item in formulas)
            for formula in formulas:
                formula.source_crop_sha256 = None
                formula.fidelity_status = "unusable"
    else:
        invalid.extend(item.id for item in formulas)
        for formula in formulas:
            formula.source_crop_sha256 = None
            formula.fidelity_status = "unusable"
    session.commit()
    return FormulaCropHashReport(document_id, tuple(hashed), tuple(invalid))
