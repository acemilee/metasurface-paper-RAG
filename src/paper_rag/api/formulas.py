from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from paper_rag.db import get_db_session
from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.services.formula_assets import (
    FormulaSourceRegionError,
    parse_formula_bbox,
    render_formula_crop_png,
)


router = APIRouter(prefix="/api/formulas", tags=["formulas"])


def _parse_formula_bbox(value: str) -> tuple[float, float, float, float]:
    """Backward-compatible Phase F validation entry point."""
    return parse_formula_bbox(value)


@router.get("/{formula_id}/image")
def render_formula_image(
    formula_id: UUID,
    session: Session = Depends(get_db_session),
) -> Response:
    formula = session.get(Formula, formula_id)
    if formula is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Formula not found")
    document = session.get(Document, formula.document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Formula source PDF not found")
    try:
        png = render_formula_crop_png(document, formula)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Formula source PDF not found",
        ) from exc
    except FormulaSourceRegionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Formula source region is invalid",
        ) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
