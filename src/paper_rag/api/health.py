from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from paper_rag.config import Settings, get_settings
from paper_rag.db import get_db_session
from paper_rag.services.metrics import build_readiness_report

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    report = build_readiness_report(session, settings)
    return JSONResponse(
        status_code=200 if report.ready else 503,
        content=report.as_dict(),
    )
