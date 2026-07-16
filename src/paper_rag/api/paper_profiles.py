from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.db import get_db_session
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.schemas.paper_profiles import (
    PaperProfileBackfillRequest,
    PaperProfileBackfillResponse,
    PaperProfileResponse,
)
from paper_rag.services.paper_profile import build_paper_profile, get_ready_profile

router = APIRouter(prefix="/api/paper-profiles", tags=["paper-profiles"])


def _response(profile) -> PaperProfileResponse:
    return PaperProfileResponse(
        profile_id=profile.id,
        document_id=profile.document_id,
        status=profile.status,
        profile_version=profile.profile_version,
        parser_version=profile.parser_version,
        source_sha256=profile.source_sha256,
        content=json.loads(profile.content_json or "{}"),
    )


@router.get("/{document_id}", response_model=PaperProfileResponse)
def read_paper_profile(
    document_id: UUID,
    session: Session = Depends(get_db_session),
) -> PaperProfileResponse:
    profile = get_ready_profile(session, document_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ready Paper Profile not found")
    return _response(profile)


@router.post("/backfill", response_model=PaperProfileBackfillResponse)
def backfill_profiles(
    request: PaperProfileBackfillRequest,
    session: Session = Depends(get_db_session),
) -> PaperProfileBackfillResponse:
    document_ids = request.document_ids or list(
        session.scalars(
            select(Document.id)
            .where(Document.status == DocumentStatus.COMPLETED)
            .order_by(Document.created_at)
            .limit(100)
        )
    )
    profiles = []
    failures = []
    for document_id in document_ids:
        try:
            profiles.append(_response(build_paper_profile(session, document_id)))
        except Exception as exc:
            failures.append(
                {"document_id": str(document_id), "error_code": type(exc).__name__, "detail": str(exc)}
            )
    return PaperProfileBackfillResponse(profiles=profiles, failures=failures)
