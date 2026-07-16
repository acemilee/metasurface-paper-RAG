from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class PaperProfileResponse(BaseModel):
    profile_id: UUID
    document_id: UUID
    status: str
    profile_version: int
    parser_version: str
    source_sha256: str
    content: dict


class PaperProfileBackfillRequest(BaseModel):
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)


class PaperProfileBackfillResponse(BaseModel):
    profiles: list[PaperProfileResponse]
    failures: list[dict]
