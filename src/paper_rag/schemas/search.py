from uuid import UUID

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str
    top_n: int = 5
    document_id: UUID | None = None
    scope: Literal["all", "selected"] = "all"
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)


class EvidenceItem(BaseModel):
    chunk_id: UUID
    document_id: UUID
    page_start: int
    page_end: int
    section_path: str | None
    content: str
    formula_ids: list[str]
    score: float
    quality_score: float
    has_low_confidence_ocr: bool


class SearchResponse(BaseModel):
    sufficient: bool
    reason: str | None
    intent: str
    intent_confidence: float
    intent_source: str
    evidence: list[EvidenceItem]
