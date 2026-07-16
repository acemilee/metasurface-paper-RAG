from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


class UploadAccepted(BaseModel):
    document_id: UUID
    job_id: UUID
    duplicate: bool
    status: str


class JobResponse(BaseModel):
    job_id: UUID
    document_id: UUID
    state: str
    error_code: str | None
    error_message: str | None


class JobBatchRequest(BaseModel):
    job_ids: list[UUID] = Field(min_length=1, max_length=100)


class JobBatchResponse(BaseModel):
    jobs: list[JobResponse]
    missing_job_ids: list[UUID]


class DocumentListItem(BaseModel):
    document_id: UUID
    original_filename: str
    status: str
    pdf_type: str | None
    page_count: int | None
    chunk_count: int
    domain_status: str
    domain_score: float | None
    domain_reasons: list[str]
    domain_assessment_id: UUID | None
    domain_decision_code: str | None
    domain_passed_requirements: list[str]
    domain_failed_requirements: list[str]
    domain_evidence: list[dict]
    document_genre: str
    genre_score: float | None
    genre_decision_source: str | None
    genre_scores: dict[str, float]
    genre_evidence: list[dict]
    genre_conflicts: list[str]
    genre_manually_overridden: bool
    profile_status: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    next_cursor: UUID | None


class DomainApproveRequest(BaseModel):
    assessment_id: UUID


class DomainReviewResponse(BaseModel):
    document_id: UUID
    job_id: UUID
    status: str
    assessment_id: UUID
    override_id: UUID


class DeletionCheckResponse(BaseModel):
    document_id: UUID
    original_filename: str
    stored_domain_status: str
    fresh_assessment_id: UUID
    fresh_domain_status: str
    fresh_decision_code: str
    passed_requirements: list[str]
    failed_requirements: list[str]
    evidence: list[dict]
    page_count: int
    chunk_count: int
    vector_count: int
    answer_audit_count: int
    warning: str | None
    confirmation_token: str
    expires_in_seconds: int


class DeleteDocumentRequest(BaseModel):
    confirmation_token: str
    confirm_filename: str


class DeleteDocumentResponse(BaseModel):
    document_id: UUID
    original_filename: str
    deleted_chunks: int
    deleted_vectors: int
