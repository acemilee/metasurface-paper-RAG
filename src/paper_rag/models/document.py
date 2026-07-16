from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from paper_rag.db import Base
from paper_rag.services.filename_search import normalize_filename_search_key


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class DomainStatus(StrEnum):
    UNCLASSIFIED = "unclassified"
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
    QUARANTINED = "quarantined"
    MANUAL_APPROVED = "manual_approved"


class DocumentGenre(StrEnum):
    UNCLASSIFIED = "unclassified"
    RESEARCH_PAPER = "research_paper"
    REVIEW_PAPER = "review_paper"
    THESIS = "thesis"
    CONFERENCE_PAPER = "conference_paper"


class FormulaIndexStatus(StrEnum):
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    STALE = "stale"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str] = mapped_column(String(512))
    filename_search_key: Mapped[str] = mapped_column(Text, nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), unique=True)
    file_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pdf_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.QUEUED)
    domain_status: Mapped[str] = mapped_column(String(32), default=DomainStatus.UNCLASSIFIED)
    domain_enforcement_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_decision_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_positive_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_negative_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    domain_classifier_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    domain_manual_override_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    document_genre: Mapped[str] = mapped_column(String(32), default=DocumentGenre.UNCLASSIFIED)
    genre_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    genre_decision_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    genre_scores_json: Mapped[str] = mapped_column(Text, default="{}")
    genre_evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    genre_conflicts_json: Mapped[str] = mapped_column(Text, default="[]")
    genre_manually_overridden: Mapped[bool] = mapped_column(Boolean, default=False)
    genre_original_prediction: Mapped[str | None] = mapped_column(String(32), nullable=True)
    genre_classifier_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    genre_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    formula_index_status: Mapped[FormulaIndexStatus] = mapped_column(
        String(32), default=FormulaIndexStatus.PENDING, index=True
    )
    formula_parser_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    formula_index_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @validates("original_filename")
    def _derive_filename_search_key(self, _key: str, value: str) -> str:
        self.filename_search_key = normalize_filename_search_key(value)
        return value

    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    pages: Mapped[list["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")
