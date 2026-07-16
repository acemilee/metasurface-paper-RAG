from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from paper_rag.db import Base


class DomainAssessment(Base):
    __tablename__ = "domain_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    trigger: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(32), index=True)
    decision_code: Mapped[str] = mapped_column(String(64))
    classifier_version: Mapped[str] = mapped_column(String(64))
    embedding_model_id: Mapped[str] = mapped_column(String(255))
    config_fingerprint: Mapped[str] = mapped_column(String(64))
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    passed_requirements_json: Mapped[str] = mapped_column(Text, default="[]")
    failed_requirements_json: Mapped[str] = mapped_column(Text, default="[]")
    parse_quality: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[int] = mapped_column(Integer)
    applied_to_document: Mapped[bool] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class DomainManualOverride(Base):
    __tablename__ = "domain_manual_overrides"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("domain_assessments.id", ondelete="RESTRICT"), index=True
    )
    action: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
