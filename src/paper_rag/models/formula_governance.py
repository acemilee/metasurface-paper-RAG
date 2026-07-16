from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from paper_rag.db import Base


class FormulaBackfillJobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FormulaBackfillJob(Base):
    __tablename__ = "formula_backfill_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[FormulaBackfillJobState] = mapped_column(
        Enum(FormulaBackfillJobState, name="formula_backfill_job_state"),
        default=FormulaBackfillJobState.QUEUED,
        index=True,
    )
    page_numbers_json: Mapped[str] = mapped_column(Text)
    source_parser_versions_json: Mapped[str] = mapped_column(Text, default="[]")
    target_parser_version: Mapped[str] = mapped_column(String(64))
    apply_safe: Mapped[bool] = mapped_column(Boolean, default=False)
    inventory_signature: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)


class FormulaGroup(Base):
    __tablename__ = "formula_groups"
    __table_args__ = (UniqueConstraint("document_id", "group_key"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    group_key: Mapped[str] = mapped_column(String(128))
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    parser_version: Mapped[str] = mapped_column(String(64))
    completeness_status: Mapped[str] = mapped_column(String(32), default="complete")


class FormulaReference(Base):
    __tablename__ = "formula_references"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    source_formula_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), index=True
    )
    target_formula_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("formulas.id", ondelete="SET NULL"), nullable=True, index=True
    )
    referenced_number: Mapped[str] = mapped_column(String(32))
    source_page: Mapped[int] = mapped_column(Integer)
    evidence_text: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[str] = mapped_column(String(32))


class VariableDefinition(Base):
    __tablename__ = "formula_variable_definitions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    formula_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(64))
    definition: Mapped[str] = mapped_column(Text)
    source_page: Mapped[int] = mapped_column(Integer)
    evidence_text: Mapped[str] = mapped_column(Text)


class ApproximationCondition(Base):
    __tablename__ = "formula_approximation_conditions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    formula_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), index=True
    )
    condition_text: Mapped[str] = mapped_column(Text)
    source_page: Mapped[int] = mapped_column(Integer)
    evidence_text: Mapped[str] = mapped_column(Text)


class DerivationEdge(Base):
    __tablename__ = "formula_derivation_edges"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    source_formula_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("formulas.id", ondelete="SET NULL"), nullable=True, index=True
    )
    target_formula_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("formulas.id", ondelete="CASCADE"), index=True
    )
    evidence_text: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[str] = mapped_column(String(32))
