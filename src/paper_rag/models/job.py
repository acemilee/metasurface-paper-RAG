from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from paper_rag.db import Base


class JobState(StrEnum):
    QUEUED = "queued"
    CLASSIFYING = "classifying"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    state: Mapped[JobState] = mapped_column(Enum(JobState), default=JobState.QUEUED, index=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    stage_durations_json: Mapped[str] = mapped_column(Text, default="{}")

    document: Mapped["Document"] = relationship(back_populates="jobs")


def claim_next_job(session: Session, worker_id: str) -> IngestionJob | None:
    statement = (
        select(IngestionJob)
        .where(IngestionJob.state == JobState.QUEUED)
        .order_by(IngestionJob.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = session.scalar(statement)
    if job is None:
        return None
    job.state = JobState.CLASSIFYING
    job.worker_id = worker_id
    job.started_at = datetime.now().astimezone()
    job.finished_at = None
    job.error_code = None
    job.error_message = None
    job.heartbeat_at = datetime.now().astimezone()
    job.attempt_count += 1
    session.commit()
    session.refresh(job)
    return job


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_job_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="idle")
