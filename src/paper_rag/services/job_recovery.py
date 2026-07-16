from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from paper_rag.models.document import DocumentStatus
from paper_rag.models.job import IngestionJob, JobState, WorkerHeartbeat


ACTIVE_STATES = {
    JobState.CLASSIFYING,
    JobState.PARSING,
    JobState.CHUNKING,
    JobState.EMBEDDING,
    JobState.INDEXING,
}


def update_worker_heartbeat(
    session: Session,
    worker_id: str,
    current_job_id: UUID | None = None,
    status: str = "idle",
) -> None:
    now = datetime.now().astimezone()
    heartbeat = session.get(WorkerHeartbeat, worker_id)
    if heartbeat is None:
        heartbeat = WorkerHeartbeat(worker_id=worker_id, last_seen_at=now)
        session.add(heartbeat)
    heartbeat.last_seen_at = now
    heartbeat.current_job_id = current_job_id
    heartbeat.status = status
    if current_job_id is not None:
        job = session.get(IngestionJob, current_job_id)
        if job is not None:
            job.heartbeat_at = now
    session.commit()


def reclaim_stale_jobs(
    session: Session, stale_after: timedelta, max_attempts: int = 3
) -> int:
    cutoff = datetime.now().astimezone() - stale_after
    jobs = list(
        session.scalars(
            select(IngestionJob).where(
                IngestionJob.state.in_(ACTIVE_STATES),
                or_(
                    IngestionJob.heartbeat_at < cutoff,
                    and_(
                        IngestionJob.heartbeat_at.is_(None),
                        IngestionJob.started_at < cutoff,
                    ),
                ),
            )
        )
    )
    for job in jobs:
        if job.attempt_count >= max_attempts:
            job.state = JobState.FAILED
            job.finished_at = datetime.now().astimezone()
            job.error_code = "max_attempts_exceeded"
            job.error_message = "Worker exited repeatedly; manual review required"
            job.document.status = DocumentStatus.FAILED
            continue
        job.state = JobState.QUEUED
        job.worker_id = None
        job.started_at = None
        job.heartbeat_at = None
        job.error_code = "stale_job_recovered"
        job.error_message = "Worker heartbeat expired; job was requeued"
        job.document.status = DocumentStatus.QUEUED
    session.commit()
    return len(jobs)
