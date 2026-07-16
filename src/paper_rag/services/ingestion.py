from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.job import IngestionJob, JobState
from paper_rag.services.storage import StoredUpload
from paper_rag.config import get_settings
from paper_rag.services.domain_admission import CLASSIFIER_VERSION

job_queue: asyncio.Queue[UUID] = asyncio.Queue(maxsize=get_settings().queue_maxsize)


def create_or_get_document_job(session: Session, upload: StoredUpload) -> tuple[Document, IngestionJob, bool]:
    existing = session.scalar(select(Document).where(Document.file_sha256 == upload.sha256))
    if existing is not None:
        job = session.scalar(select(IngestionJob).where(IngestionJob.document_id == existing.id).order_by(IngestionJob.created_at.desc()).limit(1))
        if job is None:
            job = IngestionJob(document_id=existing.id, state=JobState.QUEUED)
            session.add(job)
            session.commit()
            session.refresh(job)
        return existing, job, False
    document = Document(
        original_filename=upload.original_filename,
        stored_path=str(upload.path),
        file_sha256=upload.sha256,
        status=DocumentStatus.QUEUED,
        domain_enforcement_version=CLASSIFIER_VERSION,
    )
    session.add(document)
    session.flush()
    job = IngestionJob(document_id=document.id, state=JobState.QUEUED)
    session.add(job)
    session.commit()
    session.refresh(document)
    session.refresh(job)
    return document, job, True


async def enqueue_job(job_id: UUID) -> bool:
    try:
        job_queue.put_nowait(job_id)
        return True
    except asyncio.QueueFull:
        return False


def recover_queued_jobs(session: Session) -> list[UUID]:
    return list(session.scalars(select(IngestionJob.id).where(IngestionJob.state == JobState.QUEUED)))


def create_reindex_job(session: Session, document: Document) -> IngestionJob:
    active_states = {
        JobState.QUEUED,
        JobState.CLASSIFYING,
        JobState.PARSING,
        JobState.CHUNKING,
        JobState.EMBEDDING,
        JobState.INDEXING,
    }
    active_job = session.scalar(
        select(IngestionJob)
        .where(IngestionJob.document_id == document.id, IngestionJob.state.in_(active_states))
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    if active_job is not None:
        return active_job
    document.status = DocumentStatus.QUEUED
    job = IngestionJob(document_id=document.id, state=JobState.QUEUED)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job
