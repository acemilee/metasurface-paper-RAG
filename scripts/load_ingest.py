from __future__ import annotations

import argparse
from uuid import uuid4

from sqlalchemy import select

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.models.document import Document
from paper_rag.models.job import IngestionJob, JobState


def submit_documents(job_count: int, dry_run: bool = True) -> int:
    settings = get_settings()
    with SessionLocal() as session:
        document = session.scalar(select(Document).limit(1))
        if document is None:
            raise RuntimeError("Load test requires one existing document")
        jobs = [IngestionJob(id=uuid4(), document_id=document.id, state=JobState.QUEUED) for _ in range(job_count)]
        session.add_all(jobs)
        session.flush()
        persisted = sum(session.get(IngestionJob, job.id) is not None for job in jobs)
        bounded_buffer_peak = min(job_count, settings.queue_maxsize)
        if dry_run:
            session.rollback()
        else:
            session.commit()
    print(f"jobs={persisted} queue_maxsize={settings.queue_maxsize} buffer_peak={bounded_buffer_peak} dry_run={dry_run}")
    return persisted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 10000:
        raise SystemExit("--jobs must be between 1 and 10000")
    submit_documents(args.jobs, dry_run=not args.commit)


if __name__ == "__main__":
    main()
