from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.db import get_db_session
from paper_rag.models.job import IngestionJob
from paper_rag.schemas.documents import JobBatchRequest, JobBatchResponse, JobResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _job_response(job: IngestionJob) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        document_id=job.document_id,
        state=job.state.value,
        error_code=job.error_code,
        error_message=job.error_message,
    )


@router.post("/batch", response_model=JobBatchResponse)
def get_batch_job_status(
    request: JobBatchRequest,
    session: Session = Depends(get_db_session),
) -> JobBatchResponse:
    requested_ids = list(dict.fromkeys(request.job_ids))
    jobs = session.scalars(
        select(IngestionJob).where(IngestionJob.id.in_(requested_ids))
    ).all()
    jobs_by_id = {job.id: job for job in jobs}
    return JobBatchResponse(
        jobs=[_job_response(jobs_by_id[job_id]) for job_id in requested_ids if job_id in jobs_by_id],
        missing_job_ids=[job_id for job_id in requested_ids if job_id not in jobs_by_id],
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: UUID, session: Session = Depends(get_db_session)) -> JobResponse:
    job = session.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _job_response(job)
