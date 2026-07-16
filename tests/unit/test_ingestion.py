from uuid import uuid4

from paper_rag.services.ingestion import job_queue


def test_job_queue_is_bounded() -> None:
    assert job_queue.maxsize == 32


def test_job_ids_are_queueable() -> None:
    job_queue.put_nowait(uuid4())
    assert job_queue.get_nowait() is not None
