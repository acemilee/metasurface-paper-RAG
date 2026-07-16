from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from paper_rag.api.jobs import get_batch_job_status
from paper_rag.schemas.documents import JobBatchRequest


def test_batch_job_status_preserves_order_and_reports_missing_jobs() -> None:
    first_id = uuid4()
    missing_id = uuid4()
    third_id = uuid4()
    first = SimpleNamespace(
        id=first_id,
        document_id=uuid4(),
        state=SimpleNamespace(value="parsing"),
        error_code=None,
        error_message=None,
    )
    third = SimpleNamespace(
        id=third_id,
        document_id=uuid4(),
        state=SimpleNamespace(value="queued"),
        error_code=None,
        error_message=None,
    )
    session = MagicMock()
    session.scalars.return_value.all.return_value = [third, first]

    result = get_batch_job_status(
        JobBatchRequest(job_ids=[first_id, missing_id, third_id]),
        session,
    )

    assert [job.job_id for job in result.jobs] == [first_id, third_id]
    assert result.missing_job_ids == [missing_id]


def test_batch_job_request_rejects_more_than_one_hundred_ids() -> None:
    with pytest.raises(ValidationError):
        JobBatchRequest(job_ids=[uuid4() for _ in range(101)])
