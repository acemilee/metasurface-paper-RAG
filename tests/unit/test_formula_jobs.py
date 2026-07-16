from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import FormulaBackfillJob, FormulaBackfillJobState
from paper_rag.services.formula_jobs import (
    assess_formula_backfill_plan_for_safe_apply,
    assess_formula_records_for_safe_apply,
    claim_next_formula_job,
    enqueue_formula_backfill_job,
    enqueue_stale_formula_jobs,
    requeue_resumable_formula_jobs,
    run_formula_job_batch,
)


def _session() -> tuple[Session, Document]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    document = Document(
        original_filename="paper.pdf",
        stored_path="paper.pdf",
        file_sha256="3" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=8,
    )
    session.add(document)
    session.commit()
    return session, document


def test_formula_backfill_job_is_persistent_normalized_and_auditable() -> None:
    session, document = _session()

    job = enqueue_formula_backfill_job(
        session,
        document.id,
        [4, 2, 4],
        target_parser_version="formula-layout-v3",
        apply_safe=True,
        inventory_signature="inventory-sha",
    )
    session.expire_all()
    persisted = session.get(FormulaBackfillJob, job.id)

    assert persisted is not None
    assert persisted.state == FormulaBackfillJobState.QUEUED
    assert json.loads(persisted.page_numbers_json) == [2, 4]
    assert persisted.target_parser_version == "formula-layout-v3"
    assert persisted.apply_safe is True
    assert persisted.inventory_signature == "inventory-sha"
    assert persisted.attempt_count == 0
    assert persisted.result_json == "{}"


def test_enqueue_deduplicates_active_job_but_allows_a_new_job_after_completion() -> None:
    session, document = _session()
    first = enqueue_formula_backfill_job(
        session,
        document.id,
        [2],
        target_parser_version="formula-layout-v3",
        apply_safe=False,
    )

    duplicate = enqueue_formula_backfill_job(
        session,
        document.id,
        [2, 3],
        target_parser_version="formula-layout-v3",
        apply_safe=True,
    )

    assert duplicate.id == first.id
    assert json.loads(duplicate.page_numbers_json) == [2, 3]
    assert duplicate.apply_safe is True
    first.state = FormulaBackfillJobState.COMPLETED
    session.commit()

    replacement = enqueue_formula_backfill_job(
        session,
        document.id,
        [2],
        target_parser_version="formula-layout-v3",
        apply_safe=True,
    )

    assert replacement.id != first.id
    assert len(list(session.scalars(select(FormulaBackfillJob)))) == 2


def test_claim_next_formula_job_records_worker_attempt_and_heartbeat() -> None:
    session, document = _session()
    queued = enqueue_formula_backfill_job(
        session,
        document.id,
        [1],
        target_parser_version="formula-layout-v3",
        apply_safe=True,
    )

    claimed = claim_next_formula_job(session, "formula-worker-1")

    assert claimed is not None
    assert claimed.id == queued.id
    assert claimed.state == FormulaBackfillJobState.RUNNING
    assert claimed.worker_id == "formula-worker-1"
    assert claimed.attempt_count == 1
    assert claimed.started_at is not None
    assert claimed.heartbeat_at is not None
    assert claim_next_formula_job(session, "formula-worker-2") is None


def test_enqueue_stale_formula_jobs_is_bounded_and_stably_ordered() -> None:
    session, first_document = _session()
    documents = [first_document]
    for index in range(2):
        document = Document(
            original_filename=f"paper-{index}.pdf",
            stored_path=f"paper-{index}.pdf",
            file_sha256=str(index + 4) * 64,
            status=DocumentStatus.COMPLETED,
            page_count=2,
        )
        session.add(document)
        session.flush()
        documents.append(document)
    for index, document in enumerate(documents):
        formula_id = document.id
        session.add(
            Formula(
                id=formula_id,
                document_id=document.id,
                page_number=1,
                placeholder=f"公式_placeholder_{formula_id}",
                bbox_json="[10, 10, 100, 30]",
                raw_text="x = y (1)",
                parser_version="legacy-v1",
            )
        )
    session.commit()

    jobs = enqueue_stale_formula_jobs(session, batch_size=2, apply_safe=True)

    assert len(jobs) == 2
    assert [job.document_id for job in jobs] == sorted(
        [document.id for document in documents], key=str
    )[:2]
    assert all(json.loads(job.page_numbers_json) == [1] for job in jobs)
    assert all(job.apply_safe for job in jobs)


def test_batch_runner_isolates_failures_and_stops_at_batch_limit() -> None:
    session, document = _session()
    jobs = [
        enqueue_formula_backfill_job(
            session,
            document.id,
            [page],
            target_parser_version=f"formula-layout-v{page + 3}",
            apply_safe=True,
        )
        for page in (1, 2, 3)
    ]

    calls: list[object] = []

    def execute(claimed: FormulaBackfillJob) -> dict:
        calls.append(claimed.id)
        if len(calls) == 1:
            raise RuntimeError("injected failure")
        return {"status": "applied", "pages": json.loads(claimed.page_numbers_json)}

    report = run_formula_job_batch(
        session,
        batch_size=2,
        worker_id="batch-worker",
        execute_job=execute,
    )

    assert report.claimed == 2
    assert report.completed == 1
    assert report.failed == 1
    states = [session.get(FormulaBackfillJob, job.id).state for job in jobs]
    assert states.count(FormulaBackfillJobState.FAILED) == 1
    assert states.count(FormulaBackfillJobState.COMPLETED) == 1
    assert states.count(FormulaBackfillJobState.QUEUED) == 1


def test_resume_requeues_interrupted_and_retryable_jobs_without_infinite_retry() -> None:
    session, document = _session()
    retryable = enqueue_formula_backfill_job(
        session,
        document.id,
        [1],
        target_parser_version="formula-layout-v4",
        apply_safe=True,
    )
    exhausted = enqueue_formula_backfill_job(
        session,
        document.id,
        [2],
        target_parser_version="formula-layout-v5",
        apply_safe=True,
    )
    retryable.state = FormulaBackfillJobState.RUNNING
    retryable.attempt_count = 1
    exhausted.state = FormulaBackfillJobState.FAILED
    exhausted.attempt_count = 3
    session.commit()

    count = requeue_resumable_formula_jobs(session, max_attempts=3)

    assert count == 1
    assert retryable.state == FormulaBackfillJobState.QUEUED
    assert exhausted.state == FormulaBackfillJobState.FAILED
    assert exhausted.error_code == "max_attempts_exceeded"


def test_safe_apply_requires_structural_integrity_but_preserves_review_fidelity() -> None:
    session, document = _session()
    structurally_safe = Formula(
        id=uuid4(),
        document_id=document.id,
        page_number=1,
        placeholder="safe",
        bbox_json="[10, 10, 100, 30]",
        raw_text="x = y (1)",
        formula_number="1",
        group_key="equation-1",
        parser_version="formula-layout-v3",
        fidelity_status="needs_review",
    )
    unsafe = Formula(
        id=uuid4(),
        document_id=document.id,
        page_number=2,
        placeholder="unsafe",
        bbox_json="[10, 10, 10, 30]",
        raw_text=": (2)",
        formula_number="2",
        group_key="equation-2",
        parser_version="legacy-v1",
    )

    safe_result = assess_formula_records_for_safe_apply([structurally_safe])
    unsafe_result = assess_formula_records_for_safe_apply([unsafe])

    assert safe_result.safe is True
    assert safe_result.review_formula_ids == (str(structurally_safe.id),)
    assert unsafe_result.safe is False
    assert {
        "old_parser_version",
        "truncated_formula_text",
        "invalid_bbox",
    } <= set(unsafe_result.reasons)


def test_safe_apply_allows_chunk_only_repair_without_replacing_unsafe_formulas() -> None:
    session, document = _session()
    existing_unsafe = Formula(
        id=uuid4(),
        document_id=document.id,
        page_number=2,
        placeholder="unsafe-existing",
        bbox_json="[10, 10, 10, 30]",
        raw_text=": (2)",
        formula_number="2",
        group_key="equation-2",
        parser_version="formula-layout-v3",
    )
    reparsed_same = Formula(
        id=existing_unsafe.id,
        document_id=document.id,
        page_number=2,
        placeholder=existing_unsafe.placeholder,
        bbox_json=existing_unsafe.bbox_json,
        raw_text=existing_unsafe.raw_text,
        formula_number=existing_unsafe.formula_number,
        group_key=existing_unsafe.group_key,
        parser_version=existing_unsafe.parser_version,
    )
    plan = SimpleNamespace(
        old_formulas=(existing_unsafe,),
        new_formulas=(reparsed_same,),
    )

    result = assess_formula_backfill_plan_for_safe_apply(plan)

    assert result.safe is True
    assert result.reasons == ()
