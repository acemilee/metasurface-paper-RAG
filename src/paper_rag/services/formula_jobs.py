from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from collections.abc import Collection
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import FormulaBackfillJob, FormulaBackfillJobState
from paper_rag.config import Settings
from paper_rag.services.embeddings import EmbeddingProvider
from paper_rag.services.formula_backfill import (
    FormulaBackfillPlan,
    apply_formula_backfill,
    formula_records_changed,
    plan_formula_backfill,
)
from paper_rag.services.formula_dependencies import rebuild_formula_dependency_graph
from paper_rag.services.formula_assets import refresh_formula_source_crop_hashes
from paper_rag.services.formula_governance import derive_formula_index_status, scan_formula_inventory
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION


ACTIVE_FORMULA_JOB_STATES = {
    FormulaBackfillJobState.QUEUED,
    FormulaBackfillJobState.RUNNING,
}
FORMULA_GOVERNANCE_VERSION = "formula-governance-v4"


@dataclass(frozen=True)
class FormulaJobBatchReport:
    claimed: int
    completed: int
    needs_review: int
    failed: int
    job_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class FormulaSafeApplyAssessment:
    safe: bool
    reasons: tuple[str, ...]
    review_formula_ids: tuple[str, ...]


def _safe_bbox(value: str) -> tuple[float, float, float, float] | None:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or len(parsed) != 4:
            return None
        bbox = tuple(float(item) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    area = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    smaller = min(
        (left[2] - left[0]) * (left[3] - left[1]),
        (right[2] - right[0]) * (right[3] - right[1]),
    )
    return area / max(1.0, smaller)


def assess_formula_records_for_safe_apply(
    formulas: Collection[Formula],
    *,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> FormulaSafeApplyAssessment:
    records = list(formulas)
    reasons: set[str] = set()
    review_ids = sorted(
        str(item.id) for item in records if item.fidelity_status == "needs_review"
    )
    if not records:
        reasons.add("no_formula_records")
    bboxes_by_page: dict[int, list[tuple[Formula, tuple[float, float, float, float]]]] = defaultdict(list)
    for formula in records:
        if formula.parser_version != current_parser_version:
            reasons.add("old_parser_version")
        raw_text = (formula.raw_text or "").strip()
        if not raw_text or raw_text.startswith(":") and raw_text.endswith(")"):
            reasons.add("truncated_formula_text")
        bbox = _safe_bbox(formula.bbox_json)
        if bbox is None:
            reasons.add("invalid_bbox")
        else:
            bboxes_by_page[formula.page_number].append((formula, bbox))
    number_counts = Counter(
        (item.page_number, item.formula_number.lower())
        for item in records
        if item.formula_number
    )
    if any(count > 1 for count in number_counts.values()):
        reasons.add("duplicate_formula_number")
    for page_records in bboxes_by_page.values():
        for index, (_, left) in enumerate(page_records):
            if any(_overlap_ratio(left, right) >= 0.85 for _, right in page_records[index + 1 :]):
                reasons.add("overlapping_bbox")
    return FormulaSafeApplyAssessment(
        safe=not reasons,
        reasons=tuple(sorted(reasons)),
        review_formula_ids=tuple(review_ids),
    )


def assess_formula_backfill_plan_for_safe_apply(
    plan: FormulaBackfillPlan,
    *,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> FormulaSafeApplyAssessment:
    assessment = assess_formula_records_for_safe_apply(
        plan.new_formulas,
        current_parser_version=current_parser_version,
    )
    if formula_records_changed(plan.old_formulas, plan.new_formulas):
        return assessment
    return FormulaSafeApplyAssessment(
        safe=True,
        reasons=(),
        review_formula_ids=assessment.review_formula_ids,
    )


def _normalize_pages(document: Document, page_numbers: Collection[int]) -> list[int]:
    pages = sorted(set(page_numbers))
    if not pages or any(page < 1 for page in pages):
        raise ValueError("At least one positive page number is required")
    if document.page_count is not None and any(page > document.page_count for page in pages):
        raise ValueError("Formula backfill page is outside the document")
    return pages


def enqueue_formula_backfill_job(
    session: Session,
    document_id: UUID,
    page_numbers: Collection[int],
    *,
    target_parser_version: str,
    apply_safe: bool,
    inventory_signature: str | None = None,
) -> FormulaBackfillJob:
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    pages = _normalize_pages(document, page_numbers)
    active = session.scalar(
        select(FormulaBackfillJob)
        .where(
            FormulaBackfillJob.document_id == document_id,
            FormulaBackfillJob.target_parser_version == target_parser_version,
            FormulaBackfillJob.state.in_(ACTIVE_FORMULA_JOB_STATES),
        )
        .order_by(FormulaBackfillJob.created_at.desc())
        .limit(1)
    )
    if active is not None:
        existing_pages = json.loads(active.page_numbers_json)
        active.page_numbers_json = json.dumps(sorted(set(existing_pages) | set(pages)))
        active.apply_safe = active.apply_safe or apply_safe
        if inventory_signature is not None:
            active.inventory_signature = inventory_signature
        session.commit()
        session.refresh(active)
        return active

    source_versions = sorted(
        set(
            session.scalars(
                select(Formula.parser_version).where(Formula.document_id == document_id)
            )
        )
    )
    job = FormulaBackfillJob(
        document_id=document_id,
        state=FormulaBackfillJobState.QUEUED,
        page_numbers_json=json.dumps(pages),
        source_parser_versions_json=json.dumps(source_versions),
        target_parser_version=target_parser_version,
        apply_safe=apply_safe,
        inventory_signature=inventory_signature,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def claim_next_formula_job(session: Session, worker_id: str) -> FormulaBackfillJob | None:
    job = session.scalar(
        select(FormulaBackfillJob)
        .where(FormulaBackfillJob.state == FormulaBackfillJobState.QUEUED)
        .order_by(FormulaBackfillJob.created_at, FormulaBackfillJob.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        return None
    now = datetime.now().astimezone()
    job.state = FormulaBackfillJobState.RUNNING
    job.worker_id = worker_id
    job.started_at = now
    job.finished_at = None
    job.heartbeat_at = now
    job.error_code = None
    job.error_message = None
    job.attempt_count += 1
    session.commit()
    session.refresh(job)
    return job


def enqueue_stale_formula_jobs(
    session: Session,
    *,
    batch_size: int,
    apply_safe: bool,
    document_ids: Collection[UUID] | None = None,
    target_parser_version: str = FORMULA_PARSER_VERSION,
) -> list[FormulaBackfillJob]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    report = scan_formula_inventory(
        session,
        document_ids=document_ids,
        current_parser_version=target_parser_version,
    )
    pages_by_document: dict[UUID, set[int]] = defaultdict(set)
    anomalies_by_document: dict[UUID, list[dict[str, object]]] = defaultdict(list)
    for anomaly in report.anomalies:
        anomalies_by_document[anomaly.document_id].append(anomaly.as_dict())
        if anomaly.page_number is not None:
            pages_by_document[anomaly.document_id].add(anomaly.page_number)
    jobs: list[FormulaBackfillJob] = []
    for document_id in sorted(pages_by_document, key=str):
        if len(jobs) >= batch_size:
            break
        document_signature = hashlib.sha256(
            json.dumps(
                {
                    "governance_version": FORMULA_GOVERNANCE_VERSION,
                    "anomalies": anomalies_by_document[document_id],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        already_processed = session.scalar(
            select(FormulaBackfillJob.id).where(
                FormulaBackfillJob.document_id == document_id,
                FormulaBackfillJob.target_parser_version == target_parser_version,
                FormulaBackfillJob.inventory_signature == document_signature,
                FormulaBackfillJob.state.in_(
                    {
                        FormulaBackfillJobState.COMPLETED,
                        FormulaBackfillJobState.NEEDS_REVIEW,
                    }
                ),
            ).limit(1)
        )
        if already_processed is not None:
            continue
        jobs.append(
            enqueue_formula_backfill_job(
                session,
                document_id,
                pages_by_document[document_id],
                target_parser_version=target_parser_version,
                apply_safe=apply_safe,
                inventory_signature=document_signature,
            )
        )
    return jobs


def requeue_resumable_formula_jobs(session: Session, *, max_attempts: int = 3) -> int:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    jobs = list(
        session.scalars(
            select(FormulaBackfillJob).where(
                FormulaBackfillJob.state.in_(
                    {FormulaBackfillJobState.RUNNING, FormulaBackfillJobState.FAILED}
                )
            )
        )
    )
    requeued = 0
    for job in jobs:
        if job.attempt_count >= max_attempts:
            job.state = FormulaBackfillJobState.FAILED
            job.error_code = "max_attempts_exceeded"
            job.error_message = "Formula backfill exceeded the configured retry limit"
            job.finished_at = datetime.now().astimezone()
            continue
        job.state = FormulaBackfillJobState.QUEUED
        job.worker_id = None
        job.started_at = None
        job.finished_at = None
        job.heartbeat_at = None
        job.error_code = "interrupted_job_requeued"
        job.error_message = "Interrupted formula backfill was requeued"
        requeued += 1
    session.commit()
    return requeued


def run_formula_job_batch(
    session: Session,
    *,
    batch_size: int,
    worker_id: str,
    execute_job: Callable[[FormulaBackfillJob], dict],
) -> FormulaJobBatchReport:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    completed = 0
    needs_review = 0
    failed = 0
    job_ids: list[UUID] = []
    for _ in range(batch_size):
        job = claim_next_formula_job(session, worker_id)
        if job is None:
            break
        job_id = job.id
        job_ids.append(job_id)
        try:
            result = execute_job(job)
            session.expire_all()
            job = session.get(FormulaBackfillJob, job_id)
            if job is None:
                raise RuntimeError("Formula backfill job disappeared during execution")
            terminal = result.get("status", "completed")
            if terminal == "needs_review":
                job.state = FormulaBackfillJobState.NEEDS_REVIEW
                needs_review += 1
            else:
                job.state = FormulaBackfillJobState.COMPLETED
                completed += 1
            job.result_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
            job.finished_at = datetime.now().astimezone()
            job.heartbeat_at = datetime.now().astimezone()
            session.commit()
        except Exception as exc:
            session.rollback()
            job = session.get(FormulaBackfillJob, job_id)
            if job is not None:
                job.state = FormulaBackfillJobState.FAILED
                job.error_code = type(exc).__name__[:64]
                job.error_message = str(exc)[:2000]
                job.finished_at = datetime.now().astimezone()
                job.heartbeat_at = datetime.now().astimezone()
                session.commit()
            failed += 1
    return FormulaJobBatchReport(
        claimed=len(job_ids),
        completed=completed,
        needs_review=needs_review,
        failed=failed,
        job_ids=tuple(job_ids),
    )


def execute_persisted_formula_job(
    session: Session,
    job: FormulaBackfillJob,
    settings: Settings,
    provider: EmbeddingProvider | None = None,
    collection=None,
) -> dict:
    pages = json.loads(job.page_numbers_json)
    if not isinstance(pages, list) or not all(isinstance(item, int) for item in pages):
        raise ValueError("Formula backfill job contains invalid pages")
    plan = plan_formula_backfill(session, job.document_id, pages, settings)
    assessment = assess_formula_backfill_plan_for_safe_apply(
        plan,
        current_parser_version=job.target_parser_version,
    )
    summary = {
        "document_id": str(job.document_id),
        "pages": list(plan.page_numbers),
        "source_state_sha256": plan.source_state_sha256,
        "changed_chunks": len(plan.changed_chunks),
        "safe": assessment.safe,
        "safety_reasons": list(assessment.reasons),
        "review_formula_ids": list(assessment.review_formula_ids),
    }
    if not assessment.safe:
        return {"status": "needs_review", "mode": "blocked_unsafe", **summary}
    if not job.apply_safe:
        return {"status": "needs_review", "mode": "dry_run", **summary}
    if provider is None or collection is None:
        raise ValueError("Safe apply requires an embedding provider and vector collection")
    applied = apply_formula_backfill(session, plan, settings, provider, collection)
    crop_report = refresh_formula_source_crop_hashes(
        session,
        job.document_id,
        plan.page_numbers,
    )
    dependency_report = rebuild_formula_dependency_graph(session, job.document_id)
    formula_index_status = derive_formula_index_status(
        session,
        job.document_id,
        current_parser_version=job.target_parser_version,
    )
    return {
        "mode": "apply_safe",
        **summary,
        **applied,
        "source_crops": {
            "hashed": len(crop_report.hashed_formula_ids),
            "invalid": len(crop_report.invalid_formula_ids),
        },
        "dependency_graph": {
            "groups": dependency_report.group_count,
            "references": dependency_report.reference_count,
            "variables": dependency_report.variable_count,
            "conditions": dependency_report.condition_count,
            "derivations": dependency_report.derivation_count,
            "unresolved": dependency_report.unresolved_count,
        },
        "formula_index_status": formula_index_status.value,
    }
