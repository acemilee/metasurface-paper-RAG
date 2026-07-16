from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from paper_rag.schemas.query_plan import QueryPlan
from paper_rag.services.formula_jobs import enqueue_formula_backfill_job
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION
from paper_rag.services.query_rewrite import LinkedEntity, link_soft_query_entities
from paper_rag.services.references.parser import parse_typed_references
from paper_rag.services.references.registry import resolve_typed_references
from paper_rag.services.references.types import (
    ReferenceResolution,
    ResolutionStatus,
    TypedReference,
)


@dataclass(frozen=True)
class ReferenceControlDecision:
    proceed: bool
    action: str = "answer"
    audit_result: str = "strong_references_resolved"
    reason: str | None = None


@dataclass(frozen=True)
class PreparedReferenceControl:
    references: tuple[TypedReference, ...]
    resolutions: tuple[ReferenceResolution, ...]
    soft_entities: tuple[LinkedEntity, ...]
    decision: ReferenceControlDecision


DECISIONS = {
    ResolutionStatus.NOT_FOUND: (
        "refuse",
        "strong_reference_not_found",
        "所选论文中不存在该强标识对象",
    ),
    ResolutionStatus.AMBIGUOUS: (
        "clarify",
        "strong_reference_ambiguous",
        "该标识在当前范围内对应多个对象",
    ),
    ResolutionStatus.STALE: (
        "refuse",
        "strong_reference_stale",
        "对象索引版本已过期，修复完成前不会补写",
    ),
    ResolutionStatus.INVALID: (
        "clarify",
        "strong_reference_invalid",
        "强标识格式无效或超出范围",
    ),
    ResolutionStatus.INDEX_INCONSISTENT: (
        "refuse",
        "reference_index_inconsistent",
        "对象存在，但原始证据索引不一致",
    ),
}


def decide_reference_control(
    resolutions: Sequence[ReferenceResolution],
) -> ReferenceControlDecision:
    for resolution in resolutions:
        decision = DECISIONS.get(resolution.status)
        if decision is not None:
            action, audit_result, reason = decision
            return ReferenceControlDecision(
                proceed=False,
                action=action,
                audit_result=audit_result,
                reason=reason,
            )
    return ReferenceControlDecision(proceed=True)


def prepare_reference_control(
    session: Session,
    original_question: str,
    query_plan: QueryPlan,
    document_scope: Collection[UUID],
) -> PreparedReferenceControl:
    scope = list(dict.fromkeys(document_scope))
    references = parse_typed_references(
        original_question,
        query_plan.standalone_question,
    )
    resolutions = resolve_typed_references(session, references, scope)
    soft_entities = tuple(
        link_soft_query_entities(session, query_plan.entities, scope)
    )
    return PreparedReferenceControl(
        references=references,
        resolutions=resolutions,
        soft_entities=soft_entities,
        decision=decide_reference_control(resolutions),
    )


def enqueue_reference_repairs(
    session: Session,
    resolutions: Sequence[ReferenceResolution],
) -> tuple[UUID, ...]:
    job_ids: list[UUID] = []
    for resolution in resolutions:
        if (
            resolution.reference.kind.value != "formula"
            or resolution.status
            not in {ResolutionStatus.STALE, ResolutionStatus.INDEX_INCONSISTENT}
            or not resolution.document_ids
        ):
            continue
        pages = sorted(
            {
                int(item)
                for item in resolution.diagnostics.get("page_numbers", [])
                if int(item) > 0
            }
        )
        if not pages:
            continue
        job = enqueue_formula_backfill_job(
            session,
            resolution.document_ids[0],
            pages,
            target_parser_version=FORMULA_PARSER_VERSION,
            apply_safe=True,
        )
        if job.id not in job_ids:
            job_ids.append(job.id)
    return tuple(job_ids)
