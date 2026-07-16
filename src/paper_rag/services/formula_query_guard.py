from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from paper_rag.models.document import Document, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.services.formula_dependencies import (
    FormulaDependencyBundle,
    FormulaQueryRoute,
    build_formula_dependency_bundle,
)
from paper_rag.services.formula_jobs import enqueue_formula_backfill_job
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION


@dataclass(frozen=True)
class FormulaQueryReadiness:
    ready: bool
    audit_result: str
    reason: str
    bundle: FormulaDependencyBundle | None
    enqueued_job_ids: tuple[UUID, ...]


def route_formula_query(answer_mode) -> FormulaQueryRoute:
    value = answer_mode.value if hasattr(answer_mode, "value") else str(answer_mode)
    return {
        "extract": FormulaQueryRoute.SOURCE_RENDER,
        "derive": FormulaQueryRoute.CALCULATE_OR_DERIVE,
        "compare": FormulaQueryRoute.COMPARE,
    }.get(value, FormulaQueryRoute.EXPLAIN)


def _enqueue_repairs(
    session: Session,
    repair_pages: Mapping[UUID, Collection[int]],
) -> tuple[UUID, ...]:
    job_ids: list[UUID] = []
    for document_id in sorted(repair_pages, key=str):
        pages = sorted(set(repair_pages[document_id]))
        if not pages:
            continue
        job = enqueue_formula_backfill_job(
            session,
            document_id,
            pages,
            target_parser_version=FORMULA_PARSER_VERSION,
            apply_safe=True,
        )
        job_ids.append(job.id)
    return tuple(job_ids)


def guard_formula_query(
    session: Session,
    formulas: Collection[Formula],
    route: FormulaQueryRoute,
    *,
    repair_pages: Mapping[UUID, Collection[int]],
) -> FormulaQueryReadiness:
    records = list(formulas)
    document_ids = set(repair_pages) | {item.document_id for item in records}
    documents = {
        document_id: session.get(Document, document_id) for document_id in document_ids
    }
    stale_documents = [
        document_id
        for document_id, document in documents.items()
        if document is not None
        and (
            str(document.formula_index_status) == FormulaIndexStatus.STALE.value
            or document.formula_parser_version not in {None, FORMULA_PARSER_VERSION}
        )
    ]
    stale_records = [
        item for item in records if item.parser_version != FORMULA_PARSER_VERSION
    ]
    if stale_documents or stale_records:
        jobs = _enqueue_repairs(session, repair_pages)
        return FormulaQueryReadiness(
            False,
            "formula_index_stale",
            "公式索引版本已过期，已加入页级修复队列；修复完成前不会让模型补写公式。",
            None,
            jobs,
        )
    not_ready = [
        document_id
        for document_id, document in documents.items()
        if document is not None
        and str(document.formula_index_status)
        in {
            FormulaIndexStatus.PENDING.value,
            FormulaIndexStatus.BUILDING.value,
            FormulaIndexStatus.FAILED.value,
        }
    ]
    if not_ready:
        jobs = _enqueue_repairs(session, repair_pages)
        return FormulaQueryReadiness(
            False,
            "formula_index_not_ready",
            "公式索引尚未达到可查询状态，已加入页级修复队列。",
            None,
            jobs,
        )
    if not records:
        jobs = _enqueue_repairs(session, repair_pages)
        return FormulaQueryReadiness(
            False,
            "formula_not_extracted",
            "证据页未形成可回溯的公式记录，已加入页级修复队列。",
            None,
            jobs,
        )

    bundle = build_formula_dependency_bundle(
        session,
        [item.id for item in records],
        route,
    )
    unresolved = set(bundle.unresolved)
    if any(item.group_key and item.formula_group_id is None for item in records):
        unresolved.add("dependency_graph_missing")
    if unresolved:
        jobs = _enqueue_repairs(session, repair_pages)
        reason = "、".join(sorted(unresolved))
        return FormulaQueryReadiness(
            False,
            "formula_dependency_incomplete",
            f"公式依赖不完整（{reason}），已加入页级修复队列；系统不会猜测缺失部分。",
            bundle,
            jobs,
        )
    return FormulaQueryReadiness(
        True,
        "formula_dependency_complete",
        "公式版本与依赖完整性校验通过",
        bundle,
        (),
    )


def repair_pages_from_evidence(evidence) -> dict[UUID, set[int]]:
    pages: dict[UUID, set[int]] = defaultdict(set)
    for item in evidence:
        pages[item.document_id].update(range(item.page_start, item.page_end + 1))
    return dict(pages)
