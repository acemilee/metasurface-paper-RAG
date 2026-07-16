from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.services.formula_assets import FormulaSourceRegionError, render_formula_crop_png
from paper_rag.services.formula_dependencies import FormulaQueryRoute, build_formula_dependency_bundle
from paper_rag.services.formula_governance import scan_formula_inventory
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION, formula_placeholder


@dataclass(frozen=True)
class FormulaQualityIssue:
    severity: str
    code: str
    document_id: UUID | None = None
    formula_id: UUID | None = None
    page_number: int | None = None
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("document_id", "formula_id"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class FormulaQualityAcceptance:
    passed: bool
    requested_document_count: int
    requested_formula_count: int
    sampled_document_count: int
    sampled_formula_count: int
    sampled_document_ids: tuple[UUID, ...]
    sampled_formula_ids: tuple[UUID, ...]
    strata_counts: dict[str, int]
    issues: tuple[FormulaQualityIssue, ...]
    signature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "requested_document_count": self.requested_document_count,
            "requested_formula_count": self.requested_formula_count,
            "sampled_document_count": self.sampled_document_count,
            "sampled_formula_count": self.sampled_formula_count,
            "sampled_document_ids": [str(item) for item in self.sampled_document_ids],
            "sampled_formula_ids": [str(item) for item in self.sampled_formula_ids],
            "strata_counts": self.strata_counts,
            "issues": [item.as_dict() for item in self.issues],
            "signature": self.signature,
        }


def _stable_key(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _round_robin(groups: dict[str, list], limit: int) -> list:
    selected: list = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def _formula_stratum(document: Document, formula: Formula) -> str:
    group_shape = (
        "multipart"
        if formula.formula_number and re.fullmatch(r"\d{1,3}[a-z]", formula.formula_number, re.I)
        else "grouped" if formula.group_key else "ungrouped"
    )
    return "|".join(
        (
            document.pdf_type or "unknown",
            group_shape,
            formula.fidelity_status,
            formula.latex_verification_status,
        )
    )


def _issue_key(issue: FormulaQualityIssue) -> tuple:
    return (
        issue.severity,
        issue.code,
        str(issue.document_id or ""),
        str(issue.formula_id or ""),
        issue.page_number or 0,
        json.dumps(issue.details, ensure_ascii=False, sort_keys=True),
    )


def run_formula_quality_acceptance(
    session: Session,
    *,
    min_documents: int = 30,
    min_formulas: int = 100,
    seed: int = 20260714,
) -> FormulaQualityAcceptance:
    if min_documents < 1 or min_formulas < 1:
        raise ValueError("Acceptance sample thresholds must be positive")
    eligible_documents = list(
        session.scalars(
            select(Document)
            .join(Formula, Formula.document_id == Document.id)
            .distinct()
        )
    )
    document_groups: dict[str, list[Document]] = defaultdict(list)
    for document in eligible_documents:
        document_groups[document.pdf_type or "unknown"].append(document)
    for values in document_groups.values():
        values.sort(key=lambda item: _stable_key(seed, item.id))
    sampled_documents = _round_robin(document_groups, min_documents)
    sampled_document_ids = tuple(item.id for item in sampled_documents)
    documents_by_id = {item.id: item for item in sampled_documents}

    formulas = list(
        session.scalars(
            select(Formula).where(Formula.document_id.in_(sampled_document_ids))
        )
    ) if sampled_document_ids else []
    formulas_by_document: dict[UUID, list[Formula]] = defaultdict(list)
    for formula in formulas:
        formulas_by_document[formula.document_id].append(formula)
    for values in formulas_by_document.values():
        values.sort(key=lambda item: _stable_key(seed, item.id))

    sampled_formulas: list[Formula] = []
    sampled_ids: set[UUID] = set()
    for document in sampled_documents:
        if formulas_by_document[document.id]:
            formula = formulas_by_document[document.id][0]
            sampled_formulas.append(formula)
            sampled_ids.add(formula.id)
    remaining_groups: dict[str, list[Formula]] = defaultdict(list)
    for formula in formulas:
        if formula.id not in sampled_ids:
            remaining_groups[_formula_stratum(documents_by_id[formula.document_id], formula)].append(formula)
    for values in remaining_groups.values():
        values.sort(key=lambda item: _stable_key(seed, item.id))
    sampled_formulas.extend(
        _round_robin(remaining_groups, max(0, min_formulas - len(sampled_formulas)))
    )
    sampled_formulas = sampled_formulas[: max(min_formulas, len(sampled_documents))]
    sampled_formula_ids = tuple(item.id for item in sampled_formulas)

    issues: list[FormulaQualityIssue] = []
    if len(sampled_documents) < min_documents:
        issues.append(
            FormulaQualityIssue(
                "P0",
                "insufficient_document_sample",
                details={"required": min_documents, "available": len(sampled_documents)},
            )
        )
    if len(sampled_formulas) < min_formulas:
        issues.append(
            FormulaQualityIssue(
                "P0",
                "insufficient_formula_sample",
                details={"required": min_formulas, "available": len(sampled_formulas)},
            )
        )

    chunk_formula_ids: set[UUID] = set()
    if sampled_document_ids:
        chunks = list(
            session.scalars(select(Chunk).where(Chunk.document_id.in_(sampled_document_ids)))
        )
        for chunk in chunks:
            try:
                parsed = json.loads(chunk.formula_ids_json)
                chunk_formula_ids.update(UUID(str(item)) for item in parsed if isinstance(parsed, list))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

    for formula in sampled_formulas:
        document = documents_by_id[formula.document_id]
        if formula.parser_version != FORMULA_PARSER_VERSION:
            issues.append(FormulaQualityIssue("P0", "old_parser_version", document.id, formula.id, formula.page_number))
        if str(document.formula_index_status) in {
            FormulaIndexStatus.PENDING.value,
            FormulaIndexStatus.BUILDING.value,
            FormulaIndexStatus.FAILED.value,
            FormulaIndexStatus.STALE.value,
        }:
            issues.append(
                FormulaQualityIssue(
                    "P0",
                    "formula_index_not_queryable",
                    document.id,
                    formula.id,
                    formula.page_number,
                    {"status": str(document.formula_index_status)},
                )
            )
        if document.page_count is None or not 1 <= formula.page_number <= document.page_count:
            issues.append(FormulaQualityIssue("P0", "invalid_page_number", document.id, formula.id, formula.page_number))
        if not formula.raw_text or re.fullmatch(r"\s*:?[ ]*\(\d{1,3}[a-z]?\)\s*", formula.raw_text, re.I):
            issues.append(FormulaQualityIssue("P0", "truncated_formula_text", document.id, formula.id, formula.page_number))
        if formula.formula_number and not formula.group_key:
            issues.append(FormulaQualityIssue("P0", "missing_formula_group", document.id, formula.id, formula.page_number))
        if formula.group_key and formula.formula_group_id is None:
            issues.append(FormulaQualityIssue("P0", "missing_formula_group_record", document.id, formula.id, formula.page_number))
        if formula.placeholder != formula_placeholder(formula.id):
            issues.append(FormulaQualityIssue("P0", "unstable_formula_placeholder", document.id, formula.id, formula.page_number))
        if formula.id not in chunk_formula_ids:
            issues.append(FormulaQualityIssue("P0", "missing_chunk_formula_link", document.id, formula.id, formula.page_number))
        try:
            crop = render_formula_crop_png(document, formula)
        except (FileNotFoundError, FormulaSourceRegionError):
            issues.append(FormulaQualityIssue("P0", "invalid_source_crop", document.id, formula.id, formula.page_number))
        else:
            actual_hash = hashlib.sha256(crop).hexdigest()
            if not formula.source_crop_sha256:
                issues.append(FormulaQualityIssue("P0", "missing_source_crop_hash", document.id, formula.id, formula.page_number))
            elif formula.source_crop_sha256 != actual_hash:
                issues.append(FormulaQualityIssue("P0", "source_crop_hash_mismatch", document.id, formula.id, formula.page_number))
        bundle = build_formula_dependency_bundle(
            session,
            [formula.id],
            FormulaQueryRoute.SOURCE_RENDER,
        )
        if not bundle.complete:
            issues.append(
                FormulaQualityIssue(
                    "P0",
                    "incomplete_formula_dependency",
                    document.id,
                    formula.id,
                    formula.page_number,
                    {"unresolved": list(bundle.unresolved)},
                )
            )

    if sampled_document_ids:
        inventory = scan_formula_inventory(session, document_ids=sampled_document_ids)
        p0_codes = {
            "old_parser_version",
            "missing_formula_number",
            "missing_formula_group",
            "truncated_formula_text",
            "invalid_bbox",
            "duplicate_formula_number",
            "overlapping_bbox",
            "stale_chunk_formula_ids",
            "unresolved_formula_dependency",
        }
        sampled_set = set(sampled_formula_ids)
        for anomaly in inventory.anomalies:
            if anomaly.formula_id is not None and anomaly.formula_id not in sampled_set:
                continue
            issues.append(
                FormulaQualityIssue(
                    "P0" if anomaly.code in p0_codes else "P1",
                    anomaly.code,
                    anomaly.document_id,
                    anomaly.formula_id,
                    anomaly.page_number,
                    anomaly.details,
                )
            )

    deduplicated = { _issue_key(item): item for item in issues }
    issues = [deduplicated[key] for key in sorted(deduplicated)]
    strata_counts = dict(sorted(Counter((item.pdf_type or "unknown") for item in sampled_documents).items()))
    signature_payload = {
        "seed": seed,
        "documents": [str(item) for item in sampled_document_ids],
        "formulas": [str(item) for item in sampled_formula_ids],
        "issues": [item.as_dict() for item in issues],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return FormulaQualityAcceptance(
        passed=not any(item.severity == "P0" for item in issues),
        requested_document_count=min_documents,
        requested_formula_count=min_formulas,
        sampled_document_count=len(sampled_documents),
        sampled_formula_count=len(sampled_formulas),
        sampled_document_ids=sampled_document_ids,
        sampled_formula_ids=sampled_formula_ids,
        strata_counts=strata_counts,
        issues=tuple(issues),
        signature=signature,
    )
