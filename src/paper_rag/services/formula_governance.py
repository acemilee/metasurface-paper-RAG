from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from uuid import UUID
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import FormulaReference
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION


FORMULA_NUMBER_AT_END = re.compile(r"\((?P<number>\d{1,3}[a-z]?)\)\s*$", re.IGNORECASE)
FORMULA_REFERENCE = re.compile(
    r"(?:Equation|Eq\.|formula|\u65b9\u7a0b|\u516c\u5f0f|\u5f0f)\s*\(?\s*(?P<number>\d{1,3}[a-z]?)\s*\)?",
    re.IGNORECASE,
)
FORMULA_LIKE_TEXT = re.compile(
    r"(?:[=\u2248\u2264\u2265\u2211\u222b\u221a\u2202\u00b1]|\^\s*[-+]?\d|_\{?|\(\d{1,3}[a-z]?\)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
TRUNCATED_FORMULA = re.compile(
    r"^\s*(?::|;|,)?\s*\(\d{1,3}[a-z]?\)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FormulaAnomaly:
    code: str
    severity: str
    document_id: UUID
    page_number: int | None = None
    formula_id: UUID | None = None
    chunk_id: UUID | None = None
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("document_id", "formula_id", "chunk_id"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class FormulaInventoryReport:
    document_count: int
    formula_count: int
    chunk_count: int
    anomalies: tuple[FormulaAnomaly, ...]
    counts_by_code: dict[str, int]
    counts_by_severity: dict[str, int]
    signature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "document_count": self.document_count,
            "formula_count": self.formula_count,
            "chunk_count": self.chunk_count,
            "anomaly_count": len(self.anomalies),
            "counts_by_code": self.counts_by_code,
            "counts_by_severity": self.counts_by_severity,
            "signature": self.signature,
            "anomalies": [item.as_dict() for item in self.anomalies],
        }


def _parse_bbox(value: str) -> tuple[float, float, float, float] | None:
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


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1.0, min(left_area, right_area))


def _parse_formula_ids(value: str) -> tuple[UUID, ...] | None:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            return None
        return tuple(UUID(str(item)) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _sort_key(item: FormulaAnomaly) -> tuple[str, int, str, str, str]:
    return (
        str(item.document_id),
        item.page_number or 0,
        item.code,
        str(item.formula_id or ""),
        str(item.chunk_id or ""),
    )


def _anomaly_signature(anomalies: list[FormulaAnomaly]) -> str:
    payload = [item.as_dict() for item in sorted(anomalies, key=_sort_key)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def scan_formula_inventory(
    session: Session,
    *,
    document_ids: Collection[UUID] | None = None,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> FormulaInventoryReport:
    requested = tuple(sorted(set(document_ids or ()), key=str))
    document_statement = select(Document).order_by(Document.id)
    if requested:
        document_statement = document_statement.where(Document.id.in_(requested))
    documents = list(session.scalars(document_statement))
    selected_ids = {item.id for item in documents}
    if not selected_ids:
        return FormulaInventoryReport(0, 0, 0, (), {}, {}, _anomaly_signature([]))

    formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.document_id.in_(selected_ids))
            .order_by(Formula.document_id, Formula.page_number, Formula.id)
        )
    )
    chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id.in_(selected_ids))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
    )
    unresolved_references = list(
        session.scalars(
            select(FormulaReference)
            .where(
                FormulaReference.document_id.in_(selected_ids),
                FormulaReference.resolution_status != "resolved",
            )
            .order_by(FormulaReference.document_id, FormulaReference.source_page, FormulaReference.id)
        )
    )
    anomalies: list[FormulaAnomaly] = []
    formulas_by_page: dict[tuple[UUID, int], list[Formula]] = defaultdict(list)
    formulas_by_id = {item.id: item for item in formulas}

    for formula in formulas:
        formulas_by_page[(formula.document_id, formula.page_number)].append(formula)
        number_match = FORMULA_NUMBER_AT_END.search((formula.raw_text or "").strip())
        expected_number = number_match.group("number").lower() if number_match else None
        if formula.parser_version != current_parser_version:
            anomalies.append(
                FormulaAnomaly(
                    "old_parser_version",
                    "error",
                    formula.document_id,
                    formula.page_number,
                    formula.id,
                    details={
                        "actual": formula.parser_version,
                        "expected": current_parser_version,
                    },
                )
            )
        if formula.formula_number is None and (expected_number or formula.parser_version != current_parser_version):
            anomalies.append(
                FormulaAnomaly(
                    "missing_formula_number",
                    "error",
                    formula.document_id,
                    formula.page_number,
                    formula.id,
                    details={"detected_number": expected_number},
                )
            )
        if formula.group_key is None and (formula.formula_number or expected_number or formula.parser_version != current_parser_version):
            anomalies.append(
                FormulaAnomaly(
                    "missing_formula_group",
                    "error",
                    formula.document_id,
                    formula.page_number,
                    formula.id,
                    details={"formula_number": formula.formula_number or expected_number},
                )
            )
        if not formula.raw_text or TRUNCATED_FORMULA.fullmatch(formula.raw_text):
            anomalies.append(
                FormulaAnomaly(
                    "truncated_formula_text",
                    "error",
                    formula.document_id,
                    formula.page_number,
                    formula.id,
                    details={"raw_text": (formula.raw_text or "")[:80]},
                )
            )
        if _parse_bbox(formula.bbox_json) is None:
            anomalies.append(
                FormulaAnomaly(
                    "invalid_bbox",
                    "critical",
                    formula.document_id,
                    formula.page_number,
                    formula.id,
                )
            )
        if formula.physical_meaning and formula.formula_number:
            referenced = {
                match.group("number").lower()
                for match in FORMULA_REFERENCE.finditer(formula.physical_meaning)
            }
            own_number = formula.formula_number.lower()
            foreign = sorted(number for number in referenced if number != own_number)
            if foreign:
                anomalies.append(
                    FormulaAnomaly(
                        "cross_formula_context_binding",
                        "error",
                        formula.document_id,
                        formula.page_number,
                        formula.id,
                        details={"formula_number": own_number, "referenced_numbers": foreign},
                    )
                )

    for (document_id, page_number), page_formulas in formulas_by_page.items():
        number_counts = Counter(
            item.formula_number.lower() for item in page_formulas if item.formula_number
        )
        for number, count in sorted(number_counts.items()):
            if count > 1:
                anomalies.append(
                    FormulaAnomaly(
                        "duplicate_formula_number",
                        "critical",
                        document_id,
                        page_number,
                        details={"formula_number": number, "count": count},
                    )
                )
        valid = [
            (formula, _parse_bbox(formula.bbox_json))
            for formula in page_formulas
            if _parse_bbox(formula.bbox_json) is not None
        ]
        for index, (left_formula, left_bbox) in enumerate(valid):
            assert left_bbox is not None
            for right_formula, right_bbox in valid[index + 1 :]:
                assert right_bbox is not None
                if _bbox_overlap_ratio(left_bbox, right_bbox) >= 0.85:
                    anomalies.append(
                        FormulaAnomaly(
                            "overlapping_bbox",
                            "critical",
                            document_id,
                            page_number,
                            left_formula.id,
                            details={"other_formula_id": str(right_formula.id)},
                        )
                    )

    for chunk in chunks:
        linked_ids = _parse_formula_ids(chunk.formula_ids_json)
        stale_reasons: list[str] = []
        if linked_ids is None:
            stale_reasons.append("invalid_json")
            linked_ids = ()
        else:
            for formula_id in linked_ids:
                formula = formulas_by_id.get(formula_id)
                if formula is None:
                    stale_reasons.append(f"missing:{formula_id}")
                elif formula.document_id != chunk.document_id:
                    stale_reasons.append(f"cross_document:{formula_id}")
                elif not (
                    chunk.page_start <= formula.page_number <= chunk.page_end
                    and (formula.placeholder in chunk.content or (formula.raw_text or "") in chunk.content)
                ):
                    stale_reasons.append(f"content_mismatch:{formula_id}")
        if stale_reasons:
            anomalies.append(
                FormulaAnomaly(
                    "stale_chunk_formula_ids",
                    "critical",
                    chunk.document_id,
                    chunk.page_start,
                    chunk_id=chunk.id,
                    details={"reasons": sorted(stale_reasons)},
                )
            )
        page_formulas = [
            formula
            for page in range(chunk.page_start, chunk.page_end + 1)
            for formula in formulas_by_page.get((chunk.document_id, page), ())
        ]
        if FORMULA_LIKE_TEXT.search(chunk.content) and (not page_formulas or not linked_ids):
            anomalies.append(
                FormulaAnomaly(
                    "formula_like_chunk_without_formula",
                    "error",
                    chunk.document_id,
                    chunk.page_start,
                    chunk_id=chunk.id,
                    details={"page_end": chunk.page_end},
                )
            )

    for reference in unresolved_references:
        anomalies.append(
            FormulaAnomaly(
                "unresolved_formula_dependency",
                "error",
                reference.document_id,
                reference.source_page,
                reference.source_formula_id,
                details={"referenced_number": reference.referenced_number},
            )
        )

    anomalies.sort(key=_sort_key)
    counts_by_code = dict(sorted(Counter(item.code for item in anomalies).items()))
    counts_by_severity = dict(sorted(Counter(item.severity for item in anomalies).items()))
    return FormulaInventoryReport(
        document_count=len(documents),
        formula_count=len(formulas),
        chunk_count=len(chunks),
        anomalies=tuple(anomalies),
        counts_by_code=counts_by_code,
        counts_by_severity=counts_by_severity,
        signature=_anomaly_signature(anomalies),
    )


def assert_current_formula_records(
    formulas: Collection[Formula],
    *,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> None:
    stale_versions = sorted(
        {item.parser_version for item in formulas if item.parser_version != current_parser_version}
    )
    if stale_versions:
        raise ValueError(
            "New formula records must use "
            f"{current_parser_version}; rejected versions: {', '.join(stale_versions)}"
        )


def derive_formula_index_status(
    session: Session,
    document_id: UUID,
    *,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> FormulaIndexStatus:
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    formulas = list(
        session.scalars(select(Formula).where(Formula.document_id == document_id))
    )
    if any(item.parser_version != current_parser_version for item in formulas):
        status = FormulaIndexStatus.STALE
        parser_version = document.formula_parser_version or (
            formulas[0].parser_version if formulas else None
        )
    else:
        report = scan_formula_inventory(
            session,
            document_ids=[document_id],
            current_parser_version=current_parser_version,
        )
        has_blocking_anomaly = any(
            item.code != "source_pdf_missing" for item in report.anomalies
        )
        needs_fidelity_review = any(
            item.fidelity_status != "source_exact" for item in formulas
        )
        status = (
            FormulaIndexStatus.NEEDS_REVIEW
            if has_blocking_anomaly or needs_fidelity_review
            else FormulaIndexStatus.READY
        )
        parser_version = current_parser_version
    document.formula_index_status = status
    document.formula_parser_version = parser_version
    document.formula_index_updated_at = datetime.now().astimezone()
    session.commit()
    return status


def mark_stale_formula_indexes(
    session: Session,
    *,
    current_parser_version: str = FORMULA_PARSER_VERSION,
) -> int:
    documents = list(
        session.scalars(
            select(Document).where(
                Document.formula_parser_version.is_not(None),
                Document.formula_parser_version != current_parser_version,
                Document.formula_index_status != FormulaIndexStatus.BUILDING,
                Document.formula_index_status != FormulaIndexStatus.STALE,
            )
        )
    )
    now = datetime.now().astimezone()
    for document in documents:
        document.formula_index_status = FormulaIndexStatus.STALE
        document.formula_index_updated_at = now
    if documents:
        session.commit()
    return len(documents)
