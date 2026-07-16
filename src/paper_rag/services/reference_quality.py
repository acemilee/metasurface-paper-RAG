from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.formula import Formula
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.services.references import (
    parse_typed_references,
    resolve_typed_references,
)
from paper_rag.services.references.types import ReferenceKind, ResolutionStatus
from paper_rag.services.references.structure import caption_references


@dataclass(frozen=True)
class ReferenceAcceptanceCase:
    question: str
    document_ids: tuple[UUID, ...]
    expected_status: ResolutionStatus
    expected_target_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class ReferenceAcceptanceIssue:
    severity: str
    code: str
    question: str
    document_ids: tuple[UUID, ...]
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "question": self.question,
            "document_ids": [str(item) for item in self.document_ids],
            "details": self.details,
        }


@dataclass(frozen=True)
class ReferenceQualityAcceptance:
    passed: bool
    formula_case_count: int
    figure_case_count: int
    table_case_count: int
    section_case_count: int
    issues: tuple[ReferenceAcceptanceIssue, ...]
    signature: str

    @property
    def p0_count(self) -> int:
        return sum(item.severity == "P0" for item in self.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "formula_case_count": self.formula_case_count,
            "figure_case_count": self.figure_case_count,
            "table_case_count": self.table_case_count,
            "section_case_count": self.section_case_count,
            "p0_count": self.p0_count,
            "issues": [item.as_dict() for item in self.issues],
            "signature": self.signature,
        }


def evaluate_reference_case(
    session: Session,
    case: ReferenceAcceptanceCase,
) -> ReferenceAcceptanceIssue | None:
    references = parse_typed_references(case.question)
    if len(references) != 1:
        return ReferenceAcceptanceIssue(
            severity="P0",
            code="reference_parse_count_mismatch",
            question=case.question,
            document_ids=case.document_ids,
            details={"expected_count": 1, "actual_count": len(references)},
        )
    resolutions = resolve_typed_references(session, references, case.document_ids)
    if len(resolutions) != 1:
        return ReferenceAcceptanceIssue(
            severity="P0",
            code="reference_resolution_count_mismatch",
            question=case.question,
            document_ids=case.document_ids,
            details={"expected_count": 1, "actual_count": len(resolutions)},
        )
    actual = resolutions[0]
    if actual.status != case.expected_status:
        return ReferenceAcceptanceIssue(
            severity="P0",
            code="reference_status_mismatch",
            question=case.question,
            document_ids=case.document_ids,
            details={
                "expected_status": case.expected_status.value,
                "actual_status": actual.status.value,
                "diagnostics": actual.diagnostics,
            },
        )
    expected_ids = tuple(sorted(case.expected_target_ids, key=str))
    actual_ids = tuple(sorted(actual.target_ids, key=str))
    if actual_ids != expected_ids:
        return ReferenceAcceptanceIssue(
            severity="P0",
            code="reference_target_mismatch",
            question=case.question,
            document_ids=case.document_ids,
            details={
                "expected_target_ids": [str(item) for item in expected_ids],
                "actual_target_ids": [str(item) for item in actual_ids],
            },
        )
    return None


def _formula_cases(session: Session) -> list[ReferenceAcceptanceCase]:
    formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.formula_number.is_not(None))
            .order_by(Formula.document_id, Formula.formula_number, Formula.id)
        )
    )
    grouped: dict[tuple[UUID, str], list[Formula]] = {}
    for formula in formulas:
        number = (formula.formula_number or "").lower()
        grouped.setdefault((formula.document_id, number), []).append(formula)
    cases: list[ReferenceAcceptanceCase] = []
    for (document_id, number), records in grouped.items():
        status = (
            ResolutionStatus.RESOLVED
            if len(records) == 1
            else ResolutionStatus.AMBIGUOUS
        )
        target_ids = tuple(sorted((item.id for item in records), key=str))
        for question in (
            f"公式{number}",
            f"式（{number}）",
            f"Eq. ({number})",
            f"Equation {number}",
        ):
            cases.append(
                ReferenceAcceptanceCase(
                    question=question,
                    document_ids=(document_id,),
                    expected_status=status,
                    expected_target_ids=target_ids,
                )
            )
    return cases


def _structure_cases(
    session: Session,
) -> dict[ReferenceKind, list[ReferenceAcceptanceCase]]:
    result = {
        ReferenceKind.FIGURE: [],
        ReferenceKind.TABLE: [],
        ReferenceKind.SECTION: [],
    }
    seen: set[tuple[UUID, ReferenceKind, str, str | None]] = set()
    chunks = list(
        session.scalars(
            select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index, Chunk.id)
        )
    )
    for chunk in chunks:
        for reference in caption_references(chunk.content):
            if reference.kind not in (ReferenceKind.FIGURE, ReferenceKind.TABLE):
                continue
            if not 1 <= int(reference.normalized_key) <= 999:
                continue
            identity = (
                chunk.document_id,
                reference.kind,
                reference.normalized_key,
                reference.qualifier,
            )
            if identity in seen:
                continue
            seen.add(identity)
            label = "Figure" if reference.kind == ReferenceKind.FIGURE else "Table"
            suffix = f"({reference.qualifier})" if reference.qualifier else ""
            resolution = resolve_typed_references(
                session,
                (reference,),
                (chunk.document_id,),
            )[0]
            result[reference.kind].append(
                ReferenceAcceptanceCase(
                    question=f"{label} {reference.normalized_key}{suffix}",
                    document_ids=(chunk.document_id,),
                    expected_status=ResolutionStatus.RESOLVED,
                    expected_target_ids=resolution.target_ids,
                )
            )
        section_match = re.match(
            r"^\s*(\d+(?:\.\d+)*)\b",
            chunk.section_path or "",
        )
        if section_match:
            number = section_match.group(1)
            identity = (
                chunk.document_id,
                ReferenceKind.SECTION,
                number,
                None,
            )
            if identity not in seen:
                seen.add(identity)
                resolution = resolve_typed_references(
                    session,
                    parse_typed_references(f"Section {number}"),
                    (chunk.document_id,),
                )[0]
                result[ReferenceKind.SECTION].append(
                    ReferenceAcceptanceCase(
                        question=f"Section {number}",
                        document_ids=(chunk.document_id,),
                        expected_status=ResolutionStatus.RESOLVED,
                        expected_target_ids=resolution.target_ids,
                    )
                )
    profiles = list(
        session.scalars(
            select(PaperProfile)
            .where(PaperProfile.status == "ready")
            .order_by(PaperProfile.document_id, PaperProfile.profile_version.desc())
        )
    )
    latest_profiles: dict[UUID, PaperProfile] = {}
    for profile in profiles:
        latest_profiles.setdefault(profile.document_id, profile)
    for document_id, profile in latest_profiles.items():
        try:
            entries = json.loads(profile.content_json or "{}").get(
                "figure_table_index",
                [],
            )
        except (TypeError, json.JSONDecodeError):
            entries = []
        for entry in entries:
            caption = str(entry.get("caption") or "")
            for reference in parse_typed_references(caption):
                if reference.kind not in (ReferenceKind.FIGURE, ReferenceKind.TABLE):
                    continue
                if not 1 <= int(reference.normalized_key) <= 999:
                    continue
                identity = (
                    document_id,
                    reference.kind,
                    reference.normalized_key,
                    reference.qualifier,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                label = (
                    "Figure"
                    if reference.kind == ReferenceKind.FIGURE
                    else "Table"
                )
                suffix = (
                    f"({reference.qualifier})"
                    if reference.qualifier
                    else ""
                )
                try:
                    chunk_id = UUID(str(entry.get("chunk_id") or ""))
                except ValueError:
                    chunk_id = None
                target_ids = ()
                if chunk_id is not None:
                    chunk = session.scalar(
                        select(Chunk).where(
                            Chunk.id == chunk_id,
                            Chunk.document_id == document_id,
                        )
                    )
                    if chunk is not None:
                        resolution = resolve_typed_references(
                            session,
                            (reference,),
                            (document_id,),
                        )[0]
                        target_ids = resolution.target_ids
                result[reference.kind].append(
                    ReferenceAcceptanceCase(
                        question=(
                            f"{label} {reference.normalized_key}{suffix}"
                        ),
                        document_ids=(document_id,),
                        expected_status=ResolutionStatus.RESOLVED,
                        expected_target_ids=target_ids,
                    )
                )
    expanded: dict[ReferenceKind, list[ReferenceAcceptanceCase]] = {
        kind: [] for kind in result
    }
    for kind, cases in result.items():
        for case in cases:
            reference = parse_typed_references(case.question)[0]
            suffix = f"({reference.qualifier})" if reference.qualifier else ""
            if kind == ReferenceKind.FIGURE:
                questions = (
                    f"Figure {reference.normalized_key}{suffix}",
                    f"图{reference.normalized_key}{suffix}",
                )
            elif kind == ReferenceKind.TABLE:
                questions = (
                    f"Table {reference.normalized_key}",
                    f"表{reference.normalized_key}",
                )
            else:
                questions = (
                    f"Section {reference.normalized_key}",
                    f"第{reference.normalized_key}节",
                )
            for question in questions:
                expanded[kind].append(
                    ReferenceAcceptanceCase(
                        question=question,
                        document_ids=case.document_ids,
                        expected_status=case.expected_status,
                        expected_target_ids=case.expected_target_ids,
                    )
                )
    return expanded


def run_reference_quality_acceptance(
    session: Session,
    *,
    min_formula_cases: int,
    min_figure_cases: int,
    min_table_cases: int,
    min_section_cases: int,
    seed: int,
) -> ReferenceQualityAcceptance:
    formula_cases = _formula_cases(session)
    structures = _structure_cases(session)
    all_cases = [
        *formula_cases,
        *structures[ReferenceKind.FIGURE],
        *structures[ReferenceKind.TABLE],
        *structures[ReferenceKind.SECTION],
    ]
    issues = [
        issue
        for case in all_cases
        if (issue := evaluate_reference_case(session, case)) is not None
    ]
    counts_and_minimums = (
        ("formula", len(formula_cases), min_formula_cases),
        ("figure", len(structures[ReferenceKind.FIGURE]), min_figure_cases),
        ("table", len(structures[ReferenceKind.TABLE]), min_table_cases),
        ("section", len(structures[ReferenceKind.SECTION]), min_section_cases),
    )
    for kind, actual, minimum in counts_and_minimums:
        if actual < minimum:
            issues.append(
                ReferenceAcceptanceIssue(
                    severity="P0",
                    code=f"insufficient_{kind}_cases",
                    question="",
                    document_ids=(),
                    details={"minimum": minimum, "actual": actual},
                )
            )
    signature_payload = {
        "seed": seed,
        "cases": [
            {
                "question": item.question,
                "documents": [str(value) for value in item.document_ids],
                "status": item.expected_status.value,
                "targets": [str(value) for value in item.expected_target_ids],
            }
            for item in all_cases
        ],
        "issues": [item.code for item in issues],
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    result_issues = tuple(issues)
    return ReferenceQualityAcceptance(
        passed=not any(item.severity == "P0" for item in result_issues),
        formula_case_count=len(formula_cases),
        figure_case_count=len(structures[ReferenceKind.FIGURE]),
        table_case_count=len(structures[ReferenceKind.TABLE]),
        section_case_count=len(structures[ReferenceKind.SECTION]),
        issues=result_issues,
        signature=signature,
    )
