from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID, uuid5

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import (
    ApproximationCondition,
    DerivationEdge,
    FormulaGroup,
    FormulaReference,
    VariableDefinition,
)
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION


EXPLICIT_FORMULA_REFERENCE = re.compile(
    r"(?:Eq(?:uation)?\.?|formula|\u65b9\u7a0b|\u516c\u5f0f|\u5f0f)\s*\(\s*(?P<number>\d{1,3}[a-z]?)\s*\)",
    re.IGNORECASE,
)
VARIABLE_DEFINITION = re.compile(
    r"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]{0,30})\s+"
    r"(?:denotes|represents|is)\s+(?P<definition>[^.;,]{2,160})",
    re.IGNORECASE,
)
APPROXIMATION_MARKERS = (
    "approximation",
    "assuming",
    "assumption",
    "under the",
    "in the limit",
    "provided that",
    "when ",
)
DERIVATION_MARKERS = ("derived", "follows", "using eq", "from eq", "substituting")


class FormulaQueryRoute(StrEnum):
    SOURCE_RENDER = "source_render"
    EXPLAIN = "explain"
    CALCULATE_OR_DERIVE = "calculate_or_derive"
    COMPARE = "compare"


@dataclass(frozen=True)
class DependencyBuildReport:
    document_id: UUID
    group_count: int
    reference_count: int
    variable_count: int
    condition_count: int
    derivation_count: int
    unresolved_count: int


@dataclass(frozen=True)
class FormulaDependencyBundle:
    route: FormulaQueryRoute
    target_formula_ids: tuple[UUID, ...]
    dependency_formula_ids: tuple[UUID, ...]
    variable_definitions: tuple[dict[str, object], ...]
    approximation_conditions: tuple[dict[str, object], ...]
    citations: tuple[dict[str, object], ...]
    unresolved: tuple[str, ...]
    complete: bool


def missing_formula_group_parts(formulas: Collection[Formula]) -> list[str]:
    parsed = []
    for formula in formulas:
        match = re.fullmatch(r"(\d{1,3})([a-z])", formula.formula_number or "", re.IGNORECASE)
        if match:
            parsed.append((match.group(1), match.group(2).lower()))
    if not parsed:
        return []
    bases = {base for base, _ in parsed}
    if len(bases) != 1:
        return ["mixed_formula_number_bases"]
    suffixes = {suffix for _, suffix in parsed}
    maximum = max(ord(suffix) for suffix in suffixes)
    base = next(iter(bases))
    return [
        f"{base}{chr(code)}"
        for code in range(ord("a"), maximum + 1)
        if chr(code) not in suffixes
    ]


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if item.strip()
    ]


def _resolve_formula_reference(
    number: str,
    by_number: dict[str, list[Formula]],
    formulas: Collection[Formula],
) -> Formula | None:
    direct = by_number.get(number, [])
    if len(direct) == 1:
        return direct[0]
    if direct or not re.fullmatch(r"\d{1,3}", number):
        return None
    grouped_parts = [
        formula
        for formula in formulas
        if re.fullmatch(rf"{re.escape(number)}[a-z]", formula.formula_number or "", re.I)
        and formula.group_key
    ]
    group_keys = {formula.group_key for formula in grouped_parts}
    if len(group_keys) != 1 or missing_formula_group_parts(grouped_parts):
        return None
    return min(grouped_parts, key=lambda item: (item.part_index, str(item.id)))


def rebuild_formula_dependency_graph(
    session: Session,
    document_id: UUID,
    pages: Collection[int] | None = None,
) -> DependencyBuildReport:
    formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.document_id == document_id)
            .order_by(Formula.page_number, Formula.group_key, Formula.part_index, Formula.id)
        )
    )
    session.execute(delete(DerivationEdge).where(DerivationEdge.document_id == document_id))
    session.execute(delete(FormulaReference).where(FormulaReference.document_id == document_id))
    session.execute(delete(VariableDefinition).where(VariableDefinition.document_id == document_id))
    session.execute(
        delete(ApproximationCondition).where(ApproximationCondition.document_id == document_id)
    )
    session.execute(
        update(Formula).where(Formula.document_id == document_id).values(formula_group_id=None)
    )
    session.execute(delete(FormulaGroup).where(FormulaGroup.document_id == document_id))
    session.flush()

    groups: dict[str, list[Formula]] = defaultdict(list)
    for formula in formulas:
        if formula.group_key:
            groups[formula.group_key].append(formula)
    group_ids: dict[str, UUID] = {}
    group_records: list[FormulaGroup] = []
    for group_key, parts in sorted(groups.items()):
        group_id = uuid5(document_id, f"formula-group|{group_key}")
        group_ids[group_key] = group_id
        group_records.append(
            FormulaGroup(
                id=group_id,
                document_id=document_id,
                group_key=group_key,
                page_start=min(item.page_number for item in parts),
                page_end=max(item.page_number for item in parts),
                parser_version=FORMULA_PARSER_VERSION,
                completeness_status=(
                    "incomplete" if missing_formula_group_parts(parts) else "complete"
                ),
            )
        )
    session.add_all(group_records)
    session.flush()
    for group_key, parts in groups.items():
        for formula in parts:
            formula.formula_group_id = group_ids[group_key]
    session.flush()

    by_number: dict[str, list[Formula]] = defaultdict(list)
    for formula in formulas:
        if formula.formula_number:
            by_number[formula.formula_number.lower()].append(formula)

    references: list[FormulaReference] = []
    variables: list[VariableDefinition] = []
    conditions: list[ApproximationCondition] = []
    derivations: list[DerivationEdge] = []
    for formula in formulas:
        contexts = [
            text for text in (formula.context_before, formula.context_after, formula.physical_meaning) if text
        ]
        seen_references: set[str] = set()
        for context in contexts:
            for match in EXPLICIT_FORMULA_REFERENCE.finditer(context):
                number = match.group("number").lower()
                if number == (formula.formula_number or "").lower() or number in seen_references:
                    continue
                seen_references.add(number)
                target = _resolve_formula_reference(number, by_number, formulas)
                resolution = "resolved" if target is not None else "unresolved"
                reference_id = uuid5(formula.id, f"reference|{number}")
                references.append(
                    FormulaReference(
                        id=reference_id,
                        document_id=document_id,
                        source_formula_id=formula.id,
                        target_formula_id=target.id if target else None,
                        referenced_number=number,
                        source_page=formula.page_number,
                        evidence_text=context,
                        resolution_status=resolution,
                    )
                )
                lowered = context.lower()
                if any(marker in lowered for marker in DERIVATION_MARKERS):
                    derivations.append(
                        DerivationEdge(
                            id=uuid5(formula.id, f"derivation|{number}"),
                            document_id=document_id,
                            source_formula_id=target.id if target else None,
                            target_formula_id=formula.id,
                            evidence_text=context,
                            resolution_status=resolution,
                        )
                    )
            for sentence in _sentences(context):
                for match in VARIABLE_DEFINITION.finditer(sentence):
                    symbol = match.group("symbol")
                    definition = match.group("definition").strip()
                    variables.append(
                        VariableDefinition(
                            id=uuid5(formula.id, f"variable|{symbol.lower()}|{definition.lower()}"),
                            document_id=document_id,
                            formula_id=formula.id,
                            symbol=symbol,
                            definition=definition,
                            source_page=formula.page_number,
                            evidence_text=sentence,
                        )
                    )
                if any(marker in sentence.lower() for marker in APPROXIMATION_MARKERS):
                    conditions.append(
                        ApproximationCondition(
                            id=uuid5(formula.id, f"condition|{sentence.lower()}"),
                            document_id=document_id,
                            formula_id=formula.id,
                            condition_text=sentence,
                            source_page=formula.page_number,
                            evidence_text=sentence,
                        )
                    )
    references = list({item.id: item for item in references}.values())
    variables = list({item.id: item for item in variables}.values())
    conditions = list({item.id: item for item in conditions}.values())
    derivations = list({item.id: item for item in derivations}.values())
    session.add_all([*references, *variables, *conditions, *derivations])
    session.commit()
    return DependencyBuildReport(
        document_id=document_id,
        group_count=len(groups),
        reference_count=len(references),
        variable_count=len(variables),
        condition_count=len(conditions),
        derivation_count=len(derivations),
        unresolved_count=sum(item.resolution_status == "unresolved" for item in references),
    )


def build_formula_dependency_bundle(
    session: Session,
    formula_ids: Collection[UUID],
    route: FormulaQueryRoute,
) -> FormulaDependencyBundle:
    requested = tuple(sorted(set(formula_ids), key=str))
    formulas = list(session.scalars(select(Formula).where(Formula.id.in_(requested)))) if requested else []
    unresolved: set[str] = set()
    if len(formulas) != len(requested):
        unresolved.add("missing_target_formula")
    document_ids = {item.document_id for item in formulas}
    if len(document_ids) > 1:
        unresolved.add("cross_document_target_bundle")

    dependency_ids: set[UUID] = set()

    def add_dependency(formula_id: UUID) -> None:
        dependency = session.get(Formula, formula_id)
        if dependency is None:
            unresolved.add("missing_dependency_formula")
            return
        if not dependency.group_key:
            dependency_ids.add(dependency.id)
            return
        parts = list(
            session.scalars(
                select(Formula).where(
                    Formula.document_id == dependency.document_id,
                    Formula.group_key == dependency.group_key,
                )
            )
        )
        for missing in missing_formula_group_parts(parts):
            unresolved.add(f"missing_group_part:{missing}")
        dependency_ids.update(item.id for item in parts)

    references = list(
        session.scalars(
            select(FormulaReference).where(FormulaReference.source_formula_id.in_(requested))
        )
    ) if requested else []
    for reference in references:
        if reference.target_formula_id is None or reference.resolution_status != "resolved":
            unresolved.add(f"unresolved_formula_reference:{reference.referenced_number}")
        else:
            add_dependency(reference.target_formula_id)
    edges = list(
        session.scalars(select(DerivationEdge).where(DerivationEdge.target_formula_id.in_(requested)))
    ) if requested else []
    for edge in edges:
        if edge.source_formula_id is None or edge.resolution_status != "resolved":
            unresolved.add("unresolved_derivation_edge")
        else:
            add_dependency(edge.source_formula_id)

    for formula in formulas:
        if formula.group_key:
            parts = list(
                session.scalars(
                    select(Formula).where(
                        Formula.document_id == formula.document_id,
                        Formula.group_key == formula.group_key,
                    )
                )
            )
            for missing in missing_formula_group_parts(parts):
                unresolved.add(f"missing_group_part:{missing}")

    evidence_ids = set(requested) | dependency_ids
    variables = list(
        session.scalars(
            select(VariableDefinition)
            .where(VariableDefinition.formula_id.in_(evidence_ids))
            .order_by(VariableDefinition.source_page, VariableDefinition.symbol)
        )
    ) if evidence_ids else []
    conditions = list(
        session.scalars(
            select(ApproximationCondition)
            .where(ApproximationCondition.formula_id.in_(evidence_ids))
            .order_by(ApproximationCondition.source_page, ApproximationCondition.id)
        )
    ) if evidence_ids else []
    if route == FormulaQueryRoute.CALCULATE_OR_DERIVE and not variables:
        unresolved.add("missing_variable_definitions")
    citation_formulas = list(
        session.scalars(
            select(Formula).where(Formula.id.in_(evidence_ids)).order_by(Formula.page_number, Formula.id)
        )
    ) if evidence_ids else []
    return FormulaDependencyBundle(
        route=route,
        target_formula_ids=requested,
        dependency_formula_ids=tuple(sorted(dependency_ids, key=str)),
        variable_definitions=tuple(
            {
                "formula_id": item.formula_id,
                "symbol": item.symbol,
                "definition": item.definition,
                "page_number": item.source_page,
                "evidence_text": item.evidence_text,
            }
            for item in variables
        ),
        approximation_conditions=tuple(
            {
                "formula_id": item.formula_id,
                "condition_text": item.condition_text,
                "page_number": item.source_page,
                "evidence_text": item.evidence_text,
            }
            for item in conditions
        ),
        citations=tuple(
            {
                "formula_id": item.id,
                "page_number": item.page_number,
                "context_before": item.context_before,
                "context_after": item.context_after,
            }
            for item in citation_formulas
        ),
        unresolved=tuple(sorted(unresolved)),
        complete=not unresolved,
    )
