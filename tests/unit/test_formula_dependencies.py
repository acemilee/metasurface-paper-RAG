from __future__ import annotations

from uuid import uuid4

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import (
    ApproximationCondition,
    DerivationEdge,
    FormulaGroup,
    FormulaReference,
    VariableDefinition,
)
from paper_rag.services.formula_dependencies import (
    FormulaQueryRoute,
    build_formula_dependency_bundle,
    rebuild_formula_dependency_graph,
)
from paper_rag.services.formula_query_guard import route_formula_query


def _session() -> tuple[Session, Document]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    document = Document(
        original_filename="dependency-paper.pdf",
        stored_path="dependency-paper.pdf",
        file_sha256="c" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=3,
    )
    session.add(document)
    session.commit()
    return session, document


def _formula(
    document: Document,
    number: str,
    group_key: str,
    part_index: int,
    *,
    context_before: str = "",
    context_after: str = "",
) -> Formula:
    formula_id = uuid4()
    return Formula(
        id=formula_id,
        document_id=document.id,
        page_number=1 if number.startswith("1") else 2,
        placeholder=f"公式_placeholder_{formula_id}",
        bbox_json=f"[10, {10 + part_index * 30}, 200, {30 + part_index * 30}]",
        raw_text=f"formula body ({number})",
        formula_number=number,
        group_key=group_key,
        part_index=part_index,
        parser_version="formula-layout-v3",
        fidelity_status="source_exact",
        context_before=context_before,
        context_after=context_after,
    )


def test_dependency_graph_builds_groups_references_variables_conditions_and_edges() -> None:
    session, document = _session()
    repeated_context = (
        "where sigma denotes the sheet conductivity. "
        "Under the low-temperature approximation, losses are neglected."
    )
    part_a = _formula(
        document,
        "1a",
        "equation-1",
        0,
        context_before=repeated_context,
        context_after=repeated_context,
    )
    part_b = _formula(document, "1b", "equation-1", 1)
    derived = _formula(
        document,
        "2",
        "equation-2",
        0,
        context_before="Using Eq. (1a), Equation (2) is derived.",
    )
    session.add_all([part_a, part_b, derived])
    session.commit()

    report = rebuild_formula_dependency_graph(session, document.id)
    bundle = build_formula_dependency_bundle(
        session,
        [derived.id],
        FormulaQueryRoute.EXPLAIN,
    )

    assert report.group_count == 2
    assert session.scalar(select(func.count(FormulaGroup.id))) == 2
    assert session.scalar(select(func.count(FormulaReference.id))) >= 1
    assert session.scalar(select(func.count(VariableDefinition.id))) == 1
    assert session.scalar(select(func.count(ApproximationCondition.id))) == 1
    assert session.scalar(select(func.count(DerivationEdge.id))) >= 1
    assert part_a.formula_group_id == part_b.formula_group_id
    assert part_a.id in bundle.dependency_formula_ids
    assert bundle.complete is True
    assert bundle.unresolved == ()


def test_dependency_bundle_fails_closed_for_missing_group_part_and_unresolved_reference() -> None:
    session, document = _session()
    part_a = _formula(document, "1a", "equation-1", 0)
    part_c = _formula(document, "1c", "equation-1", 2)
    target = _formula(
        document,
        "2",
        "equation-2",
        0,
        context_before="Using Eq. (9), Equation (2) follows.",
    )
    session.add_all([part_a, part_c, target])
    session.commit()

    rebuild_formula_dependency_graph(session, document.id)
    group_bundle = build_formula_dependency_bundle(
        session,
        [part_a.id],
        FormulaQueryRoute.SOURCE_RENDER,
    )
    target_bundle = build_formula_dependency_bundle(
        session,
        [target.id],
        FormulaQueryRoute.EXPLAIN,
    )

    assert group_bundle.complete is False
    assert "missing_group_part:1b" in group_bundle.unresolved
    assert target_bundle.complete is False
    assert "unresolved_formula_reference:9" in target_bundle.unresolved


def test_base_equation_reference_resolves_to_all_parts_of_the_formula_group() -> None:
    session, document = _session()
    part_a = _formula(document, "1a", "equation-1", 0)
    part_b = _formula(document, "1b", "equation-1", 1)
    target = _formula(
        document,
        "2",
        "equation-2",
        0,
        context_before="Using Eq. (1), Equation (2) follows.",
    )
    session.add_all([part_a, part_b, target])
    session.commit()

    report = rebuild_formula_dependency_graph(session, document.id)
    bundle = build_formula_dependency_bundle(
        session,
        [target.id],
        FormulaQueryRoute.EXPLAIN,
    )

    assert report.unresolved_count == 0
    assert set(bundle.dependency_formula_ids) == {part_a.id, part_b.id}
    assert bundle.complete is True


def test_dependency_resolution_never_crosses_document_boundary() -> None:
    session, document = _session()
    other = Document(
        original_filename="other.pdf",
        stored_path="other.pdf",
        file_sha256="d" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=2,
    )
    session.add(other)
    session.flush()
    foreign_formula = _formula(other, "7", "equation-7", 0)
    target = _formula(
        document,
        "2",
        "equation-2",
        0,
        context_before="Using Eq. (7), Equation (2) follows.",
    )
    session.add_all([foreign_formula, target])
    session.commit()

    rebuild_formula_dependency_graph(session, document.id)
    reference = session.scalar(select(FormulaReference))

    assert reference is not None
    assert reference.target_formula_id is None
    assert reference.resolution_status == "unresolved"


def test_formula_query_routes_are_explicit_and_do_not_collapse_to_explain() -> None:
    assert route_formula_query("extract") == FormulaQueryRoute.SOURCE_RENDER
    assert route_formula_query("synthesize") == FormulaQueryRoute.EXPLAIN
    assert route_formula_query("derive") == FormulaQueryRoute.CALCULATE_OR_DERIVE
    assert route_formula_query("compare") == FormulaQueryRoute.COMPARE
