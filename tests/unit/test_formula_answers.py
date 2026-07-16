from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import fitz
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.document import Document, DocumentStatus, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.models.formula_governance import FormulaBackfillJob
from paper_rag.services.formula_dependencies import (
    FormulaQueryRoute,
    rebuild_formula_dependency_graph,
)
from paper_rag.services.formula_answers import build_direct_formula_response
from paper_rag.services.formula_query_guard import guard_formula_query
from paper_rag.services.query_rewrite import parse_query_plan
from paper_rag.services.retrieval import RetrievedChunk


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _build_formula_fixture(
    tmp_path: Path,
    *,
    add_formulas: bool,
) -> tuple[Session, Document, list[RetrievedChunk], object, object]:
    session = _session()
    pdf_path = tmp_path / "graphene.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((30, 30), "Formula source")
        pdf.save(pdf_path)
    document = Document(
        original_filename="graphene.pdf",
        stored_path=str(pdf_path),
        file_sha256="a" * 64,
        page_count=9,
        status=DocumentStatus.COMPLETED,
        formula_index_status=FormulaIndexStatus.NEEDS_REVIEW,
        formula_parser_version="formula-layout-v3",
    )
    session.add(document)
    session.flush()
    if add_formulas:
        formulas = []
        for part_index, number in enumerate(("1a", "1b", "1c")):
            formula_id = uuid4()
            formulas.append(
                Formula(
                    id=formula_id,
                    document_id=document.id,
                    page_number=1,
                    placeholder=f"公式_placeholder_{formula_id}",
                    bbox_json=json.dumps([10, 20 + part_index * 20, 200, 35 + part_index * 20]),
                    raw_text=f"source glyphs ({number})",
                    context_before="given by the well-established Kubo formula",
                    context_after="where variables are defined",
                    formula_number=number,
                    group_key="equation-1",
                    part_index=part_index,
                    parser_version="formula-layout-v3",
                    normalized_text=f"source glyphs ({number})",
                    fidelity_status="needs_review",
                )
            )
        session.add_all(formulas)
    session.commit()
    if add_formulas:
        rebuild_formula_dependency_graph(session, document.id)
    chunk_id = uuid4()
    evidence = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document.id,
            content="The sheet resistance is given by the Kubo formula.",
            page_start=1,
            page_end=1,
            section_path="3. Theoretical model",
            formula_ids=[],
            score=0.9,
        )
    ]

    formula_plan = parse_query_plan(
        '{"intent":"formula_explanation","answer_mode":"extract","standalone_question":"Kubo公式是什么",'
        '"retrieval_queries":[{"query":"Kubo formula","evidence_type":"formula_context"}],'
        '"entities":[],"required_evidence":["formula_context"],"scope_requirement":"single_document",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )
    method_plan = parse_query_plan(
        '{"intent":"method_mechanism","answer_mode":"synthesize","standalone_question":"方法是什么",'
        '"retrieval_queries":[{"query":"method","evidence_type":"method_or_structure"}],'
        '"entities":[],"required_evidence":["method_or_structure"],"scope_requirement":"single_document",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )
    return session, document, evidence, formula_plan, method_plan


def test_formula_extract_returns_source_images_without_llm_reconstruction(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )

    response = build_direct_formula_response(
        session,
        "Kubo公式是什么？直接给出公式",
        evidence,
        formula_plan,
    )

    assert response is not None
    assert response.action == "answer"
    assert response.audit_result == "formula_source_rendered"
    assert response.citations[0].citation_id == evidence[0].chunk_id
    assert [asset.formula_number for asset in response.formula_assets] == ["1a", "1b", "1c"]
    assert all(asset.image_url.endswith("/image") for asset in response.formula_assets)
    assert "原始公式图像" in response.answer
    assert build_direct_formula_response(session, "方法是什么", evidence, method_plan) is None


def test_formula_extract_returns_not_extracted_instead_of_falling_back_to_llm(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=False,
    )

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None
    assert response.action == "refuse"
    assert response.audit_result == "formula_not_extracted"


def test_formula_extract_rejects_invalid_source_region(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    formula = session.query(Formula).first()
    formula.bbox_json = "[10, 20, 10, 20]"
    session.commit()

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None
    assert response.action == "refuse"
    assert response.audit_result == "formula_text_corrupted"


def test_formula_extract_rejects_bbox_outside_pdf_page(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    for formula in session.query(Formula):
        formula.bbox_json = "[1000, 1000, 1100, 1100]"
    session.commit()

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None
    assert response.action == "refuse"
    assert response.audit_result == "formula_text_corrupted"


def test_formula_extract_rejects_page_outside_pdf(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    for formula in session.query(Formula):
        formula.page_number = 99
    evidence[0] = RetrievedChunk(
        evidence[0].chunk_id,
        evidence[0].document_id,
        evidence[0].content,
        99,
        99,
        evidence[0].section_path,
        evidence[0].formula_ids,
        evidence[0].score,
    )
    session.commit()

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None
    assert response.action == "refuse"
    assert response.audit_result == "formula_text_corrupted"


def test_formula_extract_is_stable_for_twenty_runs(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    signatures = []
    for _ in range(20):
        response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)
        assert response is not None
        signatures.append(
            (
                response.action,
                tuple(asset.formula_id for asset in response.formula_assets),
                tuple(asset.formula_number for asset in response.formula_assets),
                tuple(item.citation_id for item in response.citations),
            )
        )

    assert len(set(signatures)) == 1


def test_formula_query_stale_index_fails_closed_and_enqueues_one_page_repair(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    document.formula_index_status = FormulaIndexStatus.STALE
    document.formula_parser_version = "legacy-v1"
    session.commit()

    first = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)
    second = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert first is not None and first.action == "refuse"
    assert first.audit_result == "formula_index_stale"
    assert second is not None and second.audit_result == "formula_index_stale"
    assert session.scalar(select(func.count(FormulaBackfillJob.id))) == 1


def test_formula_query_incomplete_group_fails_closed_and_enqueues_repair(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    missing = session.scalar(select(Formula).where(Formula.formula_number == "1b"))
    session.delete(missing)
    session.commit()
    rebuild_formula_dependency_graph(session, document.id)

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None and response.action == "refuse"
    assert response.audit_result == "formula_dependency_incomplete"
    assert "1b" in response.refusal_reason
    assert session.scalar(select(func.count(FormulaBackfillJob.id))) == 1


def test_verified_latex_is_rendered_as_primary_mathml_with_pdf_crop_for_audit(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    for index, formula in enumerate(session.scalars(select(Formula))):
        formula.latex_text = rf"\sigma_{{{index}}}=\frac{{a}}{{b}}"
        formula.latex_verification_status = "verified"
    session.commit()

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None and response.action == "answer"
    assert all(asset.rendered_mathml.startswith("<math") for asset in response.formula_assets)
    assert all(asset.latex_verification_status == "verified" for asset in response.formula_assets)
    assert all(asset.image_url.endswith("/image") for asset in response.formula_assets)
    assert "已验证 LaTeX" in response.answer


def test_unverified_or_invalid_latex_never_becomes_rendered_html(tmp_path: Path) -> None:
    session, document, evidence, formula_plan, method_plan = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    formulas = list(session.scalars(select(Formula).order_by(Formula.part_index)))
    formulas[0].latex_text = r"\frac{a}{b}"
    formulas[0].latex_verification_status = "unverified"
    formulas[1].latex_text = r"</math><script>alert(1)</script>"
    formulas[1].latex_verification_status = "verified"
    session.commit()

    response = build_direct_formula_response(session, "Kubo公式是什么", evidence, formula_plan)

    assert response is not None
    assert response.formula_assets[0].rendered_mathml is None
    assert response.formula_assets[0].latex_verification_status == "unverified"
    assert response.formula_assets[1].rendered_mathml is None
    assert response.formula_assets[1].latex_verification_status == "invalid"
    assert all("<script" not in (asset.rendered_mathml or "") for asset in response.formula_assets)


def test_direct_formula_response_uses_resolved_formula_ids_even_if_question_wording_is_opaque(
    tmp_path: Path,
) -> None:
    session, _, evidence, formula_plan, _ = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    formula = session.scalar(
        select(Formula).where(Formula.formula_number == "1b")
    )

    response = build_direct_formula_response(
        session,
        "请解释这个对象",
        evidence,
        formula_plan,
        resolved_formula_ids=[formula.id],
    )

    assert response is not None
    assert [item.formula_number for item in response.formula_assets] == ["1b"]


def test_resolved_formula_can_render_source_when_calculation_variables_are_missing(
    tmp_path: Path,
) -> None:
    session, document, evidence, formula_plan, _ = _build_formula_fixture(
        tmp_path,
        add_formulas=True,
    )
    formula = session.scalar(
        select(Formula).where(Formula.formula_number == "1a")
    )
    repair_pages = {document.id: {formula.page_number}}
    source_readiness = guard_formula_query(
        session,
        [formula],
        FormulaQueryRoute.SOURCE_RENDER,
        repair_pages=repair_pages,
    )
    calculation_readiness = guard_formula_query(
        session,
        [formula],
        FormulaQueryRoute.CALCULATE_OR_DERIVE,
        repair_pages=repair_pages,
    )
    response = build_direct_formula_response(
        session,
        "直接给出公式1a",
        evidence,
        formula_plan,
        resolved_formula_ids=[formula.id],
    )

    assert source_readiness.ready is True
    assert calculation_readiness.ready is False
    assert "missing_variable_definitions" in calculation_readiness.bundle.unresolved
    assert response is not None and response.action == "answer"
