from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.services.formula_assets import refresh_formula_source_crop_hashes
from paper_rag.services.formula_dependencies import rebuild_formula_dependency_graph
from paper_rag.services.formula_quality import run_formula_quality_acceptance


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"


def _quality_session(tmp_path: Path) -> tuple[Session, list[Document], list[Formula]]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    documents: list[Document] = []
    formulas: list[Formula] = []
    for index, pdf_type in enumerate(("text", "scan", "mixed")):
        stored_pdf = tmp_path / f"quality-{index}.pdf"
        shutil.copyfile(SAMPLE_PDF, stored_pdf)
        document = Document(
            original_filename=f"quality-{index}.pdf",
            stored_path=str(stored_pdf),
            file_sha256=f"{index + 1:064x}",
            status=DocumentStatus.COMPLETED,
            page_count=9,
            pdf_type=pdf_type,
            formula_index_status=FormulaIndexStatus.NEEDS_REVIEW,
            formula_parser_version="formula-layout-v3",
        )
        session.add(document)
        session.flush()
        formula_id = uuid4()
        formula = Formula(
            id=formula_id,
            document_id=document.id,
            page_number=4,
            placeholder=f"公式_placeholder_{formula_id}",
            bbox_json="[37.5, 182.0, 289.0, 274.0]",
            raw_text=f"x_{index} = y_{index} (1)",
            formula_number="1",
            group_key="equation-1",
            parser_version="formula-layout-v3",
            normalized_text=f"x_{index} = y_{index} (1)",
            fidelity_status="needs_review",
        )
        session.add(formula)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                vector_id=f"quality-{index}",
                content=f"{formula.placeholder}",
                page_start=4,
                page_end=4,
                chunk_index=0,
                formula_ids_json=json.dumps([str(formula.id)]),
            )
        )
        session.commit()
        rebuild_formula_dependency_graph(session, document.id)
        refresh_formula_source_crop_hashes(session, document.id)
        documents.append(document)
        formulas.append(formula)
    return session, documents, formulas


def test_quality_acceptance_is_stratified_deterministic_and_passes_clean_sample(tmp_path: Path) -> None:
    session, documents, formulas = _quality_session(tmp_path)

    first = run_formula_quality_acceptance(
        session,
        min_documents=3,
        min_formulas=3,
        seed=20260714,
    )
    second = run_formula_quality_acceptance(
        session,
        min_documents=3,
        min_formulas=3,
        seed=20260714,
    )

    assert first.passed is True
    assert first.sampled_document_count == 3
    assert first.sampled_formula_count == 3
    assert set(first.strata_counts) == {"mixed", "scan", "text"}
    assert first.signature == second.signature
    assert first.sampled_formula_ids == second.sampled_formula_ids


def test_quality_acceptance_fails_instead_of_lowering_insufficient_sample_threshold(tmp_path: Path) -> None:
    session, documents, formulas = _quality_session(tmp_path)

    result = run_formula_quality_acceptance(
        session,
        min_documents=30,
        min_formulas=100,
        seed=20260714,
    )

    assert result.passed is False
    assert {item.code for item in result.issues} >= {
        "insufficient_document_sample",
        "insufficient_formula_sample",
    }


def test_quality_acceptance_blocks_stale_parser_and_crop_hash_tampering(tmp_path: Path) -> None:
    session, documents, formulas = _quality_session(tmp_path)
    formulas[0].parser_version = "legacy-v1"
    formulas[1].source_crop_sha256 = "0" * 64
    session.commit()

    result = run_formula_quality_acceptance(
        session,
        min_documents=3,
        min_formulas=3,
        seed=20260714,
    )

    assert result.passed is False
    p0_codes = {item.code for item in result.issues if item.severity == "P0"}
    assert "old_parser_version" in p0_codes
    assert "source_crop_hash_mismatch" in p0_codes
