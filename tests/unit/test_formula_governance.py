from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page
import pytest

from paper_rag.services.formula_governance import (
    assert_current_formula_records,
    derive_formula_index_status,
    mark_stale_formula_indexes,
    scan_formula_inventory,
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _formula(
    document_id,
    *,
    page_number: int = 1,
    raw_text: str = "x = y (1)",
    formula_number: str | None = "1",
    group_key: str | None = "equation-1",
    bbox: list[float] | str = [10, 10, 100, 30],
    parser_version: str = "formula-layout-v3",
    physical_meaning: str | None = None,
) -> Formula:
    formula_id = uuid4()
    return Formula(
        id=formula_id,
        document_id=document_id,
        page_number=page_number,
        placeholder=f"公式_placeholder_{formula_id}",
        bbox_json=bbox if isinstance(bbox, str) else json.dumps(bbox),
        raw_text=raw_text,
        formula_number=formula_number,
        group_key=group_key,
        parser_version=parser_version,
        physical_meaning=physical_meaning,
    )


def test_inventory_detects_version_structure_geometry_and_context_anomalies() -> None:
    session = _session()
    document = Document(
        original_filename="generic-paper.pdf",
        stored_path="missing.pdf",
        file_sha256="1" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=2,
    )
    session.add(document)
    session.flush()
    session.add_all(
        [
            Page(document_id=document.id, page_number=1, text="equations"),
            Page(document_id=document.id, page_number=2, text="more equations"),
        ]
    )
    legacy = _formula(
        document.id,
        raw_text=": (2)",
        formula_number=None,
        group_key=None,
        parser_version="legacy-v1",
    )
    duplicate_a = _formula(document.id, formula_number="3", group_key="equation-3")
    duplicate_b = _formula(
        document.id,
        formula_number="3",
        group_key="equation-3-copy",
        bbox=[12, 11, 98, 29],
    )
    invalid = _formula(document.id, page_number=2, bbox="[10, 20, 10, 20]")
    misbound = _formula(
        document.id,
        page_number=2,
        formula_number="6",
        group_key="equation-6",
        bbox=[20, 40, 120, 60],
        physical_meaning="Equation (7) gives the wavelength.",
    )
    session.add_all([legacy, duplicate_a, duplicate_b, invalid, misbound])
    session.commit()

    report = scan_formula_inventory(session)
    codes = {item.code for item in report.anomalies}

    assert {
        "old_parser_version",
        "missing_formula_number",
        "missing_formula_group",
        "truncated_formula_text",
        "invalid_bbox",
        "duplicate_formula_number",
        "overlapping_bbox",
        "cross_formula_context_binding",
    } <= codes
    assert report.document_count == 1
    assert report.formula_count == 5
    assert report.signature == scan_formula_inventory(session).signature


def test_inventory_detects_formula_like_chunks_without_records_and_stale_links() -> None:
    session = _session()
    document = Document(
        original_filename="another-paper.pdf",
        stored_path="missing.pdf",
        file_sha256="2" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=2,
    )
    session.add(document)
    session.flush()
    session.add_all(
        [
            Page(document_id=document.id, page_number=1, text="E = mc^2 (1)"),
            Page(document_id=document.id, page_number=2, text="sigma = integral (2)"),
        ]
    )
    formula = _formula(document.id, page_number=1)
    session.add(formula)
    session.flush()
    session.add_all(
        [
            Chunk(
                document_id=document.id,
                vector_id="vector-1",
                content="E = mc^2 (1)",
                page_start=1,
                page_end=1,
                chunk_index=0,
                formula_ids_json=json.dumps([str(uuid4())]),
            ),
            Chunk(
                document_id=document.id,
                vector_id="vector-2",
                content="sigma = integral (2)",
                page_start=2,
                page_end=2,
                chunk_index=1,
                formula_ids_json="[]",
            ),
        ]
    )
    session.commit()

    report = scan_formula_inventory(session)
    codes = {item.code for item in report.anomalies}

    assert "stale_chunk_formula_ids" in codes
    assert "formula_like_chunk_without_formula" in codes
    assert not session.dirty
    assert not session.new


def test_new_formula_records_must_use_the_current_parser_version() -> None:
    document_id = uuid4()
    current = _formula(document_id)
    legacy = _formula(document_id, parser_version="legacy-v1")

    assert_current_formula_records([current])
    with pytest.raises(ValueError, match="legacy-v1"):
        assert_current_formula_records([current, legacy])


def test_formula_index_status_is_independent_from_document_ingestion_status() -> None:
    session = _session()
    ready = Document(
        original_filename="ready.pdf",
        stored_path="ready.pdf",
        file_sha256="8" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        formula_index_status=FormulaIndexStatus.BUILDING,
    )
    review = Document(
        original_filename="review.pdf",
        stored_path="review.pdf",
        file_sha256="9" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        formula_index_status=FormulaIndexStatus.BUILDING,
    )
    session.add_all([ready, review])
    session.flush()
    ready_formula = _formula(ready.id, parser_version="formula-layout-v3")
    ready_formula.fidelity_status = "source_exact"
    session.add(ready_formula)
    review_formula = _formula(review.id, parser_version="formula-layout-v3")
    review_formula.fidelity_status = "needs_review"
    session.add(review_formula)
    session.commit()

    ready_status = derive_formula_index_status(session, ready.id)
    review_status = derive_formula_index_status(session, review.id)

    assert ready_status == FormulaIndexStatus.READY
    assert review_status == FormulaIndexStatus.NEEDS_REVIEW
    assert ready.status == DocumentStatus.COMPLETED
    assert review.status == DocumentStatus.COMPLETED


def test_parser_version_bump_marks_old_formula_indexes_stale_idempotently() -> None:
    session = _session()
    stale = Document(
        original_filename="stale.pdf",
        stored_path="stale.pdf",
        file_sha256="a" * 63 + "b",
        status=DocumentStatus.COMPLETED,
        page_count=1,
        formula_index_status=FormulaIndexStatus.READY,
        formula_parser_version="formula-layout-v2",
    )
    current = Document(
        original_filename="current.pdf",
        stored_path="current.pdf",
        file_sha256="b" * 64,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        formula_index_status=FormulaIndexStatus.READY,
        formula_parser_version="formula-layout-v3",
    )
    session.add_all([stale, current])
    session.commit()

    changed = mark_stale_formula_indexes(session, current_parser_version="formula-layout-v3")
    repeated = mark_stale_formula_indexes(session, current_parser_version="formula-layout-v3")

    assert changed == 1
    assert repeated == 0
    assert stale.formula_index_status == FormulaIndexStatus.STALE
    assert current.formula_index_status == FormulaIndexStatus.READY
