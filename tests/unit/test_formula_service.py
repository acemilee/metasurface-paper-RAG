from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page, TextBlock
from paper_rag.services.formula_service import (
    create_formula_records,
    detect_formula_regions,
    rebuild_formula_pages,
)
from paper_rag.services.pdf_parser import ParsedPage, ParsedTextBlock, parse_text_pdf


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"
requires_sample_pdf = pytest.mark.skipif(
    not SAMPLE_PDF.is_file(),
    reason="private regression PDF is not distributed",
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def test_plain_resistance_paragraph_is_not_formula() -> None:
    block = ParsedTextBlock(1, 0, "The resistance Rg of graphene changes with applied voltage.", 0, 0, 10, 10)

    assert detect_formula_regions(ParsedPage(1, block.text, [block])) == []


@requires_sample_pdf
def test_kubo_multiline_equations_are_grouped_from_real_pdf_layout() -> None:
    page = parse_text_pdf(SAMPLE_PDF, uuid4()).pages[3]

    candidates = detect_formula_regions(page)
    by_number = {candidate.formula_number: candidate for candidate in candidates}

    assert {"1a", "1b", "1c", "2"}.issubset(by_number)
    assert {by_number[number].group_key for number in ("1a", "1b", "1c")} == {"equation-1"}
    assert "sintra" in by_number["1b"].raw_text
    assert "sinter" in by_number["1c"].raw_text
    assert "Rgz" in by_number["2"].raw_text
    assert "Kubo formula" in by_number["1a"].context_before
    assert "Equation (2) reveals" in by_number["2"].context_after
    assert "377" not in by_number["3"].raw_text


@requires_sample_pdf
def test_formula_records_preserve_group_number_version_and_fidelity() -> None:
    document_id = uuid4()
    page = parse_text_pdf(SAMPLE_PDF, document_id).pages[3]

    records = create_formula_records(document_id, page)
    repeated = create_formula_records(document_id, page)
    by_number = {record.formula_number: record for record in records}

    assert by_number["1a"].group_key == "equation-1"
    assert by_number["1b"].part_index == 1
    assert by_number["1c"].part_index == 2
    assert by_number["1a"].parser_version == "formula-layout-v3"
    assert by_number["1a"].fidelity_status == "needs_review"
    assert by_number["1a"].normalized_text
    assert "reflection and transmission" in by_number["3"].physical_meaning
    assert "Equation (2) reveals" not in by_number["3"].physical_meaning
    assert [record.id for record in records] == [record.id for record in repeated]


@requires_sample_pdf
def test_numbered_formula_id_survives_small_bbox_drift() -> None:
    document_id = uuid4()
    original = parse_text_pdf(SAMPLE_PDF, document_id).pages[3]
    shifted_blocks = [
        ParsedTextBlock(
            block.page_number,
            block.reading_order,
            block.text,
            block.x0 + 0.3,
            block.y0 + 0.2,
            block.x1 + 0.3,
            block.y1 + 0.2,
            block.source,
            block.confidence,
        )
        for block in original.blocks
    ]
    shifted = ParsedPage(
        4,
        original.text,
        shifted_blocks,
        original.extraction_method,
        original.quality_score,
        original.ocr_confidence,
    )
    first = {item.formula_number: item.id for item in create_formula_records(document_id, original)}
    second = {item.formula_number: item.id for item in create_formula_records(document_id, shifted)}

    assert first["1a"] == second["1a"]
    assert first["1b"] == second["1b"]
    assert first["1c"] == second["1c"]


def test_formula_context_does_not_cross_columns_or_neighbor_boundaries() -> None:
    blocks = [
        ParsedTextBlock(4, 0, "Equation (2) reveals the impedance relation.", 40, 90, 280, 108),
        ParsedTextBlock(4, 1, "Rgz = expression", 60, 120, 230, 138),
        ParsedTextBlock(4, 2, "(2)", 260, 120, 285, 138),
        ParsedTextBlock(4, 3, "where R means wrong-column value.", 330, 140, 560, 165),
        ParsedTextBlock(4, 4, "reflection and transmission coefficients are defined below.", 40, 155, 285, 176),
        ParsedTextBlock(4, 5, "r = expression", 60, 190, 230, 208),
        ParsedTextBlock(4, 6, "(3)", 260, 190, 285, 208),
    ]
    page = ParsedPage(4, "\n".join(block.text for block in blocks), blocks)
    records = {item.formula_number: item for item in create_formula_records(uuid4(), page)}

    assert "wrong-column" not in (records["2"].physical_meaning or "")
    assert "impedance relation" in (records["2"].physical_meaning or "")
    assert "reflection and transmission" in (records["3"].physical_meaning or "")
    assert "impedance relation" not in (records["3"].physical_meaning or "")


@requires_sample_pdf
def test_targeted_formula_backfill_is_idempotent_and_does_not_rebuild_chunks() -> None:
    session = _session()
    document_id = uuid4()
    parsed_page = parse_text_pdf(SAMPLE_PDF, document_id).pages[3]
    document = Document(
        id=document_id,
        original_filename=SAMPLE_PDF.name,
        stored_path=str(SAMPLE_PDF),
        file_sha256="f" * 64,
        page_count=9,
        status=DocumentStatus.COMPLETED,
    )
    page = Page(document_id=document_id, page_number=4, text=parsed_page.text)
    session.add_all([document, page])
    session.flush()
    session.add_all(
        TextBlock(
            page_id=page.id,
            reading_order=block.reading_order,
            text=block.text,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            source=block.source,
            confidence=block.confidence,
        )
        for block in parsed_page.blocks
    )
    chunk = Chunk(
        document_id=document_id,
        vector_id=f"{document_id}:existing:0",
        content=parsed_page.text,
        page_start=4,
        page_end=4,
        chunk_index=0,
    )
    session.add(chunk)
    session.commit()

    first = rebuild_formula_pages(session, document_id, [4])
    second = rebuild_formula_pages(session, document_id, [4])

    assert {record.id for record in first} == {record.id for record in second}
    assert session.scalar(select(func.count(Formula.id))) == len(first)
    assert session.scalar(select(func.count(Chunk.id))) == 1
    assert session.get(Chunk, chunk.id).content == parsed_page.text
