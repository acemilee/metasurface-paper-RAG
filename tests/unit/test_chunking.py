from uuid import uuid4

from paper_rag.services.chunking import build_chunks, make_vector_id
from paper_rag.models.formula import Formula
from paper_rag.services.pdf_parser import ParsedDocument, ParsedPage, ParsedTextBlock


def test_chunk_uses_parsed_document_id_and_preserves_page_metadata() -> None:
    document_id = uuid4()
    block = ParsedTextBlock(1, 0, "1. Introduction\nGraphene metasurface absorption.", 0, 0, 10, 10)
    document = ParsedDocument(document_id, 1, [ParsedPage(1, block.text, [block])])

    chunks = build_chunks(document, [], target_chars=20, overlap_chars=0)

    assert chunks[0].document_id == document_id
    assert chunks[0].page_start == 1
    assert make_vector_id(document_id, "v1", 0).endswith(":v1:0")


def test_formula_ids_do_not_leak_from_the_next_page_into_the_previous_chunk() -> None:
    document_id = uuid4()
    first_text = "A" * 80
    second_text = "B" * 40 + " x = y (1) " + "C" * 40
    document = ParsedDocument(
        document_id,
        2,
        [
            ParsedPage(1, first_text, [ParsedTextBlock(1, 0, first_text, 0, 0, 10, 10)]),
            ParsedPage(2, second_text, [ParsedTextBlock(2, 0, second_text, 0, 0, 10, 10)]),
        ],
    )
    formula_id = uuid4()
    formula = Formula(
        id=formula_id,
        document_id=document_id,
        page_number=2,
        placeholder=f"公式_placeholder_{formula_id}",
        bbox_json="[0, 0, 10, 10]",
        raw_text="x = y (1)",
    )

    chunks = build_chunks(document, [formula], target_chars=100, overlap_chars=0)

    assert chunks[0].page_start == chunks[0].page_end == 1
    assert chunks[0].formula_ids == []
    assert chunks[1].page_start == chunks[1].page_end == 2
    assert chunks[1].formula_ids == [formula_id]
    assert formula.placeholder in chunks[1].content


def test_overlap_preserves_previous_page_range_when_it_carries_a_formula() -> None:
    document_id = uuid4()
    raw_formula = "x = y (1)"
    first_text = "A" * 60 + raw_formula
    second_text = "B" * 80
    document = ParsedDocument(
        document_id,
        2,
        [
            ParsedPage(1, first_text, [ParsedTextBlock(1, 0, first_text, 0, 0, 10, 10)]),
            ParsedPage(2, second_text, [ParsedTextBlock(2, 0, second_text, 0, 0, 10, 10)]),
        ],
    )
    formula_id = uuid4()
    formula = Formula(
        id=formula_id,
        document_id=document_id,
        page_number=1,
        placeholder=f"公式_placeholder_{formula_id}",
        bbox_json="[0, 0, 10, 10]",
        raw_text=raw_formula,
    )

    chunks = build_chunks(document, [formula], target_chars=100, overlap_chars=80)

    assert formula.placeholder in chunks[1].content
    assert chunks[1].formula_ids == [formula_id]
    assert chunks[1].page_start == 1
    assert chunks[1].page_end == 2
