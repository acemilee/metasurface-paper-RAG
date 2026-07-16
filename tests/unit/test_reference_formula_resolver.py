from __future__ import annotations

from paper_rag.services.references import (
    parse_typed_references,
    resolve_formula_reference,
)
from paper_rag.services.references.types import ResolutionStatus
from tests.unit.reference_test_support import make_chunk, make_formula


def test_formula_five_resolves_by_formula_number_not_placeholder(
    session,
    document,
) -> None:
    formula = make_formula(
        document,
        number="5",
        placeholder="公式_placeholder_deadbeef",
    )
    chunk = make_chunk(
        document,
        formula_ids=[formula.id],
        content=formula.placeholder,
    )
    session.add_all([formula, chunk])
    session.commit()
    reference = parse_typed_references("公式5讲了什么")[0]

    result = resolve_formula_reference(session, reference, [document.id])

    assert result.status == ResolutionStatus.RESOLVED
    assert result.target_ids == (formula.id,)
    assert result.evidence_chunk_ids == (chunk.id,)
    assert result.resolution_source == "formula.formula_number"


def test_missing_formula_number_is_not_found(session, document) -> None:
    result = resolve_formula_reference(
        session,
        parse_typed_references("公式999")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.NOT_FOUND


def test_same_formula_number_in_two_documents_is_ambiguous(
    session,
    documents,
) -> None:
    session.add_all([make_formula(item, number="5") for item in documents[:2]])
    session.commit()
    result = resolve_formula_reference(
        session,
        parse_typed_references("公式5")[0],
        [item.id for item in documents[:2]],
    )
    assert result.status == ResolutionStatus.AMBIGUOUS
    assert len(result.target_ids) == 2


def test_base_number_resolves_one_complete_formula_group(session, document) -> None:
    parts = [
        make_formula(
            document,
            number=number,
            group_key="equation-1",
            part_index=index,
        )
        for index, number in enumerate(("1a", "1b", "1c"))
    ]
    session.add_all(parts)
    session.flush()
    session.add_all(
        make_chunk(document, formula_ids=[item.id], content=item.placeholder)
        for item in parts
    )
    session.commit()

    result = resolve_formula_reference(
        session,
        parse_typed_references("公式1")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.RESOLVED
    assert result.target_ids == tuple(item.id for item in parts)


def test_incomplete_base_group_is_index_inconsistent(session, document) -> None:
    session.add_all(
        [
            make_formula(
                document,
                number="1a",
                group_key="equation-1",
                part_index=0,
            ),
            make_formula(
                document,
                number="1c",
                group_key="equation-1",
                part_index=2,
            ),
        ]
    )
    session.commit()
    result = resolve_formula_reference(
        session,
        parse_typed_references("公式1")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.INDEX_INCONSISTENT
    assert result.diagnostics["missing_parts"] == ["1b"]


def test_stale_formula_document_returns_stale(session, document) -> None:
    document.formula_index_status = "stale"
    formula = make_formula(document, number="5")
    session.add(formula)
    session.commit()
    result = resolve_formula_reference(
        session,
        parse_typed_references("公式5")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.STALE
    assert result.target_ids == (formula.id,)
    assert result.diagnostics["page_numbers"] == [formula.page_number]


def test_existing_formula_without_chunk_link_is_index_inconsistent(
    session,
    document,
) -> None:
    formula = make_formula(document, number="5")
    session.add(formula)
    session.commit()
    result = resolve_formula_reference(
        session,
        parse_typed_references("公式5")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.INDEX_INCONSISTENT


def test_out_of_range_formula_identifier_is_invalid_not_prefix_matched(
    session,
    document,
) -> None:
    reference = parse_typed_references("公式1000")[0]
    assert reference.normalized_key == "1000"

    result = resolve_formula_reference(session, reference, [document.id])

    assert result.status == ResolutionStatus.INVALID
