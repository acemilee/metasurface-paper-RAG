from __future__ import annotations

from uuid import uuid4

from paper_rag.services.references import (
    parse_typed_references,
    resolve_structure_reference,
)
from paper_rag.services.references.types import ResolutionStatus
from tests.unit.reference_test_support import (
    make_chunk,
    make_page,
    make_ready_profile,
)


def test_page_reference_resolves_original_chunks(session, document) -> None:
    page = make_page(document, page_number=8)
    chunk = make_chunk(
        document,
        page_start=8,
        page_end=8,
        content="page eight evidence",
    )
    session.add_all([page, chunk])
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references("第8页讲了什么")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.RESOLVED
    assert result.target_ids == (page.id,)
    assert result.evidence_chunk_ids == (chunk.id,)


def test_section_reference_resolves_section_path_chunks(session, document) -> None:
    chunk = make_chunk(
        document,
        section_path="4.2 Optimization",
        content="section evidence",
    )
    session.add(chunk)
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("第4.2节")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.RESOLVED
    assert result.evidence_chunk_ids == (chunk.id,)


def test_section_reference_across_two_documents_is_ambiguous(
    session,
    documents,
) -> None:
    session.add_all(
        make_chunk(item, section_path="4.2 Results")
        for item in documents[:2]
    )
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("Section 4.2")[0],
        [item.id for item in documents[:2]],
    )
    assert result.status == ResolutionStatus.AMBIGUOUS


def test_figure_reference_uses_profile_for_navigation_but_returns_original_chunk(
    session,
    document,
) -> None:
    chunk = make_chunk(
        document,
        content="Figure 3(b) shows the measured gain.",
    )
    profile = make_ready_profile(
        document,
        figure_table_index=[
            {
                "caption": "Figure 3(b) shows the measured gain",
                "chunk_id": str(chunk.id),
                "page_start": 6,
                "page_end": 6,
            }
        ],
    )
    session.add_all([chunk, profile])
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("Figure 3(b)")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.RESOLVED
    assert result.evidence_chunk_ids == (chunk.id,)
    assert result.target_ids == (chunk.id,)
    assert result.resolution_source == "paper_profile.figure_table_index+chunk"


def test_table_roman_number_resolves_original_caption_chunk(
    session,
    document,
) -> None:
    chunk = make_chunk(
        document,
        content="Table II compares the measured gain.",
    )
    session.add(chunk)
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("表Ⅱ")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.RESOLVED
    assert result.evidence_chunk_ids == (chunk.id,)


def test_same_figure_in_two_documents_is_ambiguous(session, documents) -> None:
    session.add_all(
        make_chunk(document, content="Figure 3 shows the unit cell.")
        for document in documents[:2]
    )
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("Figure 3")[0],
        [document.id for document in documents[:2]],
    )
    assert result.status == ResolutionStatus.AMBIGUOUS
    assert len(result.document_ids) == 2


def test_profile_caption_with_missing_chunk_is_index_inconsistent(
    session,
    document,
) -> None:
    missing_chunk_id = uuid4()
    profile = make_ready_profile(
        document,
        figure_table_index=[
            {
                "caption": "Figure 3 shows the unit cell",
                "chunk_id": str(missing_chunk_id),
                "page_start": 2,
                "page_end": 2,
            }
        ],
    )
    session.add(profile)
    session.commit()
    result = resolve_structure_reference(
        session,
        parse_typed_references("Figure 3")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.INDEX_INCONSISTENT
    assert result.diagnostics["missing_chunk_ids"] == [str(missing_chunk_id)]


def test_out_of_range_figure_identifier_is_invalid_not_prefix_matched(
    session,
    document,
) -> None:
    reference = parse_typed_references("Figure 1000")[0]
    assert reference.normalized_key == "1000"

    result = resolve_structure_reference(session, reference, [document.id])

    assert result.status == ResolutionStatus.INVALID


def test_zero_page_identifier_is_invalid(session, document) -> None:
    result = resolve_structure_reference(
        session,
        parse_typed_references("第0页")[0],
        [document.id],
    )
    assert result.status == ResolutionStatus.INVALID


def test_explicit_document_reference_resolves_original_filename(
    session,
    document,
) -> None:
    document.original_filename = "Alpha Metasurface.pdf"
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references("论文《alpha metasurface.pdf》")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.RESOLVED
    assert result.target_ids == (document.id,)
    assert result.resolution_source == "document.original_filename"


def test_same_explicit_filename_in_two_documents_is_ambiguous(
    session,
    documents,
) -> None:
    for document in documents[:2]:
        document.original_filename = "same-name.pdf"
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references('document "same-name.pdf"')[0],
        [document.id for document in documents[:2]],
    )

    assert result.status == ResolutionStatus.AMBIGUOUS
    assert len(result.target_ids) == 2


def test_body_mention_is_not_misclassified_as_figure_caption(
    session,
    document,
) -> None:
    session.add(
        make_chunk(
            document,
            content="The method follows Figure 3 for additional context.",
        )
    )
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references("Figure 3")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.NOT_FOUND


def test_profile_navigation_requires_matching_reference_in_original_chunk(
    session,
    document,
) -> None:
    chunk = make_chunk(document, content="unrelated original evidence")
    profile = make_ready_profile(
        document,
        figure_table_index=[
            {
                "caption": "Table IX summarizes the comparison",
                "chunk_id": str(chunk.id),
                "page_start": 4,
                "page_end": 4,
            }
        ],
    )
    session.add_all([chunk, profile])
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references("Table IX")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.INDEX_INCONSISTENT
    assert result.diagnostics["profile_chunk_reference_mismatch"] == [str(chunk.id)]


def test_profile_mismatch_recovers_from_deterministic_caption_fallback(
    session,
    document,
) -> None:
    wrong = make_chunk(document, content="unrelated original evidence")
    correct = make_chunk(
        document,
        content="Figure 3 shows the measured response.",
        chunk_index=1,
    )
    profile = make_ready_profile(
        document,
        figure_table_index=[
            {
                "caption": "Figure 3 shows the measured response",
                "chunk_id": str(wrong.id),
                "page_start": 4,
                "page_end": 4,
            }
        ],
    )
    session.add_all([wrong, correct, profile])
    session.commit()

    result = resolve_structure_reference(
        session,
        parse_typed_references("Figure 3")[0],
        [document.id],
    )

    assert result.status == ResolutionStatus.RESOLVED
    assert result.target_ids == (correct.id,)
    assert result.resolution_source == "chunk.caption_fallback"
    assert result.diagnostics["profile_chunk_reference_mismatch"] == [str(wrong.id)]
