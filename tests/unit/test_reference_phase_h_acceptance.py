from __future__ import annotations

from uuid import uuid4

from paper_rag.services.reference_quality import (
    ReferenceAcceptanceCase,
    evaluate_reference_case,
    run_reference_quality_acceptance,
)
from paper_rag.services.references.types import ResolutionStatus
from tests.unit.reference_test_support import (
    make_chunk,
    make_formula,
    make_ready_profile,
    seed_numbered_formulas,
)


def test_acceptance_generates_variants_for_every_numbered_formula(
    session,
    documents,
) -> None:
    seed_numbered_formulas(
        session,
        documents,
        numbers=("1", "1a", "5"),
    )
    result = run_reference_quality_acceptance(
        session,
        min_formula_cases=12,
        min_figure_cases=0,
        min_table_cases=0,
        min_section_cases=0,
        seed=20260715,
    )
    assert result.formula_case_count == 12
    assert result.p0_count == 0
    assert result.passed is True


def test_acceptance_checks_missing_numbers_as_not_found(session, document) -> None:
    case = ReferenceAcceptanceCase(
        question="公式999",
        document_ids=(document.id,),
        expected_status=ResolutionStatus.NOT_FOUND,
        expected_target_ids=(),
    )
    assert evaluate_reference_case(session, case) is None


def test_acceptance_checks_cross_document_same_number_as_ambiguous(
    session,
    documents,
) -> None:
    formulas = [
        make_formula(document, number="5")
        for document in documents[:2]
    ]
    session.add_all(formulas)
    session.commit()
    case = ReferenceAcceptanceCase(
        question="公式5",
        document_ids=tuple(document.id for document in documents[:2]),
        expected_status=ResolutionStatus.AMBIGUOUS,
        expected_target_ids=tuple(
            sorted((item.id for item in formulas), key=str)
        ),
    )
    assert evaluate_reference_case(session, case) is None


def test_acceptance_fails_when_structure_sample_is_below_threshold(
    session,
    document,
) -> None:
    result = run_reference_quality_acceptance(
        session,
        min_formula_cases=1,
        min_figure_cases=100,
        min_table_cases=100,
        min_section_cases=100,
        seed=20260715,
    )
    codes = {
        item.code for item in result.issues if item.severity == "P0"
    }
    assert {
        "insufficient_figure_cases",
        "insufficient_table_cases",
        "insufficient_section_cases",
    } <= codes


def test_acceptance_requires_original_chunk_for_profile_caption(
    session,
    document,
) -> None:
    profile = make_ready_profile(
        document,
        figure_table_index=[
            {
                "caption": "Figure 3 shows the unit cell",
                "chunk_id": str(uuid4()),
                "page_start": 2,
                "page_end": 2,
            }
        ],
    )
    session.add(profile)
    session.commit()
    case = ReferenceAcceptanceCase(
        question="Figure 3",
        document_ids=(document.id,),
        expected_status=ResolutionStatus.RESOLVED,
        expected_target_ids=(),
    )
    issue = evaluate_reference_case(session, case)
    assert issue is not None
    assert issue.code == "reference_status_mismatch"


def test_acceptance_counts_profile_navigation_only_when_original_chunk_exists(
    session,
    document,
) -> None:
    chunk = make_chunk(document, content="Table IX comparison evidence")
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

    result = run_reference_quality_acceptance(
        session,
        min_formula_cases=0,
        min_figure_cases=0,
        min_table_cases=1,
        min_section_cases=0,
        seed=20260715,
    )

    assert result.table_case_count == 2
    assert result.passed is True
