from __future__ import annotations

import json

import pytest

from paper_rag.models.formula_governance import FormulaBackfillJob
from paper_rag.services.references import (
    decide_reference_control,
    enqueue_reference_repairs,
    merge_resolved_reference_evidence,
    parse_typed_references,
    prepare_reference_control,
    resolve_typed_references,
)
from paper_rag.services.references.types import ReferenceKind, ResolutionStatus
from tests.unit.reference_test_support import (
    make_chunk,
    make_retrieved_chunk,
    formula_extract_plan,
    resolution_with_status,
    resolved_formula_resolution,
    seed_formula_and_page,
    seed_formula_with_chunk,
    seed_stale_formula,
    stale_resolution_for,
)


def test_registry_resolves_mixed_references_in_original_order(
    session,
    document,
) -> None:
    seed_formula_and_page(session, document, number="5", page_number=8)
    references = parse_typed_references("公式5在第8页吗")

    results = resolve_typed_references(session, references, [document.id])

    assert [item.reference.kind for item in results] == [
        ReferenceKind.FORMULA,
        ReferenceKind.PAGE,
    ]
    assert all(item.status == ResolutionStatus.RESOLVED for item in results)


def test_resolved_reference_evidence_is_pinned_ahead_of_vector_results(
    session,
    document,
) -> None:
    target = make_chunk(
        document,
        content="formula five evidence",
        chunk_index=8,
    )
    unrelated = make_retrieved_chunk(
        document,
        content="semantic but unrelated",
        score=0.99,
    )
    session.add(target)
    session.commit()
    resolution = resolved_formula_resolution(document.id, target.id)

    merged = merge_resolved_reference_evidence(
        session,
        [unrelated],
        [resolution],
    )

    assert merged[0].chunk_id == target.id
    assert {item.chunk_id for item in merged} == {
        target.id,
        unrelated.chunk_id,
    }


@pytest.mark.parametrize(
    ("status", "action", "audit_result"),
    [
        (ResolutionStatus.NOT_FOUND, "refuse", "strong_reference_not_found"),
        (ResolutionStatus.AMBIGUOUS, "clarify", "strong_reference_ambiguous"),
        (ResolutionStatus.STALE, "refuse", "strong_reference_stale"),
        (ResolutionStatus.INVALID, "clarify", "strong_reference_invalid"),
        (
            ResolutionStatus.INDEX_INCONSISTENT,
            "refuse",
            "reference_index_inconsistent",
        ),
    ],
)
def test_reference_failure_has_precise_action_and_audit_code(
    status,
    action,
    audit_result,
) -> None:
    decision = decide_reference_control([resolution_with_status(status)])
    assert (decision.action, decision.audit_result) == (action, audit_result)


def test_formula_reference_is_resolved_before_soft_entity_gate(
    session,
    document,
) -> None:
    formula, _ = seed_formula_with_chunk(session, document, number="5")
    plan = formula_extract_plan(entity_surface="公式5")

    prepared = prepare_reference_control(
        session,
        original_question="公式5讲了什么",
        query_plan=plan,
        document_scope=[document.id],
    )

    assert prepared.decision.proceed is True
    assert prepared.resolutions[0].target_ids == (formula.id,)
    assert prepared.soft_entities == ()


def test_stale_formula_resolution_enqueues_one_page_repair_idempotently(
    session,
    document,
) -> None:
    formula = seed_stale_formula(
        session,
        document,
        number="5",
        page_number=8,
    )
    resolution = stale_resolution_for(formula)

    first = enqueue_reference_repairs(session, [resolution])
    second = enqueue_reference_repairs(session, [resolution])

    assert first == second
    assert len(first) == 1
    job = session.get(FormulaBackfillJob, first[0])
    assert json.loads(job.page_numbers_json) == [8]
