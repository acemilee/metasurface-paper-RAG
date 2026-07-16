import pytest

from paper_rag.services.deepseek import _parse_model_answer, _parse_single_claim_audit


@pytest.mark.parametrize(
    "payload",
    [
        '{"verdict":"entailed","reason":"supported"}',
        '{"result":{"verdict":"entailed","reason":"supported","claim_index":0}}',
        '{"results":[{"verdict":"entailed","reason":"supported","claim_index":0}]}',
    ],
)
def test_single_claim_audit_accepts_common_response_wrappers(payload: str) -> None:
    result = _parse_single_claim_audit(payload)

    assert result.verdict == "entailed"
    assert result.supported_scope == ""
    assert result.unsupported_parts == []


def test_model_answer_accepts_nested_wrapper_and_normalizes_enums() -> None:
    citation_id = "38dabe2d-0e27-4c5d-9ec3-9cf19ed30ee5"
    result = _parse_model_answer(
        '{"result":{"answer":"supported","citation_ids":["' + citation_id + '"],'
        '"hallucination_risk":"unknown","novelty_claims":[{"claim":"a claim",'
        '"citation_id":"' + citation_id + '","claim_strength":"strong"}]}}'
    )

    assert result.hallucination_risk == "high"
    assert result.novelty_claims[0].claim_strength == "synthesized"


def test_single_claim_audit_normalizes_common_verdict_alias() -> None:
    result = _parse_single_claim_audit(
        '{"verdict":"entailment","reason":"The context supports the claim."}'
    )

    assert result.verdict == "entailed"


def test_model_answer_normalizes_common_citation_and_claim_shapes() -> None:
    citation_id = "38dabe2d-0e27-4c5d-9ec3-9cf19ed30ee5"
    result = _parse_model_answer(
        '{"answer":["claim one"],"citations":[{"chunk_id":"' + citation_id + '"}],'
        '"hallucination_risk":"low","claims":[{"text":"a supported claim",'
        '"citation":{"id":"' + citation_id + '"},"quote":"x","strength":"explicit"}]}'
    )

    assert result.answer == "claim one"
    assert str(result.citation_ids[0]) == citation_id
    assert result.novelty_claims == []
    assert result.claims[0].text == "a supported claim"
    assert result.claims[0].claim_type == "synthesized_fact"
