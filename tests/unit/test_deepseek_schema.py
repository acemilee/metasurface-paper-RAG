import pytest

from paper_rag.services.deepseek import _parse_hypothesis_generation, _parse_model_answer, _parse_novelty_entailment_audit


def test_invalid_model_json_is_rejected() -> None:
    with pytest.raises(ValueError):
        _parse_model_answer('{"answer":"missing required fields"}')


def test_cross_language_entailment_schema_parses() -> None:
    audit = _parse_novelty_entailment_audit(
        '{"answer_claims_fully_covered":true,"uncovered_answer_claims":[],"results":[{"claim_index":0,"verdict":"entailed",'
        '"reason":"The Chinese claim is an accurate translation.",'
        '"supported_scope":"wave control","unsupported_parts":[]}]}'
    )

    assert audit.results[0].verdict == "entailed"


@pytest.mark.parametrize(
    "wrapper",
    [
        '{{"hypotheses":[{item}]}}',
        '{{"hypothesis":{item}}}',
        '{item}',
    ],
)
def test_hypothesis_generation_accepts_equivalent_single_item_shapes(wrapper: str) -> None:
    item = (
        '{"claim":"The combination may change the response.",'
        '"premises":[{"claim":"Premise A","citation_ids":["38dabe2d-0e27-4c5d-9ec3-9cf19ed30ee5"]},'
        '{"claim":"Premise B","citation_ids":["44ba016f-973a-459e-aad2-542b13f85372"]}],'
        '"confidence":"medium","assumptions":["Compatibility must be verified"],'
        '"validation_needed":["Full-wave simulation"],"counterevidence":["Loss may increase"]}'
    )

    result = _parse_hypothesis_generation(wrapper.format(item=item))

    assert len(result.hypotheses) == 1


def test_hypothesis_generation_normalizes_common_field_shapes() -> None:
    result = _parse_hypothesis_generation(
        '{"hypotheses":[{"claim":"The response may change.","premises":['
        '{"text":"Premise A","citation_id":"38dabe2d-0e27-4c5d-9ec3-9cf19ed30ee5"},'
        '{"text":"Premise B","citation_id":"44ba016f-973a-459e-aad2-542b13f85372"}],'
        '"confidence":"medium","assumptions":"Compatibility must be verified",'
        '"validation_needed":"Full-wave simulation","counterevidence":"Loss may increase"}]}'
    )

    hypothesis = result.hypotheses[0]
    assert hypothesis.premises[0].claim == "Premise A"
    assert len(hypothesis.premises[0].citation_ids) == 1
    assert hypothesis.assumptions == ["Compatibility must be verified"]
    assert hypothesis.validation_needed == ["Full-wave simulation"]
    assert hypothesis.counterevidence == ["Loss may increase"]


def test_hypothesis_generation_expands_unique_citation_prefixes() -> None:
    first = "38dabe2d-0e27-4c5d-9ec3-9cf19ed30ee5"
    second = "44ba016f-973a-459e-aad2-542b13f85372"
    result = _parse_hypothesis_generation(
        '{"hypotheses":[{"conditional_claim":"The response may change.",'
        '"premise_1":{"text":"Premise A","citation_ids":["38dabe2d"]},'
        '"premise_2":{"text":"Premise B","citation_ids":["44ba016f"]},'
        '"confidence":"low","assumptions":["Compatibility is unknown"],'
        '"validation_needed":"Simulation","counterevidence":"Different operating modes"}]}',
        [first, second],
    )

    assert str(result.hypotheses[0].premises[0].citation_ids[0]) == first
    assert str(result.hypotheses[0].premises[1].citation_ids[0]) == second
