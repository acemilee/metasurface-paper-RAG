from uuid import uuid4

import pytest

from paper_rag.schemas.chat import Citation, ClaimEntailmentResult, DeterministicDerivation, EvidenceBoundedHypothesis, GroundedClaim, HypothesisPremise, ModelAnswer, NoveltyClaim, NoveltyEntailmentAudit
from paper_rag.services.answer_audit import make_hypothesis_refusal, make_refusal, render_derivations, render_grounded_claims, render_hypotheses, render_novelty_claims, salvage_partially_entailed_premises, verify_citations_exist, verify_claim_tokens_against_evidence, verify_cross_document_citations, verify_novelty_answer, verify_novelty_entailment_audit
from paper_rag.services.retrieval import RetrievedChunk


def test_refusal_answer_exposes_the_actual_gate_reason() -> None:
    response = make_refusal("已召回候选证据，但相关性不足以支持可靠回答")

    assert response.answer.startswith("已召回候选证据")
    assert "未检索到" not in response.answer
    assert response.action == "refuse"


def test_hypothesis_refusal_preserves_epistemic_contract() -> None:
    response = make_hypothesis_refusal(["前提未通过证据审计", "推理忽略频段冲突"])

    assert response.action == "refuse"
    assert response.audit_result == "hypothesis_not_supported"
    assert response.answer_mode == "hypothesize"
    assert response.epistemic_level == "evidence_bounded_hypothesis"
    assert response.unsupported_parts == ["前提未通过证据审计", "推理忽略频段冲突"]


def test_novelty_answer_is_rendered_only_from_registered_claims() -> None:
    answer = ModelAnswer(
        answer="未登记的自由文本不得展示",
        citation_ids=[],
        hallucination_risk="low",
        novelty_claims=[NoveltyClaim(claim="已登记创新主张", citation_id=uuid4(), claim_strength="explicit")],
    )

    rendered = render_novelty_claims(answer)

    assert "已登记创新主张" in rendered
    assert "未登记" not in rendered


def test_synthesis_is_rendered_from_atomic_claims() -> None:
    claim = GroundedClaim(text="采用等效电路模型解释宽带机制", citation_ids=[uuid4()], claim_type="synthesized_fact", label="理论方法")

    assert render_grounded_claims([claim]) == "1. 理论方法：采用等效电路模型解释宽带机制。"


def test_derivation_renders_inputs_operation_and_result() -> None:
    item = DeterministicDerivation(statement="绝对带宽可由频段端点计算", inputs=["下限 7 GHz", "上限 18 GHz"], operation="18 - 7", result="11 GHz", citation_ids=[uuid4()])
    rendered = render_derivations([item])

    assert "已知：下限 7 GHz；上限 18 GHz" in rendered
    assert "计算：18 - 7" in rendered
    assert "结果：11 GHz" in rendered


def test_hypothesis_rendering_marks_unverified_status() -> None:
    item = EvidenceBoundedHypothesis(
        claim="该组合可能提供动态调谐能力",
        premises=[
            HypothesisPremise(claim="材料片阻可调", citation_ids=[uuid4()]),
            HypothesisPremise(claim="结构响应依赖表面阻抗", citation_ids=[uuid4()]),
        ],
        confidence="medium",
        assumptions=["材料可按目标几何制备"],
        validation_needed=["全波仿真", "样机测量"],
    )
    rendered = render_hypotheses([item])

    assert "不是论文已经验证的结论" in rendered
    assert "关键假设" in rendered
    assert "建议验证" in rendered


def test_partially_entailed_hypothesis_premise_is_trimmed_to_supported_scope() -> None:
    first_id = uuid4()
    second_id = uuid4()
    hypothesis = EvidenceBoundedHypothesis(
        claim="The response may change.",
        premises=[
            HypothesisPremise(claim="Supported premise", citation_ids=[first_id]),
            HypothesisPremise(claim="Supported fact plus unsupported contrast", citation_ids=[second_id]),
        ],
        confidence="low",
        assumptions=["Compatibility is unknown"],
        validation_needed=["Simulation"],
    )
    report = NoveltyEntailmentAudit(
        answer_claims_fully_covered=True,
        results=[
            ClaimEntailmentResult(claim_index=0, verdict="entailed", reason="supported"),
            ClaimEntailmentResult(
                claim_index=1,
                verdict="partially_entailed",
                reason="contrast unsupported",
                supported_scope="Supported fact",
            ),
        ],
    )

    salvaged = salvage_partially_entailed_premises(hypothesis, report)

    assert salvaged is not None
    assert salvaged.premises[1].claim == "Supported fact"
    assert salvaged.premises[1].citation_ids == [second_id]


def test_cross_document_answer_requires_two_citation_documents() -> None:
    document_id = uuid4()
    citation = Citation(citation_id=uuid4(), document_id=document_id, paper_title="a.pdf", page_start=1, page_end=1, section_path=None, quoted_snippet="evidence")

    assert not verify_cross_document_citations([citation]).passed


def test_novelty_answer_does_not_treat_model_quote_as_evidence() -> None:
    document_id = uuid4()
    method = RetrievedChunk(uuid4(), document_id, "showing unprecedented wave control", 1, 1, None, [], 0.7, retrieval_roles=("novelty_claim",))
    result = RetrievedChunk(uuid4(), document_id, "The measured absorption is improved.", 8, 8, None, [], 0.7, retrieval_roles=("result_or_advantage",))
    citations = [
        Citation(citation_id=item.chunk_id, document_id=document_id, paper_title="a.pdf", page_start=item.page_start, page_end=item.page_end, section_path=None, quoted_snippet=item.content)
        for item in (method, result)
    ]
    answer = ModelAnswer(
        answer="该结构展现了前所未有的波调控能力。",
        citation_ids=[item.citation_id for item in citations],
        hallucination_risk="low",
        novelty_claims=[
            NoveltyClaim(claim="该结构展现了前所未有的波调控能力", citation_id=method.chunk_id, source_quote="model-generated paraphrase not present in source", claim_strength="explicit_strong"),
            NoveltyClaim(claim="该结构改善了吸收性能", citation_id=result.chunk_id, source_quote="The measured absorption is improved.", claim_strength="explicit"),
        ],
    )

    decision = verify_novelty_answer(answer, citations, [method, result])

    assert decision.passed


def test_novelty_entailment_audit_rejects_partial_claim() -> None:
    audit = NoveltyEntailmentAudit(
        answer_claims_fully_covered=True,
        results=[
            ClaimEntailmentResult(claim_index=0, verdict="partially_entailed", reason="性能提升有依据，但首次没有依据", unsupported_parts=["首次"])
        ],
    )

    decision = verify_novelty_entailment_audit(audit, 1)

    assert not decision.passed
    assert "partially_entailed" in decision.reason


def test_novelty_entailment_audit_rejects_duplicate_claim_indexes() -> None:
    audit = NoveltyEntailmentAudit(
        answer_claims_fully_covered=True,
        results=[
            ClaimEntailmentResult(claim_index=0, verdict="not_entailed", reason="主体不一致"),
            ClaimEntailmentResult(claim_index=0, verdict="entailed", reason="重复结果不应覆盖前一项"),
        ],
    )

    decision = verify_novelty_entailment_audit(audit, 1)

    assert not decision.passed
    assert "重复" in decision.reason


@pytest.mark.parametrize(
    ("verdict", "reason", "expected_passed"),
    [
        ("entailed", "中英文表达语义等价，主体与强度一致", True),
        ("not_entailed", "原文仅称 proposed，不能推出世界首个", False),
        ("not_entailed", "novel 修饰的是既有工作，不是本文结构", False),
        ("not_entailed", "原文称仍是挑战，不能推出已经实现", False),
    ],
)
def test_adversarial_semantic_verdicts_fail_closed(verdict: str, reason: str, expected_passed: bool) -> None:
    audit = NoveltyEntailmentAudit(
        answer_claims_fully_covered=True,
        results=[ClaimEntailmentResult(claim_index=0, verdict=verdict, reason=reason)],
    )

    decision = verify_novelty_entailment_audit(audit, 1)

    assert decision.passed is expected_passed


def test_novelty_entailment_audit_rejects_unregistered_answer_claims() -> None:
    audit = NoveltyEntailmentAudit(
        answer_claims_fully_covered=False,
        uncovered_answer_claims=["该器件还是世界首个柔性实现"],
        results=[ClaimEntailmentResult(claim_index=0, verdict="entailed", reason="已登记主张有依据")],
    )

    decision = verify_novelty_entailment_audit(audit, 1)

    assert not decision.passed
    assert "未登记" in decision.reason


def test_unknown_citation_fails_audit() -> None:
    evidence = RetrievedChunk(uuid4(), uuid4(), "Evidence", 1, 1, None, [], 0.8)
    citation = Citation(citation_id=uuid4(), document_id=evidence.document_id, paper_title="paper.pdf", page_start=1, page_end=1, section_path=None, quoted_snippet="Evidence")

    result = verify_citations_exist([citation], [evidence])

    assert not result.passed


def test_numeric_claim_must_exist_in_full_cited_evidence() -> None:
    chunk_id = uuid4()
    document_id = uuid4()
    evidence = RetrievedChunk(chunk_id, document_id, "Absorption occurs at 17.6 GHz.", 5, 5, None, [], 0.8)
    citation = Citation(citation_id=chunk_id, document_id=document_id, paper_title="paper.pdf", page_start=5, page_end=5, section_path=None, quoted_snippet="Absorption occurs")
    answer = ModelAnswer(answer="The resonance is at 17.6 GHz.", citation_ids=[chunk_id], hallucination_risk="low")

    assert verify_claim_tokens_against_evidence(answer, [citation], [evidence]).passed

    unsupported = answer.model_copy(update={"answer": "The resonance is at 100 THz."})
    assert not verify_claim_tokens_against_evidence(unsupported, [citation], [evidence]).passed


def test_ohms_per_square_ocr_variant_is_normalized() -> None:
    chunk_id = uuid4()
    document_id = uuid4()
    evidence = RetrievedChunk(chunk_id, document_id, "The sheet resistance is 70 U=sq.", 5, 5, None, [], 0.8)
    citation = Citation(citation_id=chunk_id, document_id=document_id, paper_title="paper.pdf", page_start=5, page_end=5, section_path=None, quoted_snippet="resistance")
    answer = ModelAnswer(answer="The sheet resistance is 70 Ω/sq.", citation_ids=[chunk_id], hallucination_risk="low")

    assert verify_claim_tokens_against_evidence(answer, [citation], [evidence]).passed

    unsupported = answer.model_copy(update={"answer": "The sheet resistance is 99 Ω/sq."})
    assert not verify_claim_tokens_against_evidence(unsupported, [citation], [evidence]).passed


def test_decimal_colon_extraction_is_equivalent_for_units() -> None:
    chunk_id = uuid4()
    document_id = uuid4()
    evidence = RetrievedChunk(chunk_id, document_id, "The period is 11:2 mm.", 2, 2, None, [], 0.8)
    citation = Citation(citation_id=chunk_id, document_id=document_id, paper_title="paper.pdf", page_start=2, page_end=2, section_path=None, quoted_snippet="period")
    answer = ModelAnswer(answer="The period is 11.2 mm.", citation_ids=[chunk_id], hallucination_risk="low")

    assert verify_claim_tokens_against_evidence(answer, [citation], [evidence]).passed


def test_numeric_audit_allows_deterministic_bandwidth_difference() -> None:
    chunk_id = uuid4()
    document_id = uuid4()
    evidence = RetrievedChunk(chunk_id, document_id, "The operating band extends from 7 GHz to 18 GHz.", 2, 2, None, [], 0.8)
    citation = Citation(citation_id=chunk_id, document_id=document_id, paper_title="paper.pdf", page_start=2, page_end=2, section_path=None, quoted_snippet="7 GHz to 18 GHz")
    answer = ModelAnswer(answer="The absolute bandwidth is 11 GHz.", citation_ids=[chunk_id], hallucination_risk="low")

    assert verify_claim_tokens_against_evidence(answer, [citation], [evidence]).passed


def test_bias_resistance_direction_conflict_fails_audit() -> None:
    chunk_id = uuid4()
    document_id = uuid4()
    evidence = RetrievedChunk(chunk_id, document_id, "Bias values are 0 V(300 U=sq) and 6 V(132 U=sq).", 3, 3, None, [], 0.8)
    citation = Citation(citation_id=chunk_id, document_id=document_id, paper_title="paper.pdf", page_start=3, page_end=3, section_path=None, quoted_snippet="bias")
    answer = ModelAnswer(answer="The sheet resistance increases as the bias voltage rises.", citation_ids=[chunk_id], hallucination_risk="low")

    result = verify_claim_tokens_against_evidence(answer, [citation], [evidence])

    assert not result.passed
    assert "变化方向" in result.reason
