from uuid import uuid4

import pytest

from paper_rag.schemas.query_plan import (
    AnswerMode,
    EntityType,
    EvidenceType,
    QueryEntity,
    ScopeRequirement,
)
from paper_rag.services.query_intent import QueryIntent
from paper_rag.services.query_rewrite import (
    QueryRewriteSchemaError,
    ScopeDocument,
    build_query_rewrite_messages,
    build_safe_fallback_plan,
    link_soft_query_entities,
    parse_query_plan,
    normalize_query_plan,
    resolve_linked_entities_with_evidence,
    rewrite_fidelity_score,
    validate_plan_scope,
)
from tests.unit.reference_test_support import make_chunk


def test_novelty_query_plan_parses_strict_structure() -> None:
    plan = parse_query_plan(
        """{
          "intent": "novelty_contribution",
          "standalone_question": "本文相对于已有工作的创新与贡献是什么？",
          "retrieval_queries": [
            {"query": "现有工作的局限和本文解决的问题", "evidence_type": "problem_or_gap"},
            {"query": "本文提出的新方法或新结构", "evidence_type": "novelty_claim"},
            {"query": "新方法带来的结果或优势", "evidence_type": "result_or_advantage"}
          ],
          "entities": [],
          "required_evidence": ["problem_or_gap", "novelty_claim", "result_or_advantage"],
          "scope_requirement": "single_document",
          "needs_clarification": false,
          "clarification_question": null,
          "confidence": 0.94
        }"""
    )

    assert plan.intent == QueryIntent.NOVELTY
    assert EvidenceType.NOVELTY_CLAIM in plan.required_evidence


def test_query_plan_rejects_answer_field() -> None:
    with pytest.raises(QueryRewriteSchemaError):
        parse_query_plan(
            '{"intent":"general","standalone_question":"q",'
            '"retrieval_queries":[{"query":"q","evidence_type":"general"}],'
            '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
            '"needs_clarification":false,"clarification_question":null,"confidence":0.8,'
            '"answer":"invented"}'
        )


def test_soft_entity_linker_does_not_try_to_resolve_formula_reference(
    session,
    document,
) -> None:
    entity = QueryEntity(
        surface="公式5",
        canonical=None,
        entity_type=EntityType.FORMULA,
    )
    assert link_soft_query_entities(session, [entity], [document.id]) == []


def test_soft_entity_linker_keeps_material_behavior(session, document) -> None:
    session.add(
        make_chunk(document, content="A graphene metasurface is proposed.")
    )
    session.commit()
    entity = QueryEntity(
        surface="石墨烯",
        canonical="graphene",
        entity_type=EntityType.MATERIAL,
    )
    result = link_soft_query_entities(session, [entity], [document.id])
    assert result[0].linked is True


def test_schema_error_keeps_safe_validation_details() -> None:
    with pytest.raises(QueryRewriteSchemaError) as captured:
        parse_query_plan('{"intent":"unknown"}')

    assert captured.value.raw_sha256
    assert captured.value.validation_errors
    assert all("input" not in item for item in captured.value.validation_errors)


def test_rewrite_prompt_forbids_answer_and_factual_expansion() -> None:
    messages = build_query_rewrite_messages(
        "本文的创新点是什么", [ScopeDocument(uuid4(), "paper.pdf")], "selected"
    )
    serialized = str(messages)

    assert "never answer" in serialized
    assert "never add factual claims" in serialized
    assert "sk-" not in serialized


def test_single_document_plan_rejects_ambiguous_scope() -> None:
    plan = parse_query_plan(
        '{"intent":"overview","standalone_question":"概述本文",'
        '"retrieval_queries":[{"query":"摘要与结论","evidence_type":"overview"}],'
        '"entities":[],"required_evidence":["overview"],"scope_requirement":"single_document",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )

    reason = validate_plan_scope(
        plan, [ScopeDocument(uuid4(), "a.pdf"), ScopeDocument(uuid4(), "b.pdf")]
    )

    assert reason == "该问题需要明确选择一篇论文"


def test_low_confidence_alone_does_not_force_clarification() -> None:
    plan = parse_query_plan(
        '{"intent":"general","standalone_question":"用户问题",'
        '"retrieval_queries":[{"query":"用户问题","evidence_type":"general"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":"请补充对象", "confidence":0.2}'
    )

    assert validate_plan_scope(plan, [ScopeDocument(uuid4(), "a.pdf")]) is None


class FidelityProvider:
    model_id = "test"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 2:
            return [[1.0, 0.0], [0.8, 0.6]]
        return [[1.0, 0.0] if "创新点" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_rewrite_fidelity_uses_normalized_semantic_similarity() -> None:
    assert rewrite_fidelity_score(FidelityProvider(), "创新点是什么", "主要创新与贡献") == 0.8


def test_safe_fallback_uses_semantic_novelty_plan_without_regex() -> None:
    plan = build_safe_fallback_plan("这项工作的亮点在哪里", FidelityProvider(), [ScopeDocument(uuid4(), "paper.pdf", "research_paper")])

    assert plan.standalone_question == "这项工作的亮点在哪里"


class ComparisonFallbackProvider(FidelityProvider):
    model_id = "comparison-fallback"

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.49, 0.0] if "比较论文中的" in text else [0.47, 0.0] for text in texts]


def test_multi_document_comparison_fallback_enforces_cross_document_intent() -> None:
    documents = [ScopeDocument(uuid4(), "a.pdf"), ScopeDocument(uuid4(), "b.pdf")]
    plan = build_safe_fallback_plan("比较两篇论文", ComparisonFallbackProvider(), documents)

    assert plan.intent == QueryIntent.CROSS_DOCUMENT
    assert plan.scope_requirement == ScopeRequirement.MULTIPLE_DOCUMENTS


class LowMarginResultProvider(FidelityProvider):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.40 - index * 0.01, 0.0] for index, _ in enumerate(texts)]


def test_safe_fallback_uses_domain_candidate_instead_of_general() -> None:
    plan = build_safe_fallback_plan("5 V时带宽是多少", LowMarginResultProvider(), [ScopeDocument(uuid4(), "paper.pdf")])

    assert plan.intent != QueryIntent.GENERAL


def test_novelty_plan_is_completed_for_research_paper() -> None:
    plan = parse_query_plan(
        '{"intent":"novelty_contribution","standalone_question":"本文创新点是什么",'
        '"retrieval_queries":[{"query":"本文创新点","evidence_type":"novelty_claim"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )

    normalized = normalize_query_plan(
        plan, [ScopeDocument(uuid4(), "paper.pdf", "research_paper")]
    )

    roles = {item.evidence_type for item in normalized.retrieval_queries}
    assert EvidenceType.PROBLEM_OR_GAP in roles
    assert EvidenceType.NOVELTY_CLAIM in roles
    assert EvidenceType.RESULT_OR_ADVANTAGE in roles
    assert normalized.scope_requirement == ScopeRequirement.SINGLE_DOCUMENT


def test_unclassified_novelty_uses_cross_type_evidence_plan() -> None:
    plan = parse_query_plan(
        '{"intent":"novelty_contribution","standalone_question":"本文贡献是什么",'
        '"retrieval_queries":[{"query":"贡献","evidence_type":"general"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.8}'
    )

    normalized = normalize_query_plan(plan, [ScopeDocument(uuid4(), "unknown.pdf", "unclassified")])

    roles = {item.evidence_type for item in normalized.retrieval_queries}
    assert EvidenceType.NOVELTY_CLAIM in roles
    assert EvidenceType.METHOD_OR_STRUCTURE in roles
    assert EvidenceType.RESULT_OR_ADVANTAGE in roles


def test_novelty_plan_requires_one_document_even_after_schema_repair() -> None:
    plan = parse_query_plan(
        '{"intent":"novelty_contribution","standalone_question":"本文创新点是什么",'
        '"retrieval_queries":[{"query":"本文创新点","evidence_type":"novelty_claim"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )

    normalized = normalize_query_plan(
        plan,
        [ScopeDocument(uuid4(), "a.pdf", "research_paper"), ScopeDocument(uuid4(), "b.pdf", "review_paper")],
    )

    assert normalized.needs_clarification
    assert normalized.scope_requirement == ScopeRequirement.SINGLE_DOCUMENT
    assert validate_plan_scope(normalized, []) == "请明确选择一篇论文后再询问该文创新点"


def test_method_plan_gets_required_evidence_role() -> None:
    plan = parse_query_plan(
        '{"intent":"method_mechanism","standalone_question":"方阻如何调节",'
        '"retrieval_queries":[{"query":"方阻","evidence_type":"general"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.4}'
    )

    normalized = normalize_query_plan(plan, [ScopeDocument(uuid4(), "paper.pdf")])

    assert normalized.required_evidence == [EvidenceType.METHOD_OR_STRUCTURE, EvidenceType.CONCLUSION]
    assert normalized.retrieval_queries[0].evidence_type == EvidenceType.METHOD_OR_STRUCTURE
    assert normalized.confidence == 0.6


def test_result_plan_defaults_to_extract_mode() -> None:
    plan = parse_query_plan(
        '{"intent":"result_parameter","standalone_question":"中心频率是多少",'
        '"retrieval_queries":[{"query":"中心频率","evidence_type":"result_or_advantage"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )
    normalized = normalize_query_plan(plan, [ScopeDocument(uuid4(), "paper.pdf")])

    assert normalized.answer_mode == AnswerMode.EXTRACT
    assert EvidenceType.OPERATING_CONDITIONS in {item.evidence_type for item in normalized.retrieval_queries}


def test_hypothesis_mode_requires_premises_conditions_and_counterevidence() -> None:
    plan = parse_query_plan(
        '{"intent":"general","answer_mode":"hypothesize","standalone_question":"材料A用于结构B会怎样",'
        '"retrieval_queries":[{"query":"组合效果","evidence_type":"general"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )
    normalized = normalize_query_plan(plan, [ScopeDocument(uuid4(), "paper.pdf")])
    roles = {item.evidence_type for item in normalized.retrieval_queries}

    assert normalized.answer_mode == AnswerMode.HYPOTHESIZE
    assert EvidenceType.PREMISE_FOR_MATERIAL in roles
    assert EvidenceType.PREMISE_FOR_STRUCTURE in roles
    assert EvidenceType.OPERATING_CONDITIONS in roles
    assert EvidenceType.COUNTEREVIDENCE in roles


class HypothesisModeProvider:
    model_id = "hypothesis-mode-test"
    dimension = 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Conditionally combine" in text else [0.0, 1.0] for text in texts]


def test_semantic_mode_validator_corrects_method_plan_to_hypothesis() -> None:
    plan = parse_query_plan(
        '{"intent":"method_mechanism","answer_mode":"synthesize","standalone_question":"将材料A机制用于结构B可能怎样",'
        '"retrieval_queries":[{"query":"结构方法","evidence_type":"method_or_structure"}],'
        '"entities":[],"required_evidence":[],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.6}'
    )

    normalized = normalize_query_plan(
        plan,
        [ScopeDocument(uuid4(), "a.pdf"), ScopeDocument(uuid4(), "b.pdf")],
        HypothesisModeProvider(),
    )

    assert normalized.intent == QueryIntent.CROSS_DOCUMENT
    assert normalized.answer_mode == AnswerMode.HYPOTHESIZE
    assert normalized.required_evidence == [
        EvidenceType.PREMISE_FOR_MATERIAL,
        EvidenceType.PREMISE_FOR_STRUCTURE,
        EvidenceType.COUNTEREVIDENCE,
    ]


class EntityProvider:
    model_id = "entity-test"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "石墨烯" in text or "graphene" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_cross_language_entity_resolves_against_retrieved_evidence() -> None:
    from paper_rag.services.query_rewrite import LinkedEntity
    from paper_rag.services.retrieval import RetrievedChunk

    document_id = uuid4()
    entity = LinkedEntity("石墨烯", None, "material", True, False, [])
    evidence = [RetrievedChunk(uuid4(), document_id, "The patterned graphene layer is voltage controlled.", 1, 1, None, [], 0.7)]

    resolved = resolve_linked_entities_with_evidence([entity], evidence, EntityProvider())

    assert resolved[0].linked
    assert resolved[0].matched_document_ids == [document_id]


def test_unreported_numeric_entity_never_uses_semantic_fallback() -> None:
    from paper_rag.services.query_rewrite import LinkedEntity
    from paper_rag.services.retrieval import RetrievedChunk

    entity = LinkedEntity("100 THz", None, "parameter", True, False, [])
    evidence = [RetrievedChunk(uuid4(), uuid4(), "Absorption is measured from 7 to 18 GHz.", 1, 1, None, [], 0.7)]

    resolved = resolve_linked_entities_with_evidence([entity], evidence, EntityProvider())

    assert not resolved[0].linked
