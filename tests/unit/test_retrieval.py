from uuid import uuid4

from paper_rag.config import Settings
from paper_rag.schemas.query_plan import AnswerMode, EvidenceType, QueryPlan, RetrievalQuery
from paper_rag.services import retrieval
from paper_rag.services.retrieval import EvidenceDecision, RetrievedChunk, _is_reference_dense, _tokens, has_sufficient_retrieval_evidence
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult


def test_similarity_score_can_pass_evidence_threshold() -> None:
    candidate = RetrievedChunk(uuid4(), uuid4(), "evidence", 1, 1, None, [], 0.5)

    decision = has_sufficient_retrieval_evidence("What evidence exists?", [candidate], Settings(retrieval_min_score=0.25))

    assert decision == EvidenceDecision(True, None)


def test_missing_numeric_unit_anchor_forces_refusal() -> None:
    candidate = RetrievedChunk(uuid4(), uuid4(), "The absorber operates from 5 GHz to 20 GHz.", 1, 1, None, [], 0.8)

    decision = has_sufficient_retrieval_evidence("What happens at 17.6 GHz?", [candidate], Settings(retrieval_min_score=0.25))

    assert decision == EvidenceDecision(False, "问题中的数值或单位未出现在检索证据中")


def test_low_confidence_ocr_cannot_be_only_numeric_evidence() -> None:
    candidate = RetrievedChunk(
        uuid4(), uuid4(), "Measured at 17.6 GHz.", 1, 1, None, [], 0.8, 0.7, True
    )

    decision = has_sufficient_retrieval_evidence(
        "What happens at 17.6 GHz?",
        [candidate],
        Settings(retrieval_min_score=0.25),
    )

    assert decision == EvidenceDecision(False, "精确数值仅见于低置信 OCR 证据")


def test_overview_question_uses_document_level_evidence() -> None:
    document_id = uuid4()
    candidates = [
        RetrievedChunk(uuid4(), document_id, "摘要", 1, 1, None, [], 0.3),
        RetrievedChunk(uuid4(), document_id, "结论", 10, 10, None, [], 0.3),
    ]

    intent = QueryIntentResult(QueryIntent.OVERVIEW, 1.0, 1.0, "test")
    assert has_sufficient_retrieval_evidence(
        "这篇综述讲了什么", candidates, Settings(), intent
    ).sufficient


def test_chinese_question_produces_lexical_features() -> None:
    tokens = _tokens("石墨烯如何实现动态吸收调控")

    assert "石墨" in tokens
    assert "吸收" in tokens


def test_reference_dense_chunk_is_not_treated_as_paper_method_evidence() -> None:
    content = "Conclusion text.\nREFERENCES\n" + "\n".join(
        f"[{index}] Author, A dual-band surface with independent tunability."
        for index in range(1, 9)
    )

    assert _is_reference_dense(content)


def test_reference_continuation_without_heading_is_detected() -> None:
    content = "\n".join(
        f"[{index}] Author, Journal title, vol. 1, pp. 10-20, 2024."
        for index in range(19, 27)
    )

    assert _is_reference_dense(content)


def test_candidate_retrieval_overfetches_and_excludes_reference_dense_chunks() -> None:
    reference_id = uuid4()
    evidence_id = uuid4()
    document_id = uuid4()

    class Provider:
        def embed_query(self, text):
            return [1.0, 0.0]

    class Collection:
        metadata = {"hnsw:space": "cosine"}

        def query(self, *, query_embeddings, n_results, where, include):
            assert n_results == 6
            return {
                "documents": [[
                    "REFERENCES\n" + "\n".join(f"[{index}] cited paper" for index in range(1, 8)),
                    "The measured 0.9-absorption bandwidth ranges from 7 to 18.2 GHz.",
                ]],
                "metadatas": [[
                    {"chunk_id": str(reference_id), "document_id": str(document_id), "page_start": 9, "page_end": 9, "section_path": "References", "formula_ids": "[]"},
                    {"chunk_id": str(evidence_id), "document_id": str(document_id), "page_start": 8, "page_end": 8, "section_path": "Experimental results", "formula_ids": "[]"},
                ]],
                "distances": [[0.01, 0.2]],
            }

    items = retrieval.retrieve_candidates(
        Collection(),
        Provider(),
        "吸收频带是多少",
        2,
        document_id,
    )

    assert [item.chunk_id for item in items] == [evidence_id]


def test_legacy_method_expansion_does_not_depend_on_query_plan(monkeypatch) -> None:
    candidate = RetrievedChunk(uuid4(), uuid4(), "method evidence", 2, 2, None, [], 0.8)
    monkeypatch.setattr(retrieval, "retrieve_candidates", lambda *args, **kwargs: [candidate])

    items = retrieval.retrieve_question_evidence(
        object(),
        object(),
        "本文用了什么方法",
        4,
        candidate.document_id,
        QueryIntentResult(QueryIntent.METHOD, 0.9, 0.2, "test"),
    )

    assert items == [candidate]


def test_hypothesis_gate_rejects_reference_only_structure_premise() -> None:
    material_document = uuid4()
    structure_document = uuid4()
    candidates = [
        RetrievedChunk(uuid4(), material_document, "Graphene sheet resistance is controlled by bias voltage.", 2, 2, None, [], 0.8, retrieval_roles=("premise_for_material",)),
        RetrievedChunk(uuid4(), structure_document, "REFERENCES\n[1] A varactor RFSS.\n[2] Dual-band independent tuning.\n[3] Active FSS.\n[4] Tunable radome.\n[5] Reconfigurable surface.", 5, 5, "REFERENCES", [], 0.8, retrieval_roles=("premise_for_structure", "counterevidence")),
    ]
    plan = QueryPlan(
        intent=QueryIntent.CROSS_DOCUMENT,
        answer_mode=AnswerMode.HYPOTHESIZE,
        standalone_question="材料机制用于目标结构会怎样",
        retrieval_queries=[RetrievalQuery(query="结构机制", evidence_type=EvidenceType.PREMISE_FOR_STRUCTURE)],
        confidence=0.9,
    )

    decision = has_sufficient_retrieval_evidence(
        plan.standalone_question,
        candidates,
        Settings(),
        QueryIntentResult(QueryIntent.CROSS_DOCUMENT, 0.9, 0.3, "test"),
        plan,
    )

    assert not decision.sufficient
    assert decision.reason == "证据不足以同时建立材料与结构推理前提"


def test_overview_question_rejects_ambiguous_document_scope() -> None:
    candidates = [
        RetrievedChunk(uuid4(), uuid4(), "摘要 A", 1, 1, None, [], 0.8),
        RetrievedChunk(uuid4(), uuid4(), "摘要 B", 1, 1, None, [], 0.8),
    ]

    decision = has_sufficient_retrieval_evidence(
        "这篇综述讲了什么", candidates, Settings(), QueryIntentResult(QueryIntent.OVERVIEW, 1.0, 1.0, "test")
    )

    assert not decision.sufficient
    assert decision.reason == "概述问题需要明确指定一篇论文"


def test_novelty_gate_requires_newness_and_supporting_evidence_roles() -> None:
    document_id = uuid4()
    intent = QueryIntentResult(QueryIntent.NOVELTY, 0.95, 0.4, "deepseek_rewrite")
    candidates = [
        RetrievedChunk(uuid4(), document_id, "A new structure is proposed.", 1, 1, None, [], 0.6, retrieval_roles=("novelty_claim",)),
        RetrievedChunk(uuid4(), document_id, "It improves measured absorption.", 8, 8, None, [], 0.6, retrieval_roles=("result_or_advantage",)),
    ]

    assert has_sufficient_retrieval_evidence(
        "本文的创新点是什么", candidates, Settings(), intent
    ).sufficient


def test_review_novelty_gate_uses_synthesis_and_comparison_roles() -> None:
    document_id = uuid4()
    intent = QueryIntentResult(QueryIntent.NOVELTY, 0.9, 0.3, "test")
    candidates = [
        RetrievedChunk(uuid4(), document_id, "A taxonomy is developed.", 2, 2, None, [], 0.6, retrieval_roles=("synthesis_or_taxonomy",)),
        RetrievedChunk(uuid4(), document_id, "Approaches and trends are compared.", 14, 14, None, [], 0.6, retrieval_roles=("comparison_baseline",)),
    ]

    assert has_sufficient_retrieval_evidence(
        "综述的创新点是什么", candidates, Settings(), intent, document_genres=["review_paper"]
    ).sufficient


def test_thesis_novelty_gate_uses_explicit_innovation_and_chapter_result() -> None:
    document_id = uuid4()
    intent = QueryIntentResult(QueryIntent.NOVELTY, 0.9, 0.3, "test")
    candidates = [
        RetrievedChunk(uuid4(), document_id, "主要创新点如下。", 7, 7, None, [], 0.6, retrieval_roles=("explicit_innovation",)),
        RetrievedChunk(uuid4(), document_id, "本章实验结果表明。", 95, 95, None, [], 0.6, retrieval_roles=("chapter_result",)),
    ]

    assert has_sufficient_retrieval_evidence(
        "学位论文的创新点是什么", candidates, Settings(), intent, document_genres=["thesis"]
    ).sufficient
