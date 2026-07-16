from uuid import uuid4

from paper_rag.config import Settings
from paper_rag.services.retrieval import RetrievedChunk, has_sufficient_retrieval_evidence
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult


def test_strong_lexical_support_can_rescue_borderline_semantic_score() -> None:
    evidence = RetrievedChunk(
        uuid4(), uuid4(), "The unit period, strip length and square hole width are reported.",
        2, 2, None, [], 0.45,
    )

    decision = has_sufficient_retrieval_evidence(
        "What are the unit period, strip length, and square hole width?",
        [evidence],
        Settings(),
    )

    assert decision.sufficient


def test_generic_paper_overlap_does_not_rescue_external_question() -> None:
    evidence = RetrievedChunk(
        uuid4(), uuid4(), "The authors conclude this paper with measured absorption.",
        8, 8, None, [], 0.45,
    )

    decision = has_sufficient_retrieval_evidence(
        "What did the authors publish in their next paper?", [evidence], Settings()
    )

    assert not decision.sufficient


def test_generic_chinese_paper_words_do_not_rescue_external_question() -> None:
    evidence = RetrievedChunk(
        uuid4(), uuid4(), "本文作者研究了吸收性能并报告实验结果。",
        2, 2, None, [], 0.45,
    )

    decision = has_sufficient_retrieval_evidence(
        "该论文研究量子纠缠的结果是什么？", [evidence], Settings()
    )

    assert not decision.sufficient


def test_cross_document_intent_requires_evidence_from_two_documents() -> None:
    document_id = uuid4()
    evidence = [
        RetrievedChunk(uuid4(), document_id, "Method A", 1, 1, None, [], 0.8),
        RetrievedChunk(uuid4(), document_id, "Method B", 2, 2, None, [], 0.8),
    ]
    intent = QueryIntentResult(QueryIntent.CROSS_DOCUMENT, 1.0, 1.0, "rule")

    decision = has_sufficient_retrieval_evidence(
        "比较这两篇论文的方法", evidence, Settings(), intent
    )

    assert not decision.sufficient
    assert decision.reason == "跨论文综合未召回至少两篇论文的证据"
