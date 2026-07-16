from uuid import uuid4
from unittest.mock import MagicMock

from paper_rag.config import Settings
from paper_rag.services.evidence_gate import evaluate_evidence
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult
from paper_rag.services.query_rewrite import parse_query_plan
from paper_rag.services.retrieval import RetrievedChunk, has_sufficient_retrieval_evidence


def test_unrelated_bge_score_below_calibrated_threshold_is_refused() -> None:
    evidence = RetrievedChunk(uuid4(), uuid4(), "Unrelated evidence", 1, 1, None, [], 0.4875)

    decision = has_sufficient_retrieval_evidence("请解释量子纠缠实验。", [evidence], Settings())

    assert not decision.sufficient
    assert decision.reason == "已召回候选证据，但相关性不足以支持可靠回答"


def test_formula_extract_reaches_deterministic_quality_path_without_grounded_semantics() -> None:
    evidence = RetrievedChunk(
        uuid4(),
        uuid4(),
        "The Kubo formula appears on this page.",
        4,
        4,
        "Theoretical model",
        [],
        0.9,
    )
    intent = QueryIntentResult(QueryIntent.FORMULA, 0.9, 0.5, "test")
    plan = parse_query_plan(
        '{"intent":"formula_explanation","answer_mode":"extract","standalone_question":"Kubo公式是什么",'
        '"retrieval_queries":[{"query":"Kubo formula","evidence_type":"formula_context"}],'
        '"entities":[],"required_evidence":["formula_context"],"scope_requirement":"single_document",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )

    decision = evaluate_evidence(
        MagicMock(),
        "Kubo公式是什么",
        [evidence],
        Settings(),
        intent,
        plan,
        ["research_paper"],
    )

    assert decision.sufficient
