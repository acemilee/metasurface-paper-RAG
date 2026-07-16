from unittest.mock import MagicMock
from uuid import uuid4

from paper_rag.api.chat import _attach_query_context, _generation_failure_record, _save_audit
from paper_rag.config import Settings
from paper_rag.schemas.chat import AnswerResponse, ChatRequest
from paper_rag.services.query_rewrite import ScopeDocument
from paper_rag.services.query_rewrite import parse_query_plan
from paper_rag.services.deepseek import DeepSeekSchemaError
from paper_rag.services.retrieval import RetrievedChunk


def test_audit_links_single_document_from_document_ids_scope() -> None:
    document_id = uuid4()
    session = MagicMock()
    request = ChatRequest(
        session_id="session-123",
        question="本文的创新点是什么",
        scope="selected",
        document_ids=[document_id],
    )
    response = AnswerResponse(
        answer="拒答",
        citations=[],
        evidence_status="insufficient",
        refused=True,
        refusal_reason="test",
        hallucination_risk="unknown",
        audit_result="refused_before_generation",
    )

    _save_audit(
        session,
        request,
        response,
        Settings(),
        [ScopeDocument(document_id, "paper.pdf", "research_paper")],
    )

    audit = session.add.call_args.args[0]
    assert audit.document_id == document_id
    assert str(document_id) in audit.selected_document_ids_json
    session.commit.assert_called_once()


def test_query_context_preserves_hypothesis_mode_on_error_response() -> None:
    response = AnswerResponse(
        answer="结构错误",
        citations=[],
        evidence_status="insufficient",
        refused=True,
        refusal_reason="结构错误",
        hallucination_risk="unknown",
        audit_result="schema_or_citation_failure",
        action="error",
    )
    plan = parse_query_plan(
        '{"intent":"cross_document_synthesis","answer_mode":"hypothesize","standalone_question":"组合后会怎样",'
        '"retrieval_queries":[{"query":"材料前提","evidence_type":"premise_for_material"}],'
        '"entities":[],"required_evidence":["premise_for_material"],"scope_requirement":"current_scope",'
        '"needs_clarification":false,"clarification_question":null,"confidence":0.9}'
    )

    result = _attach_query_context(response, plan)

    assert result.answer_mode == "hypothesize"
    assert result.epistemic_level == "evidence_bounded_hypothesis"


def test_generation_schema_failure_records_diagnostics_without_raw_content() -> None:
    chunk_id = uuid4()
    error = DeepSeekSchemaError(
        "invalid structured output",
        validation_errors=["claims: Field required"],
        raw_output_sha256=["a" * 64, "b" * 64],
    )
    evidence = [RetrievedChunk(chunk_id, uuid4(), "evidence", 4, 4, None, [], 0.9)]

    record = _generation_failure_record(2, error, evidence)

    assert record["attempt"] == 2
    assert record["status"] == "failed"
    assert record["error_code"] == "model_schema_failure"
    assert record["validation_errors"] == ["claims: Field required"]
    assert record["raw_output_sha256"] == ["a" * 64, "b" * 64]
    assert record["allowed_citation_ids"] == [str(chunk_id)]
    assert "raw_content" not in record
