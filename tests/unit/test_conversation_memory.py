from __future__ import annotations

from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.conversation import Conversation, ConversationMessage, ConversationTurn
from paper_rag.schemas.chat import AnswerResponse, ChatRequest
from paper_rag.services.conversation_memory import (
    begin_conversation_turn,
    build_conversation_context,
    complete_conversation_turn,
    fail_running_conversation_turn,
    resolve_conversation_request,
)
from paper_rag.services.query_rewrite import build_query_rewrite_messages, parse_query_plan


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _response(answer: str = "有证据的回答") -> AnswerResponse:
    return AnswerResponse(
        answer=answer,
        citations=[],
        evidence_status="sufficient",
        refused=False,
        refusal_reason=None,
        hallucination_risk="low",
        audit_result="passed",
    )


def _plan():
    return parse_query_plan(
        '{"intent":"result_parameter","answer_mode":"synthesize",'
        '"standalone_question":"该论文的实验结果是什么？",'
        '"retrieval_queries":[{"query":"实验结果","evidence_type":"experiment"}],'
        '"entities":[],"required_evidence":["experiment"],'
        '"scope_requirement":"current_scope","needs_clarification":false,'
        '"clarification_question":null,"confidence":0.9}'
    )


def test_turn_is_persisted_and_duplicate_client_id_replays_result() -> None:
    session = _session()
    document_id = uuid4()
    conversation = Conversation(title="Research", scope="all")
    session.add(conversation)
    session.commit()
    request = ChatRequest(
        session_id="session-123",
        conversation_id=conversation.id,
        client_turn_id="turn-12345678",
        question="它的实验结果呢？",
        scope="selected",
        document_ids=[document_id],
    )

    started = begin_conversation_turn(session, request)
    assert started.replayed_response is None
    complete_conversation_turn(session, request, _response(), _plan(), [])

    messages = list(
        session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.role.desc())
        )
    )
    turn = session.scalar(select(ConversationTurn))
    session.refresh(conversation)
    assert {message.role for message in messages} == {"user", "assistant"}
    assert turn.status == "completed"
    assert conversation.scope == "selected"
    assert str(document_id) in conversation.selected_document_ids_json

    replay = begin_conversation_turn(session, request)
    assert replay.replayed_response is not None
    assert replay.replayed_response.answer == "有证据的回答"
    assert session.scalar(select(ConversationTurn).where(ConversationTurn.client_turn_id == "turn-12345678")) == turn


def test_failed_turn_is_visible_but_excluded_from_rewrite_context() -> None:
    session = _session()
    conversation = Conversation(title="Research")
    session.add(conversation)
    session.commit()

    successful = ChatRequest(
        session_id="session-123",
        conversation_id=conversation.id,
        client_turn_id="turn-success",
        question="第一篇论文的方法是什么？",
    )
    begin_conversation_turn(session, successful)
    complete_conversation_turn(session, successful, _response("方法包括仿真和实验。"), _plan(), [])

    failed = ChatRequest(
        session_id="session-123",
        conversation_id=conversation.id,
        client_turn_id="turn-failed",
        question="错误地声称参数是 999 GHz",
    )
    begin_conversation_turn(session, failed)
    complete_conversation_turn(
        session,
        failed,
        AnswerResponse(
            answer="模型服务不可用",
            citations=[],
            evidence_status="insufficient",
            refused=True,
            refusal_reason="provider failure",
            hallucination_risk="unknown",
            audit_result="provider_failure",
            action="error",
        ),
        None,
        [],
    )

    context = build_conversation_context(session, conversation.id)
    serialized = str(context)
    assert "仿真和实验" in serialized
    assert "999 GHz" not in serialized
    assert "模型服务不可用" not in serialized


def test_unhandled_running_turn_becomes_terminal_and_replayable() -> None:
    session = _session()
    conversation = Conversation(title="Research")
    session.add(conversation)
    session.commit()
    request = ChatRequest(
        session_id="session-123",
        conversation_id=conversation.id,
        client_turn_id="turn-crashed",
        question="继续分析实验结果",
    )

    begin_conversation_turn(session, request)
    failed = fail_running_conversation_turn(session, request)

    assert failed is not None
    assert failed.action == "error"
    replay = begin_conversation_turn(session, request)
    assert replay.replayed_response is not None
    assert replay.replayed_response.audit_result == "unhandled_question_failure"
    turn = session.scalar(select(ConversationTurn).where(ConversationTurn.client_turn_id == "turn-crashed"))
    assert turn.status == "failed"


def test_rewrite_context_is_marked_untrusted_and_not_evidence() -> None:
    context = {
        "recent_messages_untrusted_data": [
            {"role": "user", "content": "这篇论文的方法是什么？"},
            {"role": "assistant", "content": "方法包括仿真和实验。"},
        ],
        "conversation_summary_untrusted_data": {},
    }

    messages = build_query_rewrite_messages("它的实验结果呢？", [], "all", context)
    serialized = str(messages)

    assert "recent_messages_untrusted_data" in serialized
    assert "context only" in serialized
    assert "never evidence" in serialized


def test_omitted_scope_inherits_conversation_scope_but_explicit_all_does_not() -> None:
    session = _session()
    document_id = uuid4()
    conversation = Conversation(
        title="Research",
        scope="selected",
        selected_document_ids_json=f'["{document_id}"]',
    )
    session.add(conversation)
    session.commit()

    inherited = resolve_conversation_request(
        session,
        ChatRequest(
            session_id="session-123",
            conversation_id=conversation.id,
            question="它的结果呢？",
        ),
    )
    explicit_all = resolve_conversation_request(
        session,
        ChatRequest(
            session_id="session-123",
            conversation_id=conversation.id,
            question="换成全部论文",
            scope="all",
        ),
    )

    assert inherited.scope == "selected"
    assert inherited.document_ids == [document_id]
    assert explicit_all.scope == "all"
    assert explicit_all.document_ids == []


def test_relevant_older_turn_is_recalled_within_bounded_context() -> None:
    session = _session()
    conversation = Conversation(title="Long research")
    session.add(conversation)
    session.commit()
    for index in range(50):
        request = ChatRequest(
            session_id="session-123",
            conversation_id=conversation.id,
            client_turn_id=f"history-turn-{index}",
            question=(
                "石墨烯方阻如何影响吸收带宽？"
                if index == 0
                else f"无关的阶段性问题 {index}"
            ),
        )
        begin_conversation_turn(session, request)
        complete_conversation_turn(
            session,
            request,
            _response(f"阶段性回答 {index}"),
            _plan(),
            [],
            question_embedding=[1.0, 0.0] if index == 0 else [0.0, 1.0],
        )

    context = build_conversation_context(
        session,
        conversation.id,
        recent_message_limit=6,
        character_budget=3000,
        query_embedding=[1.0, 0.0],
        relevant_turn_limit=2,
    )

    assert any(
        item["turn_index"] == 1
        for item in context["relevant_history_untrusted_data"]
    )
    assert len(str(context)) < 5000
    assert context["conversation_summary_untrusted_data"]["turn_count"] == 50
