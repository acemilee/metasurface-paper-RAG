from __future__ import annotations

import asyncio
from types import SimpleNamespace

import fitz
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import paper_rag.api.chat as chat_api
from paper_rag.api.chat import _background_answer_tasks, _retain_answer_task
from paper_rag.db import Base
from paper_rag.config import Settings
from paper_rag.models.conversation import Conversation, ConversationTurn
from paper_rag.models.document import DocumentStatus, FormulaIndexStatus
from paper_rag.schemas.chat import AnswerResponse, ChatRequest
from paper_rag.services.conversation_memory import begin_conversation_turn
from tests.unit.reference_test_support import (
    formula_extract_plan,
    make_chunk,
    make_formula,
)


def _response() -> AnswerResponse:
    return AnswerResponse(
        answer="done",
        citations=[],
        evidence_status="sufficient",
        refused=False,
        refusal_reason=None,
        hallucination_risk="low",
        audit_result="passed",
    )


def test_answer_task_is_retained_until_it_reaches_a_terminal_state() -> None:
    async def exercise() -> None:
        release = asyncio.Event()

        async def answer() -> AnswerResponse:
            await release.wait()
            return _response()

        task = asyncio.create_task(answer())
        _retain_answer_task(task)
        await asyncio.sleep(0)
        assert task in _background_answer_tasks
        assert not task.cancelled()

        release.set()
        await task
        await asyncio.sleep(0)
        assert task not in _background_answer_tasks

    asyncio.run(exercise())


def test_stream_converts_an_unhandled_crash_into_a_persisted_terminal_turn(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with testing_session() as session:
        conversation = Conversation(title="Crash recovery")
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id

    async def crash_after_turn_start(request, session, settings, key_store, emit):
        begin_conversation_turn(session, request)
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(chat_api, "SessionLocal", testing_session)
    monkeypatch.setattr(chat_api, "_answer_question", crash_after_turn_start)
    app = FastAPI()
    app.include_router(chat_api.router)

    response = TestClient(app).post(
        "/api/chat/stream",
        json={
            "session_id": "session-123",
            "conversation_id": str(conversation_id),
            "client_turn_id": "crashing-turn",
            "question": "触发异常后是否能恢复？",
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    with testing_session() as session:
        turn = session.scalar(
            select(ConversationTurn).where(ConversationTurn.client_turn_id == "crashing-turn")
        )
        assert turn is not None
        assert turn.status == "failed"
        assert turn.audit_result == "unhandled_question_failure"


def test_formula_reference_stream_reaches_formula_source_without_soft_entity_refusal(
    monkeypatch,
    session,
    document,
    tmp_path,
) -> None:
    pdf_path = tmp_path / "reference.pdf"
    with fitz.open() as pdf:
        pdf.new_page(width=600, height=800)
        pdf.save(pdf_path)
    document.stored_path = str(pdf_path)
    document.status = DocumentStatus.COMPLETED
    document.formula_index_status = FormulaIndexStatus.READY
    formula = make_formula(document, number="5", page_number=1)
    competing_formula = make_formula(document, number="6", page_number=1)
    chunk = make_chunk(
        document,
        content="Equation (5) source evidence",
        page_start=1,
        formula_ids=[formula.id],
    )
    session.add_all([formula, competing_formula, chunk])
    session.commit()
    plan = formula_extract_plan(entity_surface="公式5").model_copy(
        update={"standalone_question": "公式6讲了什么"}
    )

    class FakeEmbeddingProvider:
        model_id = "test-embedding"
        dimension = 2

        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    async def rewrite(*args, **kwargs):
        return plan

    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    monkeypatch.setattr(chat_api, "get_embedding_provider", lambda settings: FakeEmbeddingProvider())
    monkeypatch.setattr(chat_api, "rewrite_query", rewrite)
    monkeypatch.setattr(chat_api, "normalize_query_plan", lambda value, documents, provider: value)
    monkeypatch.setattr(chat_api, "rewrite_fidelity_score", lambda *args: 1.0)
    monkeypatch.setattr(chat_api, "get_profile_retrieval_hints", lambda *args: [])
    monkeypatch.setattr(chat_api, "run_synced_chroma_query", lambda *args: [])
    monkeypatch.setattr(
        chat_api,
        "evaluate_evidence",
        lambda *args: SimpleNamespace(sufficient=True, reason=None),
    )
    request = ChatRequest(
        session_id="session-reference",
        client_turn_id="turn-reference-stream",
        question="公式5讲了什么",
        scope="selected",
        document_ids=[document.id],
    )
    key_store = SimpleNamespace(get_key=lambda session_id: "sk-" + "x" * 32)

    response = asyncio.run(
        chat_api._answer_question(
            request,
            session,
            Settings(),
            key_store,
            emit,
        )
    )

    event_names = [event for event, _ in events]
    event_messages = " ".join(str(payload.get("message", "")) for _, payload in events)
    assert "reference_resolution" in event_names
    assert "formula_source" in event_names
    assert "关键实体未能链接" not in event_messages
    assert response.action == "answer"
    assert response.formula_assets[0].formula_number == "5"
