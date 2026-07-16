from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paper_rag.models.conversation import (
    Conversation,
    ConversationEntity,
    ConversationMessage,
    ConversationTurn,
)
from paper_rag.schemas.chat import AnswerResponse, ChatRequest


@dataclass(frozen=True)
class ConversationTurnStart:
    turn: ConversationTurn | None
    replayed_response: AnswerResponse | None = None


def _get_conversation(session: Session, conversation_id: UUID) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


def begin_conversation_turn(session: Session, request: ChatRequest) -> ConversationTurnStart:
    if request.conversation_id is None:
        return ConversationTurnStart(None)
    conversation = _get_conversation(session, request.conversation_id)
    existing = session.scalar(
        select(ConversationTurn).where(
            ConversationTurn.conversation_id == conversation.id,
            ConversationTurn.client_turn_id == request.client_turn_id,
        )
    )
    if existing is not None:
        if existing.status == "completed" and existing.response_json:
            return ConversationTurnStart(
                existing,
                AnswerResponse.model_validate_json(existing.response_json),
            )
        if existing.status == "failed" and existing.response_json:
            return ConversationTurnStart(
                existing,
                AnswerResponse.model_validate_json(existing.response_json),
            )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conversation turn is already running")

    next_index = (
        session.scalar(
            select(func.coalesce(func.max(ConversationTurn.turn_index), 0)).where(
                ConversationTurn.conversation_id == conversation.id
            )
        )
        or 0
    ) + 1
    turn = ConversationTurn(
        conversation_id=conversation.id,
        turn_index=next_index,
        client_turn_id=request.client_turn_id,
        original_question=request.question,
        scope=request.scope or conversation.scope,
        selected_document_ids_json=json.dumps(
            [str(item) for item in (request.document_ids or _decode_ids(conversation.selected_document_ids_json))]
        ),
    )
    session.add(turn)
    session.add(
        ConversationMessage(
            conversation_id=conversation.id,
            turn_index=next_index,
            role="user",
            content=request.question,
            status="running",
        )
    )
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Duplicate conversation turn") from exc
    session.refresh(turn)
    return ConversationTurnStart(turn)


def _decode_ids(value: str) -> list[UUID]:
    return [UUID(item) for item in json.loads(value or "[]")]


def resolve_conversation_request(session: Session, request: ChatRequest) -> ChatRequest:
    if request.conversation_id is None:
        return request.model_copy(update={"scope": request.scope or "all"})
    conversation = _get_conversation(session, request.conversation_id)
    if request.scope is None:
        return request.model_copy(
            update={
                "scope": conversation.scope,
                "document_ids": _decode_ids(conversation.selected_document_ids_json),
            }
        )
    if request.scope == "all":
        return request.model_copy(update={"document_ids": []})
    if not request.document_ids:
        return request.model_copy(
            update={"document_ids": _decode_ids(conversation.selected_document_ids_json)}
        )
    return request


def build_conversation_context(
    session: Session,
    conversation_id: UUID | None,
    *,
    recent_message_limit: int = 16,
    character_budget: int = 12000,
    query_embedding: list[float] | None = None,
    relevant_turn_limit: int = 4,
) -> dict[str, Any] | None:
    if conversation_id is None:
        return None
    conversation = _get_conversation(session, conversation_id)
    messages = list(
        session.scalars(
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation.id,
                ConversationMessage.status == "completed",
            )
            .order_by(ConversationMessage.turn_index.desc(), ConversationMessage.created_at.desc())
            .limit(recent_message_limit)
        )
    )
    messages.reverse()
    bounded: list[dict[str, Any]] = []
    used = 0
    recent_budget = max(1, int(character_budget * 0.7))
    for message in reversed(messages):
        remaining = recent_budget - used
        if remaining <= 0:
            break
        content = message.content[:remaining]
        bounded.append(
            {
                "turn_index": message.turn_index,
                "role": message.role,
                "content": content,
            }
        )
        used += len(content)
    bounded.reverse()
    recent_turn_indexes = {item["turn_index"] for item in bounded}
    relevant_history: list[dict[str, Any]] = []
    if query_embedding:
        candidates = list(
            session.scalars(
                select(ConversationTurn).where(
                    ConversationTurn.conversation_id == conversation.id,
                    ConversationTurn.status == "completed",
                    ConversationTurn.question_embedding_json != "[]",
                    ConversationTurn.turn_index.not_in(recent_turn_indexes),
                )
            )
        )

        def similarity(turn: ConversationTurn) -> float:
            vector = json.loads(turn.question_embedding_json or "[]")
            if len(vector) != len(query_embedding):
                return -1.0
            dot = sum(left * right for left, right in zip(query_embedding, vector, strict=True))
            norm = math.sqrt(sum(value * value for value in vector)) * math.sqrt(
                sum(value * value for value in query_embedding)
            )
            return dot / norm if norm else -1.0

        for turn in sorted(candidates, key=similarity, reverse=True)[:relevant_turn_limit]:
            remaining = character_budget - used
            if remaining <= 0:
                break
            turn_messages = list(
                session.scalars(
                    select(ConversationMessage)
                    .where(
                        ConversationMessage.conversation_id == conversation.id,
                        ConversationMessage.turn_index == turn.turn_index,
                        ConversationMessage.status == "completed",
                    )
                    .order_by(ConversationMessage.created_at)
                )
            )
            history_messages = []
            for message in turn_messages:
                remaining = character_budget - used
                if remaining <= 0:
                    break
                content = message.content[: min(1200, remaining)]
                history_messages.append({"role": message.role, "content": content})
                used += len(content)
            if history_messages:
                relevant_history.append(
                    {
                        "turn_index": turn.turn_index,
                        "standalone_question": turn.standalone_question,
                        "messages": history_messages,
                        "similarity": round(similarity(turn), 4),
                    }
                )
    return {
        "conversation_id": str(conversation.id),
        "scope": conversation.scope,
        "selected_document_ids": json.loads(conversation.selected_document_ids_json or "[]"),
        "recent_messages_untrusted_data": bounded,
        "relevant_history_untrusted_data": relevant_history,
        "conversation_summary_untrusted_data": json.loads(conversation.summary_json or "{}"),
    }


def _serialize_entity(entity: Any) -> dict[str, Any]:
    return {
        "surface": str(getattr(entity, "surface", "")),
        "canonical": str(getattr(entity, "canonical", "") or getattr(entity, "surface", "")),
        "entity_type": str(getattr(entity, "entity_type", "other")),
        "linked": bool(getattr(entity, "linked", False)),
        "matched_document_ids": [
            str(item) for item in getattr(entity, "matched_document_ids", [])
        ],
    }


def complete_conversation_turn(
    session: Session,
    request: ChatRequest,
    response: AnswerResponse,
    query_plan: Any | None,
    linked_entities: list[Any],
    question_embedding: list[float] | None = None,
) -> None:
    if request.conversation_id is None:
        return
    conversation = _get_conversation(session, request.conversation_id)
    turn = session.scalar(
        select(ConversationTurn).where(
            ConversationTurn.conversation_id == conversation.id,
            ConversationTurn.client_turn_id == request.client_turn_id,
        )
    )
    if turn is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conversation turn was not started")
    if turn.status in {"completed", "failed"}:
        return

    failed = response.action == "error"
    turn.standalone_question = (
        query_plan.standalone_question if query_plan is not None else request.question
    )
    turn.query_plan_json = json.dumps(
        query_plan.model_dump(mode="json") if query_plan is not None else {}, ensure_ascii=False
    )
    serialized_entities = (
        json.loads(json.dumps(response.entity_links, ensure_ascii=False, default=str))
        if response.entity_links
        else [_serialize_entity(item) for item in linked_entities]
    )
    turn.entity_links_json = json.dumps(serialized_entities, ensure_ascii=False)
    turn.citation_ids_json = json.dumps([str(item.citation_id) for item in response.citations])
    turn.action = response.action
    turn.audit_result = response.audit_result
    turn.status = "failed" if failed else "completed"
    turn.error_code = response.audit_result if failed else None
    turn.response_json = response.model_dump_json()
    turn.question_embedding_json = json.dumps(question_embedding or [])
    turn.completed_at = datetime.now().astimezone()

    user_message = session.scalar(
        select(ConversationMessage).where(
            ConversationMessage.conversation_id == conversation.id,
            ConversationMessage.turn_index == turn.turn_index,
            ConversationMessage.role == "user",
        )
    )
    if user_message is not None:
        user_message.status = turn.status
    session.add(
        ConversationMessage(
            conversation_id=conversation.id,
            turn_index=turn.turn_index,
            role="assistant",
            content=response.answer,
            status=turn.status,
        )
    )
    conversation.scope = request.scope or conversation.scope
    if request.document_ids or conversation.scope == "selected":
        conversation.selected_document_ids_json = json.dumps(
            [str(item) for item in request.document_ids]
        )
    conversation.updated_at = datetime.now().astimezone()
    if conversation.title == "新研究会话" and turn.turn_index == 1:
        conversation.title = request.question[:80]

    for item in serialized_entities:
        if not item["linked"] or not item["canonical"]:
            continue
        existing = session.scalar(
            select(ConversationEntity).where(
                ConversationEntity.conversation_id == conversation.id,
                ConversationEntity.entity_type == item["entity_type"],
                ConversationEntity.canonical == item["canonical"],
            )
        )
        if existing is None:
            existing = ConversationEntity(
                conversation_id=conversation.id,
                entity_type=item["entity_type"],
                canonical=item["canonical"],
                surface=item["surface"],
                last_turn_index=turn.turn_index,
            )
            session.add(existing)
        existing.surface = item["surface"]
        existing.document_ids_json = json.dumps(item["matched_document_ids"])
        existing.last_turn_index = turn.turn_index
    session.flush()
    completed_turns = list(
        session.scalars(
            select(ConversationTurn)
            .where(
                ConversationTurn.conversation_id == conversation.id,
                ConversationTurn.status == "completed",
            )
            .order_by(ConversationTurn.turn_index.desc())
            .limit(12)
        )
    )
    unresolved = [
        item.standalone_question or item.original_question
        for item in completed_turns
        if item.action in {"clarify", "refuse"}
    ][:5]
    conversation.summary_json = json.dumps(
        {
            "turn_count": session.scalar(
                select(func.count(ConversationTurn.id)).where(
                    ConversationTurn.conversation_id == conversation.id,
                    ConversationTurn.status == "completed",
                )
            )
            or 0,
            "active_scope": conversation.scope,
            "selected_document_ids": json.loads(conversation.selected_document_ids_json or "[]"),
            "recent_tasks": [
                item.standalone_question or item.original_question for item in completed_turns[:8]
            ],
            "unresolved_questions": unresolved,
            "active_entities": [
                {
                    "entity_type": item.entity_type,
                    "canonical": item.canonical,
                    "document_ids": json.loads(item.document_ids_json or "[]"),
                }
                for item in session.scalars(
                    select(ConversationEntity)
                    .where(ConversationEntity.conversation_id == conversation.id)
                    .order_by(ConversationEntity.last_turn_index.desc())
                    .limit(20)
                )
            ],
            "recent_citation_ids": list(
                dict.fromkeys(
                    citation_id
                    for item in completed_turns[:8]
                    for citation_id in json.loads(item.citation_ids_json or "[]")
                )
            ),
            "memory_policy": "context_only_not_evidence",
        },
        ensure_ascii=False,
    )
    conversation.summary_version += 1
    session.commit()


def fail_running_conversation_turn(
    session: Session,
    request: ChatRequest,
    *,
    reason: str = "问答服务异常",
    audit_result: str = "unhandled_question_failure",
) -> AnswerResponse | None:
    if request.conversation_id is None:
        return None
    turn = session.scalar(
        select(ConversationTurn).where(
            ConversationTurn.conversation_id == request.conversation_id,
            ConversationTurn.client_turn_id == request.client_turn_id,
        )
    )
    if turn is None or turn.status != "running":
        return None
    response = AnswerResponse(
        answer=reason,
        citations=[],
        evidence_status="insufficient",
        refused=True,
        refusal_reason=reason,
        hallucination_risk="unknown",
        audit_result=audit_result,
        action="error",
    )
    complete_conversation_turn(session, request, response, None, [])
    return response
