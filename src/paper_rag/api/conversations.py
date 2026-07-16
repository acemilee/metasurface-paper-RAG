from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from paper_rag.db import get_db_session
from paper_rag.models.conversation import Conversation, ConversationMessage, ConversationTurn
from paper_rag.schemas.conversations import (
    ConversationCreate,
    ConversationDetail,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationSummary,
    ConversationUpdate,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _decode_ids(value: str) -> list[UUID]:
    return [UUID(item) for item in json.loads(value or "[]")]


def _summary(session: Session, conversation: Conversation) -> ConversationSummary:
    message_count = session.scalar(
        select(func.count(ConversationMessage.id)).where(
            ConversationMessage.conversation_id == conversation.id
        )
    ) or 0
    return ConversationSummary(
        conversation_id=conversation.id,
        title=conversation.title,
        scope=conversation.scope,
        document_ids=_decode_ids(conversation.selected_document_ids_json),
        message_count=message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _detail(session: Session, conversation: Conversation) -> ConversationDetail:
    messages = list(
        session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.turn_index, ConversationMessage.created_at)
        )
    )
    turns = {
        item.turn_index: item
        for item in session.scalars(
            select(ConversationTurn).where(ConversationTurn.conversation_id == conversation.id)
        )
    }
    summary = _summary(session, conversation)
    return ConversationDetail(
        **summary.model_dump(),
        messages=[
            ConversationMessageResponse(
                message_id=message.id,
                turn_index=message.turn_index,
                role=message.role,
                content=message.content,
                status=message.status,
                created_at=message.created_at,
                response=(
                    json.loads(turns[message.turn_index].response_json)
                    if message.role == "assistant"
                    and message.turn_index in turns
                    and turns[message.turn_index].response_json
                    else None
                ),
            )
            for message in messages
        ],
        summary=json.loads(conversation.summary_json or "{}"),
    )


def _get_conversation(session: Session, conversation_id: UUID) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreate,
    session: Session = Depends(get_db_session),
) -> ConversationSummary:
    conversation = Conversation(
        title=request.title,
        scope=request.scope,
        selected_document_ids_json=json.dumps([str(item) for item in request.document_ids]),
    )
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return _summary(session, conversation)


@router.get("", response_model=ConversationListResponse)
def list_conversations(session: Session = Depends(get_db_session)) -> ConversationListResponse:
    conversations = list(
        session.scalars(select(Conversation).order_by(Conversation.updated_at.desc(), Conversation.created_at.desc()))
    )
    return ConversationListResponse(items=[_summary(session, item) for item in conversations])


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
) -> ConversationDetail:
    return _detail(session, _get_conversation(session, conversation_id))


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def update_conversation(
    conversation_id: UUID,
    request: ConversationUpdate,
    session: Session = Depends(get_db_session),
) -> ConversationSummary:
    conversation = _get_conversation(session, conversation_id)
    conversation.title = request.title
    session.commit()
    session.refresh(conversation)
    return _summary(session, conversation)


@router.post("/{conversation_id}/reset", response_model=ConversationDetail)
def reset_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
) -> ConversationDetail:
    conversation = _get_conversation(session, conversation_id)
    session.execute(delete(ConversationMessage).where(ConversationMessage.conversation_id == conversation.id))
    conversation.turns.clear()
    conversation.entities.clear()
    conversation.summary_json = "{}"
    conversation.summary_version = 0
    session.commit()
    session.refresh(conversation)
    return _detail(session, conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
) -> Response:
    conversation = _get_conversation(session, conversation_id)
    session.delete(conversation)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
