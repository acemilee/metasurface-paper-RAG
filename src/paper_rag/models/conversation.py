from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_rag.db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    scope: Mapped[str] = mapped_column(String(16), default="all")
    selected_document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    entities: Mapped[list["ConversationEntity"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_index", "role", name="uq_conversation_message_turn_role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_index", name="uq_conversation_turn_index"),
        UniqueConstraint("conversation_id", "client_turn_id", name="uq_conversation_client_turn"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer)
    client_turn_id: Mapped[str] = mapped_column(String(128))
    original_question: Mapped[str] = mapped_column(Text)
    standalone_question: Mapped[str] = mapped_column(Text, default="")
    scope: Mapped[str] = mapped_column(String(16), default="all")
    selected_document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    query_plan_json: Mapped[str] = mapped_column(Text, default="{}")
    entity_links_json: Mapped[str] = mapped_column(Text, default="[]")
    citation_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    action: Mapped[str] = mapped_column(String(32), default="error")
    audit_result: Mapped[str] = mapped_column(String(64), default="not_run")
    status: Mapped[str] = mapped_column(String(32), default="running")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    response_json: Mapped[str] = mapped_column(Text, default="")
    question_embedding_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class ConversationEntity(Base):
    __tablename__ = "conversation_entities"
    __table_args__ = (
        UniqueConstraint("conversation_id", "entity_type", "canonical", name="uq_conversation_entity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    entity_type: Mapped[str] = mapped_column(String(64))
    canonical: Mapped[str] = mapped_column(String(512))
    surface: Mapped[str] = mapped_column(String(512))
    document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    last_turn_index: Mapped[int] = mapped_column(Integer)

    conversation: Mapped[Conversation] = relationship(back_populates="entities")
