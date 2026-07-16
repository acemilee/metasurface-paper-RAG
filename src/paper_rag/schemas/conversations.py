from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationCreate(BaseModel):
    title: str = Field(default="新研究会话", min_length=1, max_length=200)
    scope: Literal["all", "selected"] = "all"
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation title cannot be empty")
        return value


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation title cannot be empty")
        return value


class ConversationMessageResponse(BaseModel):
    message_id: UUID
    turn_index: int
    role: Literal["user", "assistant"]
    content: str
    status: str
    created_at: datetime
    response: dict | None = None


class ConversationSummary(BaseModel):
    conversation_id: UUID
    title: str
    scope: Literal["all", "selected"]
    document_ids: list[UUID]
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessageResponse]
    summary: dict = Field(default_factory=dict)


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
