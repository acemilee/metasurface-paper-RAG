from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from paper_rag.db import Base


class AnswerAudit(Base):
    __tablename__ = "answer_audits"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    evidence_status: Mapped[str] = mapped_column(String(64), default="insufficient")
    refusal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    hallucination_risk: Mapped[str] = mapped_column(String(32), default="unknown")
    audit_result: Mapped[str] = mapped_column(String(64), default="not_run")
    action: Mapped[str] = mapped_column(String(32), default="refuse")
    unsupported_parts_json: Mapped[str] = mapped_column(Text, default="[]")
    citation_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    model_name: Mapped[str] = mapped_column(String(128), default="deepseek-v4-flash")
    prompt_version: Mapped[str] = mapped_column(String(64), default="grounded-answer-v1")
    selected_document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    query_plan_json: Mapped[str] = mapped_column(Text, default="{}")
    entity_links_json: Mapped[str] = mapped_column(Text, default="[]")
    rewrite_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rewrite_error_json: Mapped[str] = mapped_column(Text, default="{}")
    document_genres_json: Mapped[str] = mapped_column(Text, default="{}")
    generation_attempts_json: Mapped[str] = mapped_column(Text, default="[]")
    semantic_audit_json: Mapped[str] = mapped_column(Text, default="[]")
