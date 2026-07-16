from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from paper_rag.db import Base


class Formula(Base):
    __tablename__ = "formulas"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    placeholder: Mapped[str] = mapped_column(String(255), unique=True)
    bbox_json: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_before: Mapped[str] = mapped_column(Text, default="")
    context_after: Mapped[str] = mapped_column(Text, default="")
    physical_meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
    semantic_status: Mapped[str] = mapped_column(String(64), default="insufficient_context")
    formula_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    group_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    formula_group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("formula_groups.id", ondelete="SET NULL"), nullable=True, index=True
    )
    part_index: Mapped[int] = mapped_column(Integer, default=0)
    parser_version: Mapped[str] = mapped_column(String(64), default="formula-layout-v3")
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    fidelity_status: Mapped[str] = mapped_column(String(32), default="needs_review")
    latex_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    latex_verification_status: Mapped[str] = mapped_column(String(32), default="absent")
    latex_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    source_crop_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
