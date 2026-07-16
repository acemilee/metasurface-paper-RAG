from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from paper_rag.db import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    vector_id: Mapped[str] = mapped_column(String(255), unique=True)
    content: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str] = mapped_column(String(64), default="paragraph")
    formula_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    chunk_index: Mapped[int] = mapped_column(Integer)
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    has_low_confidence_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
