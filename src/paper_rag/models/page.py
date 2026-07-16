from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_rag.db import Base


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_pages_document_page"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    extraction_method: Mapped[str] = mapped_column(String(32), default="digital_text")
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="pages")
    blocks: Mapped[list["TextBlock"]] = relationship(back_populates="page", cascade="all, delete-orphan")


class TextBlock(Base):
    __tablename__ = "text_blocks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    reading_order: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    x0: Mapped[float] = mapped_column(Float)
    y0: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="digital_text")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    page: Mapped[Page] = relationship(back_populates="blocks")
