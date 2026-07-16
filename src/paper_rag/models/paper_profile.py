from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from paper_rag.db import Base


class PaperProfile(Base):
    __tablename__ = "paper_profiles"
    __table_args__ = (
        UniqueConstraint("document_id", "profile_version", name="uq_paper_profile_document_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="building", index=True)
    profile_version: Mapped[int] = mapped_column(Integer)
    parser_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    source_sha256: Mapped[str] = mapped_column(String(64), index=True)
    content_json: Mapped[str] = mapped_column(Text, default="{}")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    claims: Mapped[list["PaperProfileClaim"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )
    relations: Mapped[list["PaperProfileRelation"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


class PaperProfileClaim(Base):
    __tablename__ = "paper_profile_claims"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_profiles.id", ondelete="CASCADE"), index=True
    )
    claim_type: Mapped[str] = mapped_column(String(64), index=True)
    claim_text: Mapped[str] = mapped_column(Text)
    citation_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    audit_verdict: Mapped[str] = mapped_column(String(64))
    evidence_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    profile: Mapped[PaperProfile] = relationship(back_populates="claims")

    @property
    def citation_ids(self) -> list[uuid.UUID]:
        return [uuid.UUID(item) for item in json.loads(self.citation_ids_json or "[]")]


class PaperProfileRelation(Base):
    __tablename__ = "paper_profile_relations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_profiles.id", ondelete="CASCADE"), index=True
    )
    source_entity: Mapped[str] = mapped_column(String(512))
    relation: Mapped[str] = mapped_column(String(128))
    target_entity: Mapped[str] = mapped_column(String(512))
    conditions_json: Mapped[str] = mapped_column(Text, default="[]")
    citation_ids_json: Mapped[str] = mapped_column(Text, default="[]")

    profile: Mapped[PaperProfile] = relationship(back_populates="relations")
