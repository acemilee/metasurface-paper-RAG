from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.paper_profile import PaperProfile, PaperProfileClaim
from paper_rag.schemas.query_plan import EvidenceType

PROFILE_PARSER_VERSION = "extractive-profile-v2"
PROFILE_PROMPT_VERSION = "no-llm-extractive-v1"

VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|GHz|THz|MHz|kHz|Hz|nm|[μu]m|mm|cm|V|mV|K|dB|Ω(?:/sq)?|ohm(?:/sq)?)\b",
    re.IGNORECASE,
)
CAPTION_PATTERN = re.compile(
    r"(?:Fig(?:ure)?\.?\s*\d+|Table\s*(?:[IVX]+|\d+)|图\s*\d+|表\s*\d+)[^\n。.!?]{0,220}",
    re.IGNORECASE,
)
FORMULA_PLACEHOLDER_PATTERN = re.compile(r"(?:公式|formula)_placeholder_[A-Za-z0-9_-]+", re.IGNORECASE)
MECHANISM_MARKERS = (
    "because", "due to", "therefore", "resulting in", "leads to", "mechanism",
    "由于", "因此", "从而", "导致", "机理", "机制",
)


def _source_hash(document: Document, chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    digest.update(PROFILE_PARSER_VERSION.encode("ascii"))
    digest.update(document.file_sha256.encode("ascii"))
    for chunk in chunks:
        digest.update(str(chunk.id).encode("ascii"))
        digest.update(str(chunk.page_start).encode("ascii"))
        digest.update((chunk.section_path or "").encode("utf-8"))
        digest.update(chunk.content.encode("utf-8"))
    return digest.hexdigest()


def _chunk_roles(chunk: Chunk) -> set[str]:
    section = (chunk.section_path or "").lower()
    text = chunk.content.lower()
    roles: set[str] = set()
    if "abstract" in section or "摘要" in section or chunk.chunk_index == 0:
        roles.add(EvidenceType.OVERVIEW.value)
    if any(item in section for item in ("introduction", "background", "引言", "绪论")):
        roles.add(EvidenceType.OVERVIEW.value)
    if any(item in text for item in ("lack ", "challenge", "limitation of", "existing ", "however", "不足", "挑战", "现有")):
        roles.add(EvidenceType.PROBLEM_OR_GAP.value)
    if any(item in section for item in ("method", "design", "model", "structure", "方法", "设计", "模型", "结构")):
        roles.add(EvidenceType.METHOD_OR_STRUCTURE.value)
    if any(item in text for item in ("we propose", "we present", "we demonstrate", "novel", "proposed", "提出", "创新", "首次")):
        roles.add(EvidenceType.NOVELTY_CLAIM.value)
    if any(item in section for item in ("result", "discussion", "performance", "结果", "讨论", "性能")):
        roles.add(EvidenceType.RESULT_OR_ADVANTAGE.value)
    if any(item in text for item in ("measured", "measurement", "experiment", "fabricated", "测量", "实验", "制备")):
        roles.add(EvidenceType.EXPERIMENT.value)
    if any(item in section or item in text for item in ("comparison", "compared", "versus", "对比", "比较")):
        roles.add(EvidenceType.COMPARISON_BASELINE.value)
    if any(item in section for item in ("conclusion", "summary", "结论", "总结")):
        roles.add(EvidenceType.CONCLUSION.value)
    if any(item in text for item in ("future work", "remains", "limitation", "challenge", "未来", "局限", "仍需")):
        roles.add(EvidenceType.LIMITATION.value)
    if chunk.formula_ids_json and chunk.formula_ids_json != "[]":
        roles.add(EvidenceType.FORMULA_CONTEXT.value)
    if any(item in text for item in ("ghz", "thz", "mhz", "voltage", "bias", "temperature", "方阻", "偏压", "温度")):
        roles.add(EvidenceType.OPERATING_CONDITIONS.value)
    return roles or {EvidenceType.GENERAL.value}


def _entry(chunk: Chunk) -> dict:
    return {
        "chunk_id": str(chunk.id),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_path": chunk.section_path,
        "text": chunk.content[:1200],
    }


def _build_content(document: Document, chunks: list[Chunk]) -> tuple[dict, list[tuple[str, dict]]]:
    role_index: dict[str, list[dict]] = defaultdict(list)
    claims: list[tuple[str, dict]] = []
    fact_ledger: list[dict] = []
    formula_index: list[dict] = []
    figure_table_index: list[dict] = []
    mechanism_statements: list[dict] = []
    for chunk in chunks:
        entry = _entry(chunk)
        for role in _chunk_roles(chunk):
            if len(role_index[role]) < 4:
                role_index[role].append(entry)
                claims.append((role, entry))
        for match in VALUE_PATTERN.finditer(chunk.content):
            context_start = max(0, match.start() - 100)
            context_end = min(len(chunk.content), match.end() + 140)
            fact_ledger.append(
                {
                    "value_text": match.group(0),
                    "context": chunk.content[context_start:context_end],
                    "chunk_id": str(chunk.id),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "audit_verdict": "exact_extract",
                }
            )
        for placeholder in FORMULA_PLACEHOLDER_PATTERN.findall(chunk.content):
            formula_index.append({"placeholder": placeholder, **entry})
        for caption in CAPTION_PATTERN.findall(chunk.content):
            figure_table_index.append({"caption": caption.strip(), **entry})
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", chunk.content):
            if any(marker in sentence.lower() for marker in MECHANISM_MARKERS):
                mechanism_statements.append({"statement": sentence[:1200], **entry})
    section_tree = list(dict.fromkeys(chunk.section_path for chunk in chunks if chunk.section_path))
    digest_role_map = {
        "background": [EvidenceType.OVERVIEW.value, EvidenceType.PROBLEM_OR_GAP.value],
        "innovation": [EvidenceType.NOVELTY_CLAIM.value],
        "method": [EvidenceType.METHOD_OR_STRUCTURE.value],
        "results": [EvidenceType.RESULT_OR_ADVANTAGE.value],
        "validation": [EvidenceType.EXPERIMENT.value],
        "comparison": [EvidenceType.COMPARISON_BASELINE.value],
        "limitations": [EvidenceType.LIMITATION.value],
        "conclusion": [EvidenceType.CONCLUSION.value],
    }
    digest = {}
    for field, roles in digest_role_map.items():
        digest[field] = [entry for role in roles for entry in role_index.get(role, [])[:2]]
    return (
        {
            "document_id": str(document.id),
            "title": document.original_filename,
            "document_genre": document.document_genre,
            "page_count": document.page_count,
            "section_tree": section_tree,
            "role_index": dict(role_index),
            "structured_digest": digest,
            "fact_ledger": fact_ledger[:200],
            "formula_index": formula_index[:100],
            "figure_table_index": figure_table_index[:100],
            "mechanism_statements": mechanism_statements[:100],
            "evidence_policy": "navigation_only_original_chunks_required",
        },
        claims,
    )


def get_ready_profile(session: Session, document_id: UUID) -> PaperProfile | None:
    return session.scalar(
        select(PaperProfile)
        .where(PaperProfile.document_id == document_id, PaperProfile.status == "ready")
        .order_by(PaperProfile.profile_version.desc())
        .limit(1)
    )


def build_paper_profile(session: Session, document_id: UUID) -> PaperProfile:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.status != DocumentStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document is not completed")
    chunks = list(
        session.scalars(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
        )
    )
    if not chunks:
        raise ValueError("Cannot build a Paper Profile without chunks")
    source_sha256 = _source_hash(document, chunks)
    current = get_ready_profile(session, document.id)
    if current is not None and current.source_sha256 == source_sha256:
        return current
    next_version = (
        session.scalar(
            select(func.coalesce(func.max(PaperProfile.profile_version), 0)).where(
                PaperProfile.document_id == document.id
            )
        )
        or 0
    ) + 1
    profile = PaperProfile(
        document_id=document.id,
        status="building",
        profile_version=next_version,
        parser_version=PROFILE_PARSER_VERSION,
        prompt_version=PROFILE_PROMPT_VERSION,
        source_sha256=source_sha256,
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    try:
        content, claims = _build_content(document, chunks)
        profile.content_json = json.dumps(content, ensure_ascii=False)
        profile.claims = [
            PaperProfileClaim(
                claim_type=role,
                claim_text=entry["text"],
                citation_ids_json=json.dumps([entry["chunk_id"]]),
                audit_verdict="exact_extract",
                evidence_roles_json=json.dumps([role]),
                confidence=1.0,
            )
            for role, entry in claims
        ]
        profile.status = "ready"
        profile.error_code = None
        profile.error_message = None
        session.flush()
        if current is not None:
            current.status = "stale"
        session.commit()
        session.refresh(profile)
        return profile
    except Exception as exc:
        session.rollback()
        failed = session.get(PaperProfile, profile.id)
        if failed is not None:
            failed.status = "failed"
            failed.error_code = type(exc).__name__
            failed.error_message = str(exc)
            session.commit()
        raise


def get_profile_retrieval_hints(
    session: Session,
    document_ids: list[UUID],
    required_roles: list[EvidenceType],
    *,
    max_hints: int = 8,
) -> list[tuple[str, str]]:
    hints: list[tuple[str, str]] = []
    for document_id in document_ids:
        profile = get_ready_profile(session, document_id)
        if profile is None:
            continue
        content = json.loads(profile.content_json or "{}")
        role_index = content.get("role_index", {})
        for role in required_roles:
            entries = role_index.get(role.value, [])
            if entries:
                hints.append((entries[0]["text"][:500], role.value))
                if len(hints) >= max_hints:
                    return hints
    return hints


def backfill_paper_profiles(session: Session, document_ids: list[UUID]) -> list[PaperProfile]:
    return [build_paper_profile(session, document_id) for document_id in document_ids]
