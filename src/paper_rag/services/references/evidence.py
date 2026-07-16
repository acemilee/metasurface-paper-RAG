from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.services.references.types import ReferenceResolution, ResolutionStatus
from paper_rag.services.retrieval import RetrievedChunk


def chunk_to_retrieved(chunk: Chunk, *, score: float = 1.0) -> RetrievedChunk:
    try:
        formula_ids = [str(item) for item in json.loads(chunk.formula_ids_json or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        formula_ids = []
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        formula_ids=formula_ids,
        score=score,
        quality_score=chunk.quality_score,
        has_low_confidence_ocr=chunk.has_low_confidence_ocr,
        retrieval_roles=("resolved_reference",),
    )


def merge_resolved_reference_evidence(
    session: Session,
    retrieved: Sequence[RetrievedChunk],
    resolutions: Iterable[ReferenceResolution],
) -> list[RetrievedChunk]:
    pinned_ids = []
    seen_ids = set()
    for resolution in resolutions:
        if resolution.status != ResolutionStatus.RESOLVED:
            continue
        for chunk_id in resolution.evidence_chunk_ids:
            if chunk_id not in seen_ids:
                pinned_ids.append(chunk_id)
                seen_ids.add(chunk_id)
    if not pinned_ids:
        return list(retrieved)
    chunks = {
        item.id: item
        for item in session.scalars(
            select(Chunk).where(Chunk.id.in_(pinned_ids))
        )
    }
    pinned = [
        chunk_to_retrieved(chunks[item], score=1.0)
        for item in pinned_ids
        if item in chunks
    ]
    pinned_chunk_ids = {item.chunk_id for item in pinned}
    return [
        *pinned,
        *(item for item in retrieved if item.chunk_id not in pinned_chunk_ids),
    ]
