from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document
from paper_rag.models.page import Page
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.services.references.parser import parse_typed_references
from paper_rag.services.references.types import (
    ReferenceKind,
    ReferenceResolution,
    ResolutionStatus,
    TypedReference,
)


def _resolve_page(
    session: Session,
    reference: TypedReference,
    scope: tuple[UUID, ...],
) -> ReferenceResolution:
    number = int(reference.normalized_key)
    pages = list(
        session.scalars(
            select(Page).where(
                Page.document_id.in_(scope),
                Page.page_number == number,
            )
        )
    )
    if not pages:
        return ReferenceResolution(
            reference,
            ResolutionStatus.NOT_FOUND,
            scope,
            resolution_source="page.page_number",
        )
    if len(pages) > 1:
        return ReferenceResolution(
            reference,
            ResolutionStatus.AMBIGUOUS,
            tuple(sorted({item.document_id for item in pages}, key=str)),
            tuple(sorted((item.id for item in pages), key=str)),
            resolution_source="page.page_number",
        )
    page = pages[0]
    chunks = tuple(
        session.scalars(
            select(Chunk.id)
            .where(
                Chunk.document_id == page.document_id,
                Chunk.page_start <= number,
                Chunk.page_end >= number,
            )
            .order_by(Chunk.chunk_index, Chunk.id)
        )
    )
    status = (
        ResolutionStatus.RESOLVED
        if chunks
        else ResolutionStatus.INDEX_INCONSISTENT
    )
    return ReferenceResolution(
        reference,
        status,
        (page.document_id,),
        (page.id,),
        chunks,
        "page.page_number",
        {"page_number": number},
    )


def _resolve_section(
    session: Session,
    reference: TypedReference,
    scope: tuple[UUID, ...],
) -> ReferenceResolution:
    boundary = re.compile(
        rf"^\s*{re.escape(reference.normalized_key)}(?:\D|$)",
        re.IGNORECASE,
    )
    candidates = [
        chunk
        for chunk in session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id.in_(scope),
                Chunk.section_path.is_not(None),
            )
            .order_by(Chunk.document_id, Chunk.chunk_index, Chunk.id)
        )
        if chunk.section_path and boundary.match(chunk.section_path)
    ]
    if not candidates:
        return ReferenceResolution(
            reference,
            ResolutionStatus.NOT_FOUND,
            scope,
            resolution_source="chunk.section_path",
        )
    document_ids = tuple(
        sorted({item.document_id for item in candidates}, key=str)
    )
    chunk_ids = tuple(item.id for item in candidates)
    status = (
        ResolutionStatus.AMBIGUOUS
        if len(document_ids) > 1
        else ResolutionStatus.RESOLVED
    )
    return ReferenceResolution(
        reference,
        status,
        document_ids,
        chunk_ids,
        chunk_ids,
        "chunk.section_path",
        {"section_number": reference.normalized_key},
    )


def _reference_matches(candidate: TypedReference, requested: TypedReference) -> bool:
    return (
        candidate.kind == requested.kind
        and candidate.normalized_key == requested.normalized_key
        and (
            requested.qualifier is None
            or candidate.qualifier == requested.qualifier
        )
    )


CAPTION_LINE = re.compile(
    r"^(?:figure\b|fig\.?\b|图|table\b|表)",
    re.IGNORECASE,
)


def caption_references(content: str) -> tuple[TypedReference, ...]:
    references: list[TypedReference] = []
    for line in content.splitlines() or [content]:
        stripped = line.strip()
        if CAPTION_LINE.match(stripped):
            references.extend(parse_typed_references(stripped))
    return tuple(references)


def _resolve_figure_or_table(
    session: Session,
    reference: TypedReference,
    scope: tuple[UUID, ...],
) -> ReferenceResolution:
    profiles = list(
        session.scalars(
            select(PaperProfile)
            .where(
                PaperProfile.document_id.in_(scope),
                PaperProfile.status == "ready",
            )
            .order_by(PaperProfile.document_id, PaperProfile.profile_version.desc())
        )
    )
    latest: dict[UUID, PaperProfile] = {}
    for profile in profiles:
        latest.setdefault(profile.document_id, profile)

    matched_chunk_ids: list[UUID] = []
    matched_document_ids: set[UUID] = set()
    missing_chunk_ids: list[str] = []
    mismatched_chunk_ids: list[str] = []
    matched_profile_entry = False
    for document_id, profile in latest.items():
        try:
            entries = json.loads(profile.content_json or "{}").get(
                "figure_table_index",
                [],
            )
        except (TypeError, json.JSONDecodeError):
            entries = []
        for entry in entries:
            caption = str(entry.get("caption") or "")
            profile_references = parse_typed_references(caption)
            if not any(
                _reference_matches(candidate, reference)
                for candidate in profile_references
            ):
                continue
            matched_profile_entry = True
            raw_chunk_id = str(entry.get("chunk_id") or "")
            try:
                chunk_id = UUID(raw_chunk_id)
            except ValueError:
                missing_chunk_ids.append(raw_chunk_id)
                continue
            chunk = session.scalar(
                select(Chunk).where(
                    Chunk.id == chunk_id,
                    Chunk.document_id == document_id,
                )
            )
            if chunk is None:
                missing_chunk_ids.append(raw_chunk_id)
                continue
            if not any(
                _reference_matches(candidate, reference)
                for candidate in parse_typed_references(chunk.content)
            ):
                mismatched_chunk_ids.append(raw_chunk_id)
                continue
            matched_chunk_ids.append(chunk.id)
            matched_document_ids.add(document_id)

    if matched_chunk_ids:
        chunk_ids = tuple(sorted(set(matched_chunk_ids), key=str))
        document_ids = tuple(sorted(matched_document_ids, key=str))
        status = (
            ResolutionStatus.AMBIGUOUS
            if len(document_ids) > 1
            else ResolutionStatus.RESOLVED
        )
        return ReferenceResolution(
            reference,
            status,
            document_ids,
            chunk_ids,
            chunk_ids,
            "paper_profile.figure_table_index+chunk",
        )
    caption_candidates = [
        chunk
        for chunk in session.scalars(
            select(Chunk)
            .where(Chunk.document_id.in_(scope))
            .order_by(Chunk.document_id, Chunk.chunk_index, Chunk.id)
        )
        if any(
            _reference_matches(candidate, reference)
            for candidate in caption_references(chunk.content)
        )
    ]
    if caption_candidates:
        document_ids = tuple(
            sorted({item.document_id for item in caption_candidates}, key=str)
        )
        chunk_ids = tuple(item.id for item in caption_candidates)
        status = (
            ResolutionStatus.AMBIGUOUS
            if len(document_ids) > 1
            else ResolutionStatus.RESOLVED
        )
        return ReferenceResolution(
            reference,
            status,
            document_ids,
            chunk_ids,
            chunk_ids,
            "chunk.caption_fallback",
            {
                "missing_chunk_ids": sorted(set(missing_chunk_ids)),
                "profile_chunk_reference_mismatch": sorted(
                    set(mismatched_chunk_ids)
                ),
            },
        )
    if matched_profile_entry and (missing_chunk_ids or mismatched_chunk_ids):
        return ReferenceResolution(
            reference,
            ResolutionStatus.INDEX_INCONSISTENT,
            scope,
            resolution_source="paper_profile.figure_table_index+chunk",
            diagnostics={
                "missing_chunk_ids": sorted(set(missing_chunk_ids)),
                "profile_chunk_reference_mismatch": sorted(
                    set(mismatched_chunk_ids)
                ),
            },
        )
    return ReferenceResolution(
        reference,
        ResolutionStatus.NOT_FOUND,
        scope,
        resolution_source="chunk.caption",
    )


def _resolve_document(
    session: Session,
    reference: TypedReference,
    scope: tuple[UUID, ...],
) -> ReferenceResolution:
    key = unicodedata.normalize("NFKC", reference.normalized_key).strip().casefold()
    if not key:
        return ReferenceResolution(
            reference,
            ResolutionStatus.INVALID,
            scope,
            resolution_source="document.original_filename",
        )
    documents = [
        item
        for item in session.scalars(
            select(Document).where(Document.id.in_(scope)).order_by(Document.id)
        )
        if unicodedata.normalize("NFKC", item.original_filename).strip().casefold()
        == key
    ]
    if not documents:
        return ReferenceResolution(
            reference,
            ResolutionStatus.NOT_FOUND,
            scope,
            resolution_source="document.original_filename",
        )
    document_ids = tuple(sorted((item.id for item in documents), key=str))
    status = (
        ResolutionStatus.RESOLVED
        if len(documents) == 1
        else ResolutionStatus.AMBIGUOUS
    )
    return ReferenceResolution(
        reference,
        status,
        document_ids,
        document_ids,
        resolution_source="document.original_filename",
        diagnostics={"candidate_count": len(documents)},
    )


def resolve_structure_reference(
    session: Session,
    reference: TypedReference,
    document_ids: Collection[UUID],
) -> ReferenceResolution:
    scope = tuple(sorted(set(document_ids), key=str))
    if not scope:
        return ReferenceResolution(reference, ResolutionStatus.NOT_FOUND)
    if reference.kind in (ReferenceKind.FIGURE, ReferenceKind.TABLE):
        if (
            not reference.normalized_key.isdigit()
            or not 1 <= int(reference.normalized_key) <= 999
        ):
            return ReferenceResolution(
                reference,
                ResolutionStatus.INVALID,
                scope,
                diagnostics={"reason": "structure_number_out_of_range"},
            )
    if reference.kind == ReferenceKind.PAGE and (
        not reference.normalized_key.isdigit()
        or int(reference.normalized_key) < 1
    ):
        return ReferenceResolution(
            reference,
            ResolutionStatus.INVALID,
            scope,
            diagnostics={"reason": "page_number_out_of_range"},
        )
    if reference.kind == ReferenceKind.PAGE:
        return _resolve_page(session, reference, scope)
    if reference.kind == ReferenceKind.SECTION:
        return _resolve_section(session, reference, scope)
    if reference.kind in (ReferenceKind.FIGURE, ReferenceKind.TABLE):
        return _resolve_figure_or_table(session, reference, scope)
    if reference.kind == ReferenceKind.DOCUMENT:
        return _resolve_document(session, reference, scope)
    return ReferenceResolution(reference, ResolutionStatus.INVALID, scope)
