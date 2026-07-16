from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from paper_rag.config import Settings
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page
from paper_rag.services.chunking import ChunkDraft, build_chunks
from paper_rag.services.embeddings import EmbeddingProvider
from paper_rag.services.formula_service import build_formula_records_for_pages
from paper_rag.services.pdf_parser import ParsedDocument, ParsedPage, ParsedTextBlock
from paper_rag.services.vector_store import (
    finalize_chroma_persistence,
    upsert_chunks,
    verify_index_counts,
)


@dataclass(frozen=True)
class ChunkChange:
    chunk_index: int
    vector_id: str
    old_content_sha256: str | None
    new_content_sha256: str | None
    old_formula_ids: tuple[str, ...]
    new_formula_ids: tuple[str, ...]
    action: Literal["create", "update", "delete"]


@dataclass(frozen=True)
class FormulaBackfillPlan:
    document_id: UUID
    page_numbers: tuple[int, ...]
    old_formulas: tuple[Formula, ...]
    new_formulas: tuple[Formula, ...]
    chunk_drafts: tuple[ChunkDraft, ...]
    changed_chunks: tuple[ChunkChange, ...]
    source_state_sha256: str


class FormulaBackfillConflict(RuntimeError):
    pass


class VectorRecoveryRequired(RuntimeError):
    def __init__(self, vector_ids: list[str]) -> None:
        super().__init__(f"Vector recovery required for {len(vector_ids)} vector IDs")
        self.vector_ids = vector_ids


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _formula_ids(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return ()
    return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()


def _to_parsed_page(page: Page) -> ParsedPage:
    return ParsedPage(
        page.page_number,
        page.text,
        [
            ParsedTextBlock(
                page.page_number,
                block.reading_order,
                block.text,
                block.x0,
                block.y0,
                block.x1,
                block.y1,
                block.source,
                block.confidence,
            )
            for block in sorted(page.blocks, key=lambda item: item.reading_order)
        ],
        page.extraction_method,
        page.quality_score,
        page.ocr_confidence,
    )


def _load_parsed_document(session: Session, document_id: UUID) -> ParsedDocument:
    pages = list(
        session.scalars(
            select(Page)
            .options(selectinload(Page.blocks))
            .where(Page.document_id == document_id)
            .order_by(Page.page_number)
        )
    )
    if not pages:
        raise ValueError("Cannot backfill formulas without parsed pages")
    return ParsedDocument(document_id, len(pages), [_to_parsed_page(page) for page in pages])


def _state_hash(formulas: list[Formula], chunks: list[Chunk]) -> str:
    payload = {
        "formulas": [
            [str(item.id), item.page_number, item.bbox_json, item.raw_text, item.parser_version]
            for item in sorted(formulas, key=lambda value: (value.page_number, str(value.id)))
        ],
        "chunks": [
            [item.chunk_index, item.vector_id, _content_hash(item.content), item.formula_ids_json]
            for item in sorted(chunks, key=lambda value: value.chunk_index)
        ],
    }
    return _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _compare_chunks(existing: list[Chunk], drafts: list[ChunkDraft]) -> tuple[ChunkChange, ...]:
    old_by_index = {item.chunk_index: item for item in existing}
    new_by_index = {item.chunk_index: item for item in drafts}
    changes: list[ChunkChange] = []
    for index in sorted(set(old_by_index) | set(new_by_index)):
        old = old_by_index.get(index)
        new = new_by_index.get(index)
        old_ids = _formula_ids(old.formula_ids_json) if old else ()
        new_ids = tuple(str(item) for item in new.formula_ids) if new else ()
        if old is None and new is not None:
            action: Literal["create", "update", "delete"] = "create"
        elif old is not None and new is None:
            action = "delete"
        else:
            assert old is not None and new is not None
            unchanged = (
                old.content == new.content
                and old_ids == new_ids
                and old.page_start == new.page_start
                and old.page_end == new.page_end
                and old.section_path == new.section_path
            )
            if unchanged:
                continue
            action = "update"
        changes.append(
            ChunkChange(
                chunk_index=index,
                vector_id=(old.vector_id if old else f"{new.document_id}:phase2-v1:{new.chunk_index}"),
                old_content_sha256=_content_hash(old.content) if old else None,
                new_content_sha256=_content_hash(new.content) if new else None,
                old_formula_ids=old_ids,
                new_formula_ids=new_ids,
                action=action,
            )
        )
    return tuple(changes)


def plan_formula_backfill(
    session: Session,
    document_id: UUID,
    page_numbers: list[int],
    settings: Settings,
) -> FormulaBackfillPlan:
    requested_pages = tuple(sorted(set(page_numbers)))
    if not requested_pages or any(number < 1 for number in requested_pages):
        raise ValueError("At least one positive page number is required")
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    parsed = _load_parsed_document(session, document_id)
    existing_formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.document_id == document_id)
            .order_by(Formula.page_number, Formula.group_key, Formula.part_index)
        )
    )
    existing_chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
    )
    replacements = build_formula_records_for_pages(session, document_id, list(requested_pages))
    replacement_pages = set(requested_pages)
    combined = [item for item in existing_formulas if item.page_number not in replacement_pages]
    combined.extend(replacements)
    drafts = build_chunks(
        parsed,
        combined,
        settings.chunk_target_chars,
        settings.chunk_overlap_chars,
        settings.ocr_numeric_min_confidence,
    )
    return FormulaBackfillPlan(
        document_id=document_id,
        page_numbers=requested_pages,
        old_formulas=tuple(item for item in existing_formulas if item.page_number in replacement_pages),
        new_formulas=tuple(replacements),
        chunk_drafts=tuple(drafts),
        changed_chunks=_compare_chunks(existing_chunks, drafts),
        source_state_sha256=_state_hash(existing_formulas, existing_chunks),
    )


def _formula_signature(items: tuple[Formula, ...] | list[Formula]) -> tuple[tuple, ...]:
    return tuple(
        (
            str(item.id),
            item.page_number,
            item.formula_number,
            item.group_key,
            item.part_index,
            item.bbox_json,
            item.raw_text,
            item.normalized_text,
            item.semantic_status,
            item.fidelity_status,
            item.parser_version,
        )
        for item in sorted(items, key=lambda value: (value.page_number, value.part_index, str(value.id)))
    )


def formula_records_changed(
    existing: tuple[Formula, ...] | list[Formula],
    replacements: tuple[Formula, ...] | list[Formula],
) -> bool:
    return _formula_signature(existing) != _formula_signature(replacements)


def _chunk_from_draft(draft: ChunkDraft) -> Chunk:
    return Chunk(
        document_id=draft.document_id,
        vector_id=f"{draft.document_id}:phase2-v1:{draft.chunk_index}",
        content=draft.content,
        page_start=draft.page_start,
        page_end=draft.page_end,
        section_path=draft.section_path,
        content_type=draft.content_type,
        formula_ids_json=json.dumps([str(item) for item in draft.formula_ids]),
        chunk_index=draft.chunk_index,
        quality_score=draft.quality_score,
        has_low_confidence_ocr=draft.has_low_confidence_ocr,
    )


def apply_formula_backfill(
    session: Session,
    plan: FormulaBackfillPlan,
    settings: Settings,
    provider: EmbeddingProvider,
    collection,
    *,
    finalize: bool = True,
) -> dict:
    current_formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.document_id == plan.document_id)
            .order_by(Formula.page_number, Formula.group_key, Formula.part_index)
        )
    )
    current_chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id == plan.document_id)
            .order_by(Chunk.chunk_index)
        )
    )
    if _state_hash(current_formulas, current_chunks) != plan.source_state_sha256:
        raise FormulaBackfillConflict("Document formulas or chunks changed after dry-run")

    current_target_formulas = tuple(
        item for item in current_formulas if item.page_number in set(plan.page_numbers)
    )
    replace_formulas = formula_records_changed(current_target_formulas, plan.new_formulas)
    if not replace_formulas and not plan.changed_chunks:
        return {
            "status": "applied",
            "document_id": str(plan.document_id),
            "pages": list(plan.page_numbers),
            "changed_vector_ids": [],
            "changed_chunks": [],
        }

    drafts_by_index = {item.chunk_index: item for item in plan.chunk_drafts}
    chunks_by_index = {item.chunk_index: item for item in current_chunks}
    upsert_actions = [item for item in plan.changed_chunks if item.action != "delete"]
    upsert_vectors = provider.embed_documents(
        [drafts_by_index[item.chunk_index].content for item in upsert_actions]
    ) if upsert_actions else []
    retried_vector_ids: list[str] = []

    try:
        if replace_formulas:
            session.execute(
                delete(Formula).where(
                    Formula.document_id == plan.document_id,
                    Formula.page_number.in_(plan.page_numbers),
                )
            )
            session.add_all(plan.new_formulas)

        upsert_chunks_list: list[Chunk] = []
        deleted_vector_ids: list[str] = []
        for change in plan.changed_chunks:
            existing = chunks_by_index.get(change.chunk_index)
            draft = drafts_by_index.get(change.chunk_index)
            if change.action == "delete":
                if existing is not None:
                    deleted_vector_ids.append(existing.vector_id)
                    session.delete(existing)
                continue
            assert draft is not None
            if existing is None:
                existing = _chunk_from_draft(draft)
                session.add(existing)
            else:
                existing.content = draft.content
                existing.page_start = draft.page_start
                existing.page_end = draft.page_end
                existing.section_path = draft.section_path
                existing.content_type = draft.content_type
                existing.formula_ids_json = json.dumps([str(item) for item in draft.formula_ids])
                existing.quality_score = draft.quality_score
                existing.has_low_confidence_ocr = draft.has_low_confidence_ocr
            upsert_chunks_list.append(existing)

        session.flush()
        if upsert_chunks_list:
            upsert_chunks(collection, upsert_chunks_list, upsert_vectors)
            requested_ids = [item.vector_id for item in upsert_chunks_list]
            present_ids = set(collection.get(ids=requested_ids, include=[])["ids"])
            retried_vector_ids = [item for item in requested_ids if item not in present_ids]
            if retried_vector_ids:
                vector_by_id = dict(zip(requested_ids, upsert_vectors, strict=True))
                chunk_by_id = {item.vector_id: item for item in upsert_chunks_list}
                upsert_chunks(
                    collection,
                    [chunk_by_id[item] for item in retried_vector_ids],
                    [vector_by_id[item] for item in retried_vector_ids],
                )
        if deleted_vector_ids:
            collection.delete(ids=deleted_vector_ids)
        counts = verify_index_counts(session, collection, plan.document_id)
        if counts[0] != counts[1]:
            raise RuntimeError(
                f"Formula backfill index mismatch: postgres={counts[0]}, chroma={counts[1]}"
            )
        if finalize:
            finalize_chroma_persistence(settings, collection)
        session.commit()
    except Exception:
        session.rollback()
        recovery_ids = {
            item.vector_id
            for item in plan.changed_chunks
            if item.action in {"update", "delete"}
        }
        created_ids = [
            item.vector_id for item in plan.changed_chunks if item.action == "create"
        ]
        try:
            recovery_chunks = list(
                session.scalars(
                    select(Chunk).where(
                        Chunk.document_id == plan.document_id,
                        Chunk.vector_id.in_(recovery_ids),
                    )
                )
            ) if recovery_ids else []
            if recovery_chunks:
                recovery_vectors = provider.embed_documents(
                    [item.content for item in recovery_chunks]
                )
                upsert_chunks(collection, recovery_chunks, recovery_vectors)
            if created_ids:
                collection.delete(ids=created_ids)
            if finalize:
                finalize_chroma_persistence(settings, collection)
        except Exception as recovery_exc:
            raise VectorRecoveryRequired(sorted(recovery_ids | set(created_ids))) from recovery_exc
        raise

    return {
        "status": "applied",
        "document_id": str(plan.document_id),
        "pages": list(plan.page_numbers),
        "changed_vector_ids": [item.vector_id for item in plan.changed_chunks],
        "retried_vector_ids": retried_vector_ids,
        "changed_chunks": [
            {"chunk_index": item.chunk_index, "action": item.action, "vector_id": item.vector_id}
            for item in plan.changed_chunks
        ],
    }
