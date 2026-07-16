from __future__ import annotations

import json
import re
from collections.abc import Collection
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.services.formula_dependencies import missing_formula_group_parts
from paper_rag.services.references.types import (
    ReferenceResolution,
    ResolutionStatus,
    TypedReference,
)


def _chunk_ids_for_formulas(
    session: Session,
    document_ids: Collection[UUID],
    formula_ids: set[UUID],
) -> tuple[UUID, ...]:
    scope = tuple(sorted(set(document_ids), key=str))
    if not scope or not formula_ids:
        return ()
    matched: list[UUID] = []
    chunks = session.scalars(select(Chunk).where(Chunk.document_id.in_(scope)))
    for chunk in chunks:
        try:
            linked = {
                UUID(str(item))
                for item in json.loads(chunk.formula_ids_json or "[]")
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if linked & formula_ids:
            matched.append(chunk.id)
    return tuple(sorted(set(matched), key=str))


def _complete_group_parts(formulas: list[Formula], base: str) -> list[Formula]:
    candidates = [
        item
        for item in formulas
        if item.formula_number
        and re.fullmatch(
            rf"{re.escape(base)}[a-z]",
            item.formula_number,
            re.IGNORECASE,
        )
        and item.group_key
    ]
    group_keys = {item.group_key for item in candidates}
    if len(group_keys) != 1 or missing_formula_group_parts(candidates):
        return []
    return sorted(
        candidates,
        key=lambda item: (item.part_index, item.formula_number or "", str(item.id)),
    )


def resolve_formula_reference(
    session: Session,
    reference: TypedReference,
    document_ids: Collection[UUID],
) -> ReferenceResolution:
    scope = tuple(sorted(set(document_ids), key=str))
    if not scope:
        return ReferenceResolution(reference, ResolutionStatus.NOT_FOUND)
    key_match = re.fullmatch(r"(\d{1,3})([a-z]?)", reference.normalized_key, re.I)
    if key_match is None or int(key_match.group(1)) < 1:
        return ReferenceResolution(
            reference,
            ResolutionStatus.INVALID,
            scope,
            resolution_source="formula.formula_number",
            diagnostics={"reason": "formula_number_out_of_range"},
        )
    stale_document_ids = tuple(
        sorted(
            session.scalars(
                select(Document.id).where(
                    Document.id.in_(scope),
                    Document.formula_index_status.in_(("stale", "failed")),
                )
            ),
            key=str,
        )
    )
    if stale_document_ids:
        stale_formulas = list(
            session.scalars(
                select(Formula).where(
                    Formula.document_id.in_(stale_document_ids),
                    Formula.formula_number == reference.normalized_key,
                )
            )
        )
        return ReferenceResolution(
            reference,
            ResolutionStatus.STALE,
            stale_document_ids,
            tuple(sorted((item.id for item in stale_formulas), key=str)),
            resolution_source="document.formula_index_status",
            diagnostics={
                "formula_index_status": "stale_or_failed",
                "page_numbers": sorted(
                    {item.page_number for item in stale_formulas}
                ),
            },
        )
    formulas = list(
        session.scalars(
            select(Formula).where(
                Formula.document_id.in_(scope),
                Formula.formula_number == reference.normalized_key,
            )
        )
    )
    resolved_as_group = False
    if not formulas and re.fullmatch(r"\d{1,3}", reference.normalized_key):
        scoped_formulas = list(
            session.scalars(select(Formula).where(Formula.document_id.in_(scope)))
        )
        group_candidates = [
            item
            for item in scoped_formulas
            if item.formula_number
            and re.fullmatch(
                rf"{re.escape(reference.normalized_key)}[a-z]",
                item.formula_number,
                re.IGNORECASE,
            )
            and item.group_key
        ]
        candidate_documents = {item.document_id for item in group_candidates}
        candidate_groups = {(item.document_id, item.group_key) for item in group_candidates}
        if len(candidate_documents) > 1 or len(candidate_groups) > 1:
            return ReferenceResolution(
                reference,
                ResolutionStatus.AMBIGUOUS,
                tuple(sorted(candidate_documents, key=str)),
                tuple(sorted((item.id for item in group_candidates), key=str)),
                resolution_source="formula.formula_number_group",
                diagnostics={"candidate_count": len(group_candidates)},
            )
        missing_parts = missing_formula_group_parts(group_candidates)
        if group_candidates and missing_parts:
            return ReferenceResolution(
                reference,
                ResolutionStatus.INDEX_INCONSISTENT,
                tuple(sorted(candidate_documents, key=str)),
                tuple(sorted((item.id for item in group_candidates), key=str)),
                resolution_source="formula.formula_number_group",
                diagnostics={"missing_parts": missing_parts},
            )
        formulas = _complete_group_parts(scoped_formulas, reference.normalized_key)
        resolved_as_group = bool(formulas)
    if not formulas:
        return ReferenceResolution(reference, ResolutionStatus.NOT_FOUND, scope)
    if resolved_as_group:
        formula_ids = {item.id for item in formulas}
        chunks = _chunk_ids_for_formulas(session, scope, formula_ids)
        status = (
            ResolutionStatus.RESOLVED
            if chunks
            else ResolutionStatus.INDEX_INCONSISTENT
        )
        return ReferenceResolution(
            reference,
            status,
            tuple(sorted({item.document_id for item in formulas}, key=str)),
            tuple(item.id for item in formulas),
            chunks,
            "formula.formula_number_group",
            {
                "page_numbers": sorted({item.page_number for item in formulas}),
                "group_key": formulas[0].group_key,
            },
        )
    if len(formulas) > 1:
        return ReferenceResolution(
            reference,
            ResolutionStatus.AMBIGUOUS,
            tuple(sorted({item.document_id for item in formulas}, key=str)),
            tuple(sorted((item.id for item in formulas), key=str)),
            resolution_source="formula.formula_number",
            diagnostics={"candidate_count": len(formulas)},
        )
    formula = formulas[0]
    chunks = _chunk_ids_for_formulas(session, scope, {formula.id})
    status = (
        ResolutionStatus.RESOLVED
        if chunks
        else ResolutionStatus.INDEX_INCONSISTENT
    )
    return ReferenceResolution(
        reference,
        status,
        (formula.document_id,),
        (formula.id,),
        chunks,
        "formula.formula_number",
        {"page_numbers": [formula.page_number]},
    )
