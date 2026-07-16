from __future__ import annotations

import json
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, FormulaIndexStatus
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.schemas.query_plan import (
    AnswerMode,
    EntityType,
    EvidenceType,
    QueryEntity,
    QueryPlan,
    RetrievalQuery,
)
from paper_rag.services.query_intent import QueryIntent
from paper_rag.services.retrieval import RetrievedChunk


def make_formula(
    document: Document,
    *,
    number: str,
    placeholder: str | None = None,
    page_number: int = 1,
    group_key: str | None = None,
    part_index: int = 0,
) -> Formula:
    formula_id = uuid4()
    return Formula(
        id=formula_id,
        document_id=document.id,
        page_number=page_number,
        placeholder=placeholder or f"FORMULA_{formula_id.hex}",
        bbox_json="[0, 0, 100, 40]",
        raw_text=f"equation ({number})",
        formula_number=number,
        group_key=group_key,
        part_index=part_index,
        normalized_text=f"equation {number}",
    )


def make_chunk(
    document: Document,
    *,
    content: str = "reference evidence",
    page_start: int = 1,
    page_end: int | None = None,
    section_path: str | None = None,
    formula_ids: list[UUID] | tuple[UUID, ...] = (),
    chunk_index: int = 0,
) -> Chunk:
    chunk_id = uuid4()
    return Chunk(
        id=chunk_id,
        document_id=document.id,
        vector_id=f"reference:{chunk_id}",
        content=content,
        page_start=page_start,
        page_end=page_start if page_end is None else page_end,
        section_path=section_path,
        formula_ids_json=json.dumps([str(item) for item in formula_ids]),
        chunk_index=chunk_index,
    )


def make_page(document: Document, *, page_number: int, text: str = "page evidence") -> Page:
    return Page(id=uuid4(), document_id=document.id, page_number=page_number, text=text)


def make_ready_profile(
    document: Document,
    *,
    figure_table_index: list[dict[str, object]],
) -> PaperProfile:
    return PaperProfile(
        id=uuid4(),
        document_id=document.id,
        status="ready",
        profile_version=1,
        parser_version="test-parser",
        prompt_version="test-prompt",
        source_sha256=document.file_sha256,
        content_json=json.dumps({"figure_table_index": figure_table_index}),
    )


def make_retrieved_chunk(
    document: Document,
    *,
    content: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(uuid4(), document.id, content, 1, 1, None, [], score)


def seed_formula_with_chunk(
    session: Session,
    document: Document,
    *,
    number: str,
    page_number: int = 1,
) -> tuple[Formula, Chunk]:
    formula = make_formula(document, number=number, page_number=page_number)
    chunk = make_chunk(
        document,
        content=f"{formula.placeholder} ({number})",
        page_start=page_number,
        formula_ids=[formula.id],
    )
    session.add_all([formula, chunk])
    session.commit()
    return formula, chunk


def seed_formula_and_page(
    session: Session,
    document: Document,
    *,
    number: str,
    page_number: int,
) -> tuple[Formula, Page]:
    formula, _ = seed_formula_with_chunk(
        session, document, number=number, page_number=page_number
    )
    page = make_page(document, page_number=page_number)
    session.add(page)
    session.commit()
    return formula, page


def seed_stale_formula(
    session: Session,
    document: Document,
    *,
    number: str,
    page_number: int,
) -> Formula:
    formula, _ = seed_formula_with_chunk(
        session, document, number=number, page_number=page_number
    )
    document.formula_index_status = FormulaIndexStatus.STALE
    session.commit()
    return formula


def seed_numbered_formulas(
    session: Session,
    documents: list[Document],
    *,
    numbers: tuple[str, ...],
) -> list[Formula]:
    seeded = [make_formula(documents[0], number=number) for number in numbers]
    session.add_all(seeded)
    session.flush()
    session.add_all(
        make_chunk(
            documents[0],
            formula_ids=[item.id],
            content=item.placeholder,
            chunk_index=index,
        )
        for index, item in enumerate(seeded)
    )
    session.commit()
    return seeded


def formula_extract_plan(*, entity_surface: str) -> QueryPlan:
    return QueryPlan(
        intent=QueryIntent.FORMULA,
        answer_mode=AnswerMode.EXTRACT,
        standalone_question=entity_surface,
        retrieval_queries=[
            RetrievalQuery(
                query=entity_surface,
                evidence_type=EvidenceType.FORMULA_CONTEXT,
            )
        ],
        entities=[
            QueryEntity(
                surface=entity_surface,
                entity_type=EntityType.FORMULA,
                must_link=True,
            )
        ],
        required_evidence=[EvidenceType.FORMULA_CONTEXT],
        confidence=1.0,
    )


def _formula_reference(number: str = "5"):
    from paper_rag.services.references.types import (
        ReferenceKind,
        ReferenceSource,
        TypedReference,
    )

    return TypedReference(
        ReferenceKind.FORMULA,
        f"公式{number}",
        number,
        None,
        ReferenceSource.ORIGINAL_QUESTION,
    )


def resolution_with_status(status):
    from paper_rag.services.references.types import ReferenceResolution

    return ReferenceResolution(reference=_formula_reference(), status=status)


def resolved_formula_resolution(
    document_id: UUID,
    chunk_id: UUID,
    *,
    target_id: UUID | None = None,
):
    from paper_rag.services.references.types import ReferenceResolution, ResolutionStatus

    return ReferenceResolution(
        reference=_formula_reference(),
        status=ResolutionStatus.RESOLVED,
        document_ids=(document_id,),
        target_ids=() if target_id is None else (target_id,),
        evidence_chunk_ids=(chunk_id,),
        resolution_source="formula.formula_number",
    )


def stale_resolution_for(formula: Formula):
    from paper_rag.services.references.types import ReferenceResolution, ResolutionStatus

    return ReferenceResolution(
        reference=_formula_reference(formula.formula_number or ""),
        status=ResolutionStatus.STALE,
        document_ids=(formula.document_id,),
        target_ids=(formula.id,),
        diagnostics={"page_numbers": [formula.page_number]},
    )


def sample_resolved_formula():
    return resolved_formula_resolution(uuid4(), uuid4(), target_id=uuid4())
