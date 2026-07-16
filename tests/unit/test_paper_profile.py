from __future__ import annotations

from uuid import uuid4

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.paper_profile import PaperProfile, PaperProfileClaim
from paper_rag.schemas.query_plan import EvidenceType
from paper_rag.services.paper_profile import (
    build_paper_profile,
    get_profile_retrieval_hints,
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _document_with_chunks(session: Session) -> Document:
    document = Document(
        original_filename="paper.pdf",
        stored_path=f"data/{uuid4()}.pdf",
        file_sha256="a" * 64,
        status=DocumentStatus.COMPLETED,
        document_genre="research_paper",
        page_count=8,
    )
    session.add(document)
    session.flush()
    contents = [
        ("Abstract", "We propose a transparent graphene metasurface for tunable absorption.", 1),
        ("1 Introduction", "Existing absorbers lack simultaneous flexibility and optical transparency.", 2),
        ("2 Method", "The patterned graphene structure is modeled with an equivalent circuit.", 3),
        ("3 Results", "The measured absorption exceeds 90% from 7 to 18 GHz.", 5),
        ("4 Comparison", "Compared with metal absorbers, the device is flexible and transparent.", 6),
        ("5 Conclusion", "The experiments validate tunable broadband absorption; durability remains future work.", 8),
    ]
    for index, (section, content, page) in enumerate(contents):
        session.add(
            Chunk(
                document_id=document.id,
                vector_id=f"{document.id}:test:{index}",
                content=content,
                page_start=page,
                page_end=page,
                section_path=section,
                chunk_index=index,
            )
        )
    session.commit()
    return document


def test_profile_is_evidence_bound_and_idempotent() -> None:
    session = _session()
    document = _document_with_chunks(session)
    before_chunks = session.scalar(select(func.count(Chunk.id)))

    first = build_paper_profile(session, document.id)
    second = build_paper_profile(session, document.id)

    assert first.id == second.id
    assert first.status == "ready"
    assert first.profile_version == 1
    assert session.scalar(select(func.count(PaperProfile.id))) == 1
    assert session.scalar(select(func.count(Chunk.id))) == before_chunks
    claims = list(session.scalars(select(PaperProfileClaim)))
    valid_chunk_ids = set(session.scalars(select(Chunk.id)))
    assert claims
    assert all(claim.audit_verdict == "exact_extract" for claim in claims)
    assert all(set(claim.citation_ids).issubset(valid_chunk_ids) for claim in claims)
    content = __import__("json").loads(first.content_json)
    assert any("18 GHz" in item["value_text"] for item in content["fact_ledger"])
    assert all(item["chunk_id"] for item in content["fact_ledger"])
    assert "mechanism_statements" in content
    assert "figure_table_index" in content


def test_changed_source_creates_new_version_after_success() -> None:
    session = _session()
    document = _document_with_chunks(session)
    first = build_paper_profile(session, document.id)
    chunk = session.scalar(select(Chunk).where(Chunk.document_id == document.id).limit(1))
    chunk.content += " Updated evidence."
    session.commit()

    second = build_paper_profile(session, document.id)
    session.refresh(first)

    assert second.id != first.id
    assert second.profile_version == 2
    assert second.status == "ready"
    assert first.status == "stale"


def test_profile_hints_select_required_roles_without_becoming_evidence() -> None:
    session = _session()
    document = _document_with_chunks(session)
    build_paper_profile(session, document.id)

    hints = get_profile_retrieval_hints(
        session,
        [document.id],
        [EvidenceType.PROBLEM_OR_GAP, EvidenceType.METHOD_OR_STRUCTURE, EvidenceType.RESULT_OR_ADVANTAGE],
    )

    assert {hint[1] for hint in hints} == {
        EvidenceType.PROBLEM_OR_GAP.value,
        EvidenceType.METHOD_OR_STRUCTURE.value,
        EvidenceType.RESULT_OR_ADVANTAGE.value,
    }
    assert all(len(hint) == 2 for hint in hints)
