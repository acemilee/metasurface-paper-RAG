from pathlib import Path
from uuid import uuid4
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.config import Settings
from paper_rag.db import Base
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page, TextBlock
from paper_rag.services.chunking import ChunkDraft
from paper_rag.services.formula_backfill import ChunkChange, apply_formula_backfill, plan_formula_backfill
from paper_rag.services.pdf_parser import parse_text_pdf
from scripts.backfill_formulas import parse_pages


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"
requires_sample_pdf = pytest.mark.skipif(
    not SAMPLE_PDF.is_file(),
    reason="private regression PDF is not distributed",
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _build_backfill_fixture() -> tuple[Session, Document, Chunk]:
    session = _session()
    document_id = uuid4()
    parsed_page = parse_text_pdf(SAMPLE_PDF, document_id).pages[3]
    document = Document(
        id=document_id,
        original_filename=SAMPLE_PDF.name,
        stored_path=str(SAMPLE_PDF),
        file_sha256="d" * 64,
        page_count=9,
        status=DocumentStatus.COMPLETED,
    )
    page = Page(document_id=document_id, page_number=4, text=parsed_page.text)
    session.add_all([document, page])
    session.flush()
    session.add_all(
        TextBlock(
            page_id=page.id,
            reading_order=block.reading_order,
            text=block.text,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            source=block.source,
            confidence=block.confidence,
        )
        for block in parsed_page.blocks
    )
    legacy_id = uuid4()
    session.add(
        Formula(
            id=legacy_id,
            document_id=document_id,
            page_number=4,
            placeholder=f"公式_placeholder_{legacy_id}",
            bbox_json="[10, 10, 20, 20]",
            raw_text=": (2)",
            parser_version="legacy-v1",
        )
    )
    chunk = Chunk(
        document_id=document_id,
        vector_id=f"{document_id}:phase2-v1:0",
        content=parsed_page.text,
        page_start=4,
        page_end=4,
        chunk_index=0,
    )
    session.add(chunk)
    session.commit()
    return session, document, chunk


def _counts(session: Session) -> tuple[int, int]:
    return (
        session.scalar(select(func.count(Formula.id))) or 0,
        session.scalar(select(func.count(Chunk.id))) or 0,
    )


@requires_sample_pdf
def test_backfill_plan_is_read_only_and_reports_only_changed_chunks() -> None:
    session, document, chunk = _build_backfill_fixture()
    before = _counts(session)
    settings = Settings(chunk_target_chars=1400, chunk_overlap_chars=180)

    plan = plan_formula_backfill(session, document.id, [4], settings)

    assert plan.document_id == document.id
    assert plan.page_numbers == (4,)
    assert {item.formula_number for item in plan.new_formulas} >= {"1a", "1b", "1c", "2"}
    assert plan.changed_chunks
    assert _counts(session) == before
    assert session.get(Chunk, chunk.id).content == chunk.content


class FakeProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self, initial_ids: list[str]) -> None:
        self.ids = set(initial_ids)
        self.upserts: list[list[str]] = []
        self.deletes: list[list[str]] = []

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        values = list(ids)
        self.ids.update(values)
        self.upserts.append(values)

    def delete(self, *, ids) -> None:
        values = list(ids)
        self.ids.difference_update(values)
        self.deletes.append(values)

    def get(self, *, ids=None, where=None, include=None) -> dict:
        if ids is not None:
            return {"ids": [value for value in ids if value in self.ids]}
        return {"ids": sorted(self.ids)}


class FailingCollection(FakeCollection):
    def __init__(self, initial_ids: list[str]) -> None:
        super().__init__(initial_ids)
        self.fail_once = True

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        super().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("injected chroma failure")


class DropLastVectorOnceCollection(FakeCollection):
    def __init__(self, initial_ids: list[str]) -> None:
        super().__init__(initial_ids)
        self.drop_once = True

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        ids = list(ids)
        embeddings = list(embeddings)
        documents = list(documents)
        metadatas = list(metadatas)
        if self.drop_once:
            self.drop_once = False
            ids = ids[:-1]
            embeddings = embeddings[:-1]
            documents = documents[:-1]
            metadatas = metadatas[:-1]
        super().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


@requires_sample_pdf
def test_backfill_apply_updates_only_changed_chunks_and_is_idempotent() -> None:
    session, document, chunk = _build_backfill_fixture()
    settings = Settings(chunk_target_chars=1400, chunk_overlap_chars=180)
    collection = FakeCollection([chunk.vector_id])
    plan = plan_formula_backfill(session, document.id, [4], settings)
    changed_ids = {item.vector_id for item in plan.changed_chunks if item.action != "delete"}

    report = apply_formula_backfill(
        session,
        plan,
        settings,
        FakeProvider(),
        collection,
        finalize=False,
    )

    assert report["status"] == "applied"
    assert set(collection.upserts[0]) == changed_ids
    assert {item.formula_number for item in session.scalars(select(Formula))} >= {"1a", "1b", "1c", "2"}
    second = plan_formula_backfill(session, document.id, [4], settings)
    assert second.changed_chunks == ()

    second_report = apply_formula_backfill(
        session,
        second,
        settings,
        FakeProvider(),
        collection,
        finalize=False,
    )
    assert second_report["changed_vector_ids"] == []
    assert len(collection.upserts) == 1


@requires_sample_pdf
def test_backfill_compensates_vectors_and_rolls_back_database_on_partial_failure() -> None:
    session, document, chunk = _build_backfill_fixture()
    original_content = chunk.content
    settings = Settings(chunk_target_chars=1400, chunk_overlap_chars=180)
    collection = FailingCollection([chunk.vector_id])
    plan = plan_formula_backfill(session, document.id, [4], settings)

    with pytest.raises(RuntimeError, match="injected chroma failure"):
        apply_formula_backfill(
            session,
            plan,
            settings,
            FakeProvider(),
            collection,
            finalize=False,
        )

    assert session.get(Chunk, chunk.id).content == original_content
    assert len(collection.upserts) == 2
    assert collection.upserts[-1] == [chunk.vector_id]


def test_parse_pages_accepts_ranges_and_rejects_invalid_values() -> None:
    assert parse_pages("4,6-7") == [4, 6, 7]
    with pytest.raises(ValueError):
        parse_pages("0")
    with pytest.raises(ValueError):
        parse_pages("7-6")
    with pytest.raises(ValueError):
        parse_pages("four")


@requires_sample_pdf
def test_backfill_retries_a_vector_missing_after_the_first_batch_upsert() -> None:
    session, document, chunk = _build_backfill_fixture()
    settings = Settings(chunk_target_chars=1400, chunk_overlap_chars=180)
    base_plan = plan_formula_backfill(session, document.id, [4], settings)
    created_id = f"{document.id}:phase2-v1:1"
    extra_draft = ChunkDraft(
        document.id,
        1,
        "extra deterministic chunk",
        4,
        4,
        None,
        "paragraph",
        [],
    )
    extra_change = ChunkChange(
        chunk_index=1,
        vector_id=created_id,
        old_content_sha256=None,
        new_content_sha256="new",
        old_formula_ids=(),
        new_formula_ids=(),
        action="create",
    )
    plan = replace(
        base_plan,
        chunk_drafts=(*base_plan.chunk_drafts, extra_draft),
        changed_chunks=(*base_plan.changed_chunks, extra_change),
    )
    collection = DropLastVectorOnceCollection([chunk.vector_id])

    report = apply_formula_backfill(
        session,
        plan,
        settings,
        FakeProvider(),
        collection,
        finalize=False,
    )

    assert report["status"] == "applied"
    assert created_id in collection.ids
    assert session.scalar(select(func.count(Chunk.id))) == 2
    assert len(collection.upserts) == 2
