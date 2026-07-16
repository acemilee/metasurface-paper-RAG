from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from paper_rag.config import Settings
from paper_rag.models.document import Document, FormulaIndexStatus
from paper_rag.models.chunk import Chunk
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page
from paper_rag.services.chunking import build_chunks
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.formula_service import create_formula_records
from paper_rag.services.formula_governance import assert_current_formula_records, derive_formula_index_status
from paper_rag.services.formula_dependencies import rebuild_formula_dependency_graph
from paper_rag.services.formula_assets import refresh_formula_source_crop_hashes
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION
from datetime import datetime
from paper_rag.services.pdf_parser import ParsedDocument, ParsedPage, ParsedTextBlock
from paper_rag.services.vector_store import delete_document_vectors, get_chroma_collection, replace_document_chunks, upsert_chunks, verify_index_counts


def _load_parsed_document(session: Session, document: Document) -> ParsedDocument:
    pages = []
    for page in session.scalars(select(Page).where(Page.document_id == document.id).order_by(Page.page_number)):
        blocks = [ParsedTextBlock(page.page_number, block.reading_order, block.text, block.x0, block.y0, block.x1, block.y1, block.source, block.confidence) for block in sorted(page.blocks, key=lambda item: item.reading_order)]
        pages.append(ParsedPage(page.page_number, page.text, blocks, page.extraction_method, page.quality_score, page.ocr_confidence))
    return ParsedDocument(document.id, len(pages), pages)


def index_document(
    session: Session,
    document: Document,
    settings: Settings,
    stage_callback: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    document.formula_index_status = FormulaIndexStatus.BUILDING
    document.formula_parser_version = FORMULA_PARSER_VERSION
    document.formula_index_updated_at = datetime.now().astimezone()
    session.commit()
    parsed = _load_parsed_document(session, document)
    if not parsed.pages:
        raise ValueError("Cannot index a document without parsed pages")
    if stage_callback:
        stage_callback("chunking")
    new_formulas = [
        formula
        for page in parsed.pages
        for formula in create_formula_records(document.id, page)
    ]
    assert_current_formula_records(new_formulas)
    session.execute(delete(Formula).where(Formula.document_id == document.id))
    session.add_all(new_formulas)
    session.commit()
    rebuild_formula_dependency_graph(session, document.id)
    refresh_formula_source_crop_hashes(session, document.id)
    formulas = list(session.scalars(select(Formula).where(Formula.document_id == document.id)))
    provider = get_embedding_provider(settings)
    collection = get_chroma_collection(settings, provider)
    existing_chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    delete_document_vectors(collection, existing_chunks, settings)
    drafts = build_chunks(parsed, formulas, settings.chunk_target_chars, settings.chunk_overlap_chars, settings.ocr_numeric_min_confidence)
    chunks = replace_document_chunks(session, document.id, drafts, "phase2-v1")
    if stage_callback:
        stage_callback("embedding")
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    if stage_callback:
        stage_callback("indexing")
    upsert_chunks(collection, chunks, vectors, settings)
    counts = verify_index_counts(session, collection, document.id)
    if counts[0] == 0 or counts[0] != counts[1]:
        raise RuntimeError(
            f"Index consistency check failed: postgres={counts[0]}, chroma={counts[1]}"
        )
    derive_formula_index_status(session, document.id)
    return counts
