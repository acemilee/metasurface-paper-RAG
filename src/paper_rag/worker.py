from __future__ import annotations

import asyncio
import json
import socket
import threading
from datetime import datetime
from datetime import timedelta
from time import perf_counter
from uuid import UUID

import psutil
from sqlalchemy import delete, func, select

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.models.document import Document, DocumentStatus, DomainStatus, FormulaIndexStatus
from paper_rag.models.domain_admission import DomainAssessment
from paper_rag.models.job import IngestionJob, JobState, claim_next_job
from paper_rag.models.page import Page, TextBlock
from paper_rag.services.pdf_classifier import PdfType, classify_pdf
from paper_rag.services.job_recovery import reclaim_stale_jobs, update_worker_heartbeat
from paper_rag.services.pdf_parser import parse_pdf, write_page_jsonl
from paper_rag.services.indexing import index_document
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.domain_admission import AdmissionPage, evaluate_domain_admission
from paper_rag.services.domain_assessment import (
    apply_domain_assessment,
    clear_knowledge_artifacts_for_review,
)
from paper_rag.services.document_genre import classify_document_genre
from paper_rag.services.vector_store import get_chroma_collection
from paper_rag.services.paper_profile import build_paper_profile
from paper_rag.services.formula_governance import mark_stale_formula_indexes
from paper_rag.services.formula_service import FORMULA_PARSER_VERSION


def _process_ingestion_job_sync(job_id: UUID, worker_id: str) -> None:
    settings = get_settings()
    with SessionLocal() as session:
        job = session.get(IngestionJob, job_id)
        if job is None:
            return
        document = session.get(Document, job.document_id)
        if document is None:
            return
        try:
            classification = classify_pdf(document.stored_path)
            document.pdf_type = classification.pdf_type.value
            document.page_count = classification.page_count
            if classification.page_count > settings.max_pdf_pages:
                raise ValueError(f"PDF exceeds page limit of {settings.max_pdf_pages}")
            if classification.pdf_type is PdfType.ENCRYPTED_OR_INVALID:
                raise ValueError("PDF is encrypted or invalid")
            job.state = JobState.PARSING
            document.status = DocumentStatus.PARSING
            session.commit()
            stage_started = perf_counter()
            parsed = parse_pdf(document.stored_path, document.id, settings)
            stage_durations = {"parsing": perf_counter() - stage_started}
            write_page_jsonl(parsed, settings.parsed_dir / f"{document.id}.jsonl")
            session.execute(delete(Page).where(Page.document_id == document.id))
            session.commit()
            for parsed_page in parsed.pages:
                page = Page(document_id=document.id, page_number=parsed_page.page_number, text=parsed_page.text, extraction_method=parsed_page.extraction_method, quality_score=parsed_page.quality_score, ocr_confidence=parsed_page.ocr_confidence)
                session.add(page)
                session.flush()
                session.add_all([
                    TextBlock(page_id=page.id, reading_order=block.reading_order, text=block.text, x0=block.x0, y0=block.y0, x1=block.x1, y1=block.y1, source=block.source, confidence=block.confidence)
                    for block in parsed_page.blocks
                ])
            session.commit()

            provider = get_embedding_provider(settings)
            domain_result = evaluate_domain_admission(
                [
                    AdmissionPage(
                        page.page_number,
                        page.text,
                        page.quality_score,
                        page.ocr_confidence,
                    )
                    for page in parsed.pages
                ],
                provider,
                settings,
            )
            previous_assessment_count = session.scalar(
                select(func.count(DomainAssessment.id)).where(
                    DomainAssessment.document_id == document.id
                )
            ) or 0
            assessment_trigger = (
                "upload"
                if previous_assessment_count == 0
                and document.domain_enforcement_version is not None
                and document.domain_status != DomainStatus.MANUAL_APPROVED
                else "reindex"
            )
            admission = apply_domain_assessment(
                session,
                document,
                domain_result,
                trigger=assessment_trigger,
            )
            if not admission.may_index:
                collection = get_chroma_collection(settings, provider)
                clear_knowledge_artifacts_for_review(
                    session,
                    document,
                    collection,
                    settings,
                )
                job.state = admission.terminal_job_state or JobState.REVIEW_REQUIRED
                job.finished_at = datetime.now().astimezone()
                job.heartbeat_at = datetime.now().astimezone()
                job.stage_durations_json = json.dumps(stage_durations)
                session.commit()
                return

            genre_result = classify_document_genre(
                document.original_filename,
                [page.text for page in parsed.pages],
                provider,
            )
            if not document.genre_manually_overridden:
                if document.genre_original_prediction is None:
                    document.genre_original_prediction = genre_result.genre
                document.document_genre = genre_result.genre
                document.genre_score = genre_result.score
                document.genre_decision_source = genre_result.decision_source
                document.genre_scores_json = json.dumps(genre_result.scores, ensure_ascii=False)
                document.genre_evidence_json = json.dumps(genre_result.evidence, ensure_ascii=False)
                document.genre_conflicts_json = json.dumps(genre_result.conflicts, ensure_ascii=False)
                document.genre_classifier_version = genre_result.classifier_version
                document.genre_checked_at = datetime.now().astimezone()

            state_by_stage = {
                "chunking": JobState.CHUNKING,
                "embedding": JobState.EMBEDDING,
                "indexing": JobState.INDEXING,
            }

            def update_stage(stage: str) -> None:
                nonlocal stage_started
                previous_stage = job.state.value
                stage_durations[previous_stage] = perf_counter() - stage_started
                job.state = state_by_stage[stage]
                job.heartbeat_at = datetime.now().astimezone()
                job.stage_durations_json = json.dumps(stage_durations)
                session.commit()
                stage_started = perf_counter()

            index_document(session, document, settings, stage_callback=update_stage)
            stage_durations[job.state.value] = perf_counter() - stage_started
            job.state = JobState.COMPLETED
            job.finished_at = datetime.now().astimezone()
            document.status = DocumentStatus.COMPLETED
            job.heartbeat_at = datetime.now().astimezone()
            job.stage_durations_json = json.dumps(stage_durations)
            session.commit()
            try:
                build_paper_profile(session, document.id)
            except Exception:
                session.rollback()
        except Exception as exc:
            session.rollback()
            job = session.get(IngestionJob, job_id)
            document = session.get(Document, job.document_id) if job is not None else None
            if job is None or document is None:
                return
            job.state = JobState.FAILED
            job.error_code = type(exc).__name__
            job.error_message = str(exc)
            job.finished_at = datetime.now().astimezone()
            job.heartbeat_at = datetime.now().astimezone()
            document.status = DocumentStatus.FAILED
            document.formula_index_status = FormulaIndexStatus.FAILED
            document.formula_parser_version = FORMULA_PARSER_VERSION
            document.formula_index_updated_at = datetime.now().astimezone()
            session.commit()


async def process_ingestion_job(job_id: UUID, worker_id: str) -> None:
    settings = get_settings()
    stop_heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(settings.worker_heartbeat_seconds):
            with SessionLocal() as heartbeat_session:
                update_worker_heartbeat(
                    heartbeat_session,
                    worker_id,
                    current_job_id=job_id,
                    status="working",
                )

    previous_stack_size = threading.stack_size()
    threading.stack_size(8 * 1024 * 1024)
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop, name="worker-heartbeat", daemon=True
    )
    threading.stack_size(previous_stack_size)
    heartbeat_thread.start()
    try:
        _process_ingestion_job_sync(job_id, worker_id)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=settings.worker_heartbeat_seconds + 1)


async def run_ingestion_worker(worker_id: str | None = None) -> None:
    worker_id = worker_id or socket.gethostname()
    get_embedding_provider(get_settings())
    with SessionLocal() as session:
        mark_stale_formula_indexes(session)
    while True:
        if psutil.virtual_memory().percent >= get_settings().worker_max_memory_percent:
            with SessionLocal() as session:
                update_worker_heartbeat(session, worker_id, status="backpressure")
            await asyncio.sleep(get_settings().worker_heartbeat_seconds)
            continue
        with SessionLocal() as session:
            reclaim_stale_jobs(
                session,
                timedelta(seconds=get_settings().stale_job_seconds),
                get_settings().max_job_attempts,
            )
            update_worker_heartbeat(session, worker_id, status="idle")
            job = claim_next_job(session, worker_id)
            job_id = job.id if job is not None else None
        if job is None:
            await asyncio.sleep(1)
            continue
        await process_ingestion_job(job_id, worker_id)


def main() -> None:
    asyncio.run(run_ingestion_worker())


if __name__ == "__main__":
    main()
