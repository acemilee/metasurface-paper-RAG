import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from paper_rag.config import Settings, get_settings
from paper_rag.db import get_db_session
from paper_rag.models.audit import AnswerAudit
from paper_rag.models.document import Document, DomainStatus
from paper_rag.models.domain_admission import DomainAssessment
from paper_rag.models.page import Page
from paper_rag.models.job import IngestionJob, JobState
from paper_rag.models.chunk import Chunk
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.schemas.documents import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DeletionCheckResponse,
    DocumentListItem,
    DocumentListResponse,
    DomainApproveRequest,
    DomainReviewResponse,
    UploadAccepted,
)
from paper_rag.services.deletion_confirmation import deletion_confirmations
from paper_rag.services.domain_admission import AdmissionPage, evaluate_domain_admission
from paper_rag.services.domain_assessment import (
    DomainAssessmentConflict,
    approve_domain_assessment,
    record_domain_assessment,
)
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.filename_search import normalize_filename_search_key
from paper_rag.services.ingestion import create_or_get_document_job, create_reindex_job
from paper_rag.services.storage import save_uploaded_pdf
from paper_rag.services.vector_store import delete_document_vectors, get_chroma_collection

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    filename: str | None = Query(default=None, max_length=512),
    session: Session = Depends(get_db_session),
) -> DocumentListResponse:
    filename_query = (
        normalize_filename_search_key(filename.strip())
        if filename and filename.strip()
        else None
    )
    latest_profile_status = (
        select(PaperProfile.status)
        .where(PaperProfile.document_id == Document.id)
        .order_by(PaperProfile.profile_version.desc())
        .limit(1)
        .correlate(Document)
        .scalar_subquery()
    )
    statement = (
        select(
            Document,
            func.count(Chunk.id).label("chunk_count"),
            latest_profile_status.label("profile_status"),
        )
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc(), Document.id.desc())
    )
    if filename_query:
        statement = statement.where(
            Document.filename_search_key.contains(filename_query, autoescape=True)
        )
    if cursor is not None:
        cursor_document = session.get(Document, cursor)
        if cursor_document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document cursor not found")
        if filename_query and filename_query not in cursor_document.filename_search_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document cursor does not match filename filter",
            )
        statement = statement.where(
            or_(
                Document.created_at < cursor_document.created_at,
                and_(
                    Document.created_at == cursor_document.created_at,
                    Document.id < cursor_document.id,
                ),
            )
        )
    rows = session.execute(statement.limit(limit + 1)).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    document_ids = [document.id for document, _count, _profile in rows]
    latest_assessments: dict[UUID, DomainAssessment] = {}
    if document_ids:
        assessments = session.scalars(
            select(DomainAssessment)
            .where(
                DomainAssessment.document_id.in_(document_ids),
                DomainAssessment.applied_to_document.is_(True),
            )
            .order_by(
                DomainAssessment.document_id,
                DomainAssessment.created_at.desc(),
                DomainAssessment.id.desc(),
            )
        )
        for assessment in assessments:
            latest_assessments.setdefault(assessment.document_id, assessment)
    items = [
        DocumentListItem(
            document_id=document.id,
            original_filename=document.original_filename,
            status=document.status.value,
            pdf_type=document.pdf_type,
            page_count=document.page_count,
            chunk_count=chunk_count,
            domain_status=document.domain_status,
            domain_score=document.domain_score,
            domain_reasons=json.loads(document.domain_reasons_json or "[]"),
            domain_assessment_id=(
                latest_assessments[document.id].id
                if document.id in latest_assessments
                else None
            ),
            domain_decision_code=(
                latest_assessments[document.id].decision_code
                if document.id in latest_assessments
                else document.domain_decision_code
            ),
            domain_passed_requirements=(
                json.loads(
                    latest_assessments[document.id].passed_requirements_json
                    or "[]"
                )
                if document.id in latest_assessments
                else []
            ),
            domain_failed_requirements=(
                json.loads(
                    latest_assessments[document.id].failed_requirements_json
                    or "[]"
                )
                if document.id in latest_assessments
                else []
            ),
            domain_evidence=(
                json.loads(latest_assessments[document.id].evidence_json or "[]")
                if document.id in latest_assessments
                else []
            ),
            document_genre=document.document_genre,
            genre_score=document.genre_score,
            genre_decision_source=document.genre_decision_source,
            genre_scores=json.loads(document.genre_scores_json or "{}"),
            genre_evidence=json.loads(document.genre_evidence_json or "[]"),
            genre_conflicts=json.loads(document.genre_conflicts_json or "[]"),
            genre_manually_overridden=document.genre_manually_overridden,
            profile_status=profile_status,
            created_at=document.created_at,
        )
        for document, chunk_count, profile_status in rows
    ]
    return DocumentListResponse(
        items=items,
        next_cursor=items[-1].document_id if has_more and items else None,
    )


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile, session: Session = Depends(get_db_session), settings: Settings = Depends(get_settings)) -> UploadAccepted:
    upload = await save_uploaded_pdf(file, settings)
    document, job, created = create_or_get_document_job(session, upload)
    if not created:
        upload.path.unlink(missing_ok=True)
    return UploadAccepted(document_id=document.id, job_id=job.id, duplicate=not created, status=job.state.value)


@router.post("/{document_id}/reindex", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(
    document_id: UUID,
    session: Session = Depends(get_db_session),
) -> UploadAccepted:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    job = create_reindex_job(session, document)
    return UploadAccepted(
        document_id=document.id,
        job_id=job.id,
        duplicate=True,
        status=job.state.value,
    )


@router.post("/{document_id}/approve", response_model=DomainReviewResponse, status_code=status.HTTP_202_ACCEPTED)
async def approve_document(
    document_id: UUID,
    request: DomainApproveRequest,
    session: Session = Depends(get_db_session),
) -> DomainReviewResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.domain_status != DomainStatus.REVIEW_REQUIRED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document does not require approval")
    try:
        override = approve_domain_assessment(
            session,
            document,
            request.assessment_id,
            actor="local_user",
        )
    except DomainAssessmentConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    session.commit()
    job = create_reindex_job(session, document)
    return DomainReviewResponse(
        document_id=document.id,
        job_id=job.id,
        status=job.state.value,
        assessment_id=request.assessment_id,
        override_id=override.id,
    )


@router.post("/{document_id}/deletion-check", response_model=DeletionCheckResponse)
def deletion_check(
    document_id: UUID,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> DeletionCheckResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    pages = list(
        session.scalars(
            select(Page)
            .where(Page.document_id == document.id)
            .order_by(Page.page_number)
        )
    )
    provider = get_embedding_provider(settings)
    result = evaluate_domain_admission(
        [
            AdmissionPage(
                page.page_number,
                page.text,
                page.quality_score,
                page.ocr_confidence,
            )
            for page in pages
        ],
        provider,
        settings,
    )
    assessment = record_domain_assessment(
        session,
        document,
        result,
        trigger="deletion_check",
        applied=False,
    )
    session.commit()
    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    collection = get_chroma_collection(settings, provider)
    vector_count = len(collection.get(where={"document_id": str(document.id)}, include=[])["ids"])
    audit_count = session.scalar(select(func.count(AnswerAudit.id)).where(AnswerAudit.document_id == document.id)) or 0
    warning = None
    if result.decision == DomainStatus.ACCEPTED:
        warning = "相关性复检显示该文档属于超表面知识库；删除会移除可用于问答的证据。"
    token = deletion_confirmations.issue(document.id, document.original_filename, settings.deletion_token_ttl_seconds)
    return DeletionCheckResponse(
        document_id=document.id,
        original_filename=document.original_filename,
        stored_domain_status=document.domain_status,
        fresh_assessment_id=assessment.id,
        fresh_domain_status=result.decision,
        fresh_decision_code=result.decision_code,
        passed_requirements=list(result.passed_requirements),
        failed_requirements=list(result.failed_requirements),
        evidence=json.loads(assessment.evidence_json or "[]"),
        page_count=document.page_count or len(pages),
        chunk_count=len(chunks),
        vector_count=vector_count,
        answer_audit_count=audit_count,
        warning=warning,
        confirmation_token=token,
        expires_in_seconds=settings.deletion_token_ttl_seconds,
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: UUID,
    request: DeleteDocumentRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> DeleteDocumentResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not deletion_confirmations.consume(request.confirmation_token, document.id, request.confirm_filename):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Deletion confirmation is invalid or expired")
    active_states = {
        JobState.QUEUED,
        JobState.CLASSIFYING,
        JobState.PARSING,
        JobState.CHUNKING,
        JobState.EMBEDDING,
        JobState.INDEXING,
    }
    active_job_count = session.scalar(
        select(func.count(IngestionJob.id)).where(
            IngestionJob.document_id == document.id,
            IngestionJob.state.in_(active_states),
        )
    ) or 0
    if active_job_count:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document is still being processed")
    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    provider = get_embedding_provider(settings)
    collection = get_chroma_collection(settings, provider)
    existing_vector_count = len(collection.get(ids=[chunk.vector_id for chunk in chunks], include=[])["ids"]) if chunks else 0
    delete_document_vectors(collection, chunks, settings)
    stored_path = Path(document.stored_path)
    original_filename = document.original_filename
    parsed_path = settings.parsed_dir / f"{document.id}.jsonl"
    session.delete(document)
    session.commit()
    stored_path.unlink(missing_ok=True)
    parsed_path.unlink(missing_ok=True)
    return DeleteDocumentResponse(
        document_id=document_id,
        original_filename=original_filename,
        deleted_chunks=len(chunks),
        deleted_vectors=existing_vector_count,
    )
