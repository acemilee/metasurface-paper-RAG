from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.config import Settings, get_settings
from paper_rag.db import get_db_session
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.schemas.search import EvidenceItem, SearchRequest, SearchResponse
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.retrieval import has_sufficient_retrieval_evidence, retrieve_question_evidence
from paper_rag.services.query_intent import QueryIntent, classify_query_intent
from paper_rag.services.vector_store import VectorIndexUnavailableError, run_synced_chroma_query

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_evidence(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> SearchResponse:
    requested_ids = request.document_ids or ([request.document_id] if request.document_id else [])
    document_scope = None if request.scope == "all" and not requested_ids else list(dict.fromkeys(requested_ids))
    if request.scope == "selected" and not document_scope:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected scope requires at least one document")
    if document_scope:
        documents = list(session.scalars(select(Document).where(Document.id.in_(document_scope))))
        if len(documents) != len(document_scope):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Selected document not found")
        if any(document.status != DocumentStatus.COMPLETED for document in documents):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected document is not ready")
    provider = get_embedding_provider(settings)
    intent_result = classify_query_intent(request.question, provider)
    if intent_result.intent == QueryIntent.CROSS_DOCUMENT and document_scope is None:
        document_scope = list(session.scalars(select(Document.id).where(Document.status == DocumentStatus.COMPLETED)))
    if intent_result.intent == QueryIntent.OVERVIEW and (
        document_scope is None or len(document_scope) != 1
    ):
        return SearchResponse(sufficient=False, reason="概述问题需要明确指定一篇论文", intent=intent_result.intent, intent_confidence=intent_result.confidence, intent_source=intent_result.source, evidence=[])
    if intent_result.intent == QueryIntent.CROSS_DOCUMENT and len(document_scope or []) < 2:
        return SearchResponse(sufficient=False, reason="跨论文综合至少需要选择两篇论文", intent=intent_result.intent, intent_confidence=intent_result.confidence, intent_source=intent_result.source, evidence=[])
    query_document_ids = document_scope or list(
        session.scalars(select(Document.id).where(Document.status == DocumentStatus.COMPLETED))
    )
    try:
        candidates = run_synced_chroma_query(
            session,
            settings,
            provider,
            query_document_ids,
            lambda collection: retrieve_question_evidence(
                collection, provider, request.question, request.top_n, document_scope, intent_result
            ),
        )
    except VectorIndexUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "vector_index_unavailable", "message": str(exc)},
        ) from exc
    decision = has_sufficient_retrieval_evidence(request.question, candidates, settings, intent_result)
    return SearchResponse(sufficient=decision.sufficient, reason=decision.reason, intent=intent_result.intent, intent_confidence=intent_result.confidence, intent_source=intent_result.source, evidence=[EvidenceItem(**candidate.__dict__) for candidate in candidates])
