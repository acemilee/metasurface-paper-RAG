from __future__ import annotations

import json
import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.config import Settings, get_settings
from paper_rag.db import SessionLocal, get_db_session
from paper_rag.models.audit import AnswerAudit
from paper_rag.models.document import Document, DocumentStatus, DomainStatus
from paper_rag.models.formula import Formula
from paper_rag.schemas.chat import AnswerResponse, ChatRequest, GroundedClaim, ModelAnswer, SetApiKeyRequest
from paper_rag.schemas.query_plan import AnswerMode
from paper_rag.services.answer_audit import (
    AuditResult,
    MissingFormulaClaimError,
    UnknownCitationError,
    make_hypothesis_refusal,
    make_refusal,
    map_citation_ids,
    verify_citations_exist,
    verify_claim_tokens_against_evidence,
    verify_cross_document_citations,
    verify_formula_claims,
    verify_novelty_answer,
    verify_novelty_entailment_audit,
    render_derivations,
    render_grounded_claims,
    render_hypotheses,
    render_novelty_claims,
    salvage_partially_entailed_premises,
)
from paper_rag.services.deepseek import (
    DeepSeekProviderError,
    DeepSeekSchemaError,
    DeepSeekSessionKeyStore,
    generate_grounded_answer,
    generate_evidence_bounded_hypotheses,
    repair_evidence_bounded_hypothesis,
    audit_grounded_claims,
    audit_novelty_entailment,
    audit_grounded_answer,
    audit_hypothesis,
)
from paper_rag.services.conversation_memory import (
    begin_conversation_turn,
    build_conversation_context,
    complete_conversation_turn,
    fail_running_conversation_turn,
    resolve_conversation_request,
)
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.evidence_gate import evaluate_evidence
from paper_rag.services.formula_answers import build_direct_formula_response, load_formula_records_by_ids, load_formula_records_for_evidence, select_relevant_formula_records
from paper_rag.services.formula_query_guard import guard_formula_query, repair_pages_from_evidence, route_formula_query
from paper_rag.services.paper_profile import get_profile_retrieval_hints
from paper_rag.services.retrieval import retrieve_planned_evidence
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult
from paper_rag.services.query_rewrite import (
    QueryRewriteProviderError,
    QueryRewriteSchemaError,
    ScopeDocument,
    link_soft_query_entities,
    build_safe_fallback_plan,
    normalize_query_plan,
    resolve_linked_entities_with_evidence,
    repair_query_plan,
    rewrite_query,
    rewrite_fidelity_score,
    validate_plan_scope,
)
from paper_rag.services.references import (
    enqueue_reference_repairs,
    merge_resolved_reference_evidence,
    prepare_reference_control,
)
from paper_rag.services.references.types import ReferenceKind, ResolutionStatus
from paper_rag.services.vector_store import VectorIndexUnavailableError, run_synced_chroma_query

router = APIRouter(prefix="/api", tags=["chat"])

_background_answer_tasks: set[asyncio.Task[AnswerResponse]] = set()


def _retain_answer_task(task: asyncio.Task[AnswerResponse]) -> None:
    _background_answer_tasks.add(task)

    def release(completed: asyncio.Task[AnswerResponse]) -> None:
        _background_answer_tasks.discard(completed)
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(release)
_key_store: DeepSeekSessionKeyStore | None = None


def _attach_query_context(
    response: AnswerResponse,
    query_plan,
    linked_entities=None,
    reference_resolutions=None,
) -> AnswerResponse:
    response.query_plan = query_plan.model_dump(mode="json")
    response.answer_mode = query_plan.answer_mode
    response.epistemic_level = {
        AnswerMode.EXTRACT: "source_fact",
        AnswerMode.SYNTHESIZE: "evidence_synthesis",
        AnswerMode.COMPARE: "evidence_synthesis",
        AnswerMode.DERIVE: "deterministic_derivation",
        AnswerMode.HYPOTHESIZE: "evidence_bounded_hypothesis",
    }[query_plan.answer_mode]
    soft = [
        {
            "surface": entity.surface,
            "canonical": entity.canonical,
            "entity_type": entity.entity_type,
            "must_link": entity.must_link,
            "linked": entity.linked,
            "matched_document_ids": [str(document_id) for document_id in entity.matched_document_ids],
        }
        for entity in (linked_entities or [])
    ]
    strong = [item.as_dict() for item in (reference_resolutions or [])]
    response.entity_links = [*strong, *soft]
    return response


def _generation_failure_record(
    attempt: int,
    error: Exception,
    evidence: list,
    *,
    stage: str = "generation",
    latency_ms: int = 0,
    formula_diagnostics: list[dict] | None = None,
) -> dict:
    if isinstance(error, DeepSeekSchemaError):
        error_code = "model_schema_failure"
        validation_errors = error.validation_errors
        raw_hashes = error.raw_output_sha256
    elif isinstance(error, MissingFormulaClaimError):
        error_code = "missing_formula_claim"
        validation_errors = [str(error)[:300]]
        raw_hashes = []
    elif isinstance(error, UnknownCitationError):
        error_code = "unknown_citation"
        validation_errors = [str(error)[:300]]
        raw_hashes = []
    elif isinstance(error, DeepSeekProviderError):
        error_code = "provider_failure"
        validation_errors = [type(error).__name__]
        raw_hashes = []
    else:
        error_code = type(error).__name__
        validation_errors = [str(error)[:300]]
        raw_hashes = []
    return {
        "attempt": attempt,
        "stage": stage,
        "status": "failed",
        "error_code": error_code,
        "validation_errors": validation_errors[:10],
        "raw_output_sha256": raw_hashes[:3],
        "latency_ms": max(0, latency_ms),
        "allowed_citation_ids": [str(item.chunk_id) for item in evidence],
        "formula_diagnostics": list(formula_diagnostics or []),
    }


def _safe_formula_bbox(value: str) -> list[float] | None:
    try:
        bbox = [float(item) for item in json.loads(value)]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(bbox) != 4 or not all(math.isfinite(item) for item in bbox):
        return None
    return bbox


def _formula_diagnostics(session: Session, evidence: list) -> list[dict]:
    formula_to_chunks: dict[UUID, list[str]] = {}
    for item in evidence:
        for value in item.formula_ids:
            try:
                formula_id = UUID(value)
            except (TypeError, ValueError):
                continue
            formula_to_chunks.setdefault(formula_id, []).append(str(item.chunk_id))
    if not formula_to_chunks:
        return []
    formulas = session.scalars(select(Formula).where(Formula.id.in_(formula_to_chunks)))
    return [
        {
            "formula_id": str(formula.id),
            "group_key": formula.group_key,
            "formula_number": formula.formula_number,
            "page_number": formula.page_number,
            "bbox": _safe_formula_bbox(formula.bbox_json),
            "semantic_status": formula.semantic_status,
            "fidelity_status": formula.fidelity_status,
            "chunk_ids": formula_to_chunks[formula.id],
        }
        for formula in formulas
    ]


def _resolve_document_scope(request: ChatRequest, session: Session) -> list | None:
    requested_ids = request.document_ids or ([request.document_id] if request.document_id else [])
    if request.scope == "all" and not requested_ids:
        return None
    if not requested_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Selected scope requires at least one document",
        )
    documents = list(session.scalars(select(Document).where(Document.id.in_(requested_ids))))
    if len(documents) != len(set(requested_ids)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Selected document not found")
    if any(
        document.status != DocumentStatus.COMPLETED
        or document.domain_status
        not in {DomainStatus.ACCEPTED, DomainStatus.MANUAL_APPROVED}
        for document in documents
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected document is not ready")
    return list(dict.fromkeys(requested_ids))


def get_key_store(settings: Settings = Depends(get_settings)) -> DeepSeekSessionKeyStore:
    global _key_store
    if _key_store is None:
        _key_store = DeepSeekSessionKeyStore(settings.deepseek_key_ttl_seconds)
    return _key_store


def _save_audit(
    session: Session,
    request: ChatRequest,
    response: AnswerResponse,
    settings: Settings,
    scope_documents: list[ScopeDocument] | None = None,
    query_plan=None,
    linked_entities=None,
    rewrite_source: str | None = None,
    rewrite_error: dict | None = None,
    generation_attempts: list[dict] | None = None,
    semantic_audits: list[dict] | None = None,
) -> None:
    scope_documents = scope_documents or []
    selected_ids = [document.document_id for document in scope_documents]
    session.add(
        AnswerAudit(
            document_id=request.document_id or (selected_ids[0] if len(selected_ids) == 1 else None) or (response.citations[0].document_id if response.citations else None),
            question=request.question,
            answer=response.answer,
            evidence_status=response.evidence_status,
            refusal_reason=response.refusal_reason,
            hallucination_risk=response.hallucination_risk,
            audit_result=response.audit_result,
            action=response.action,
            unsupported_parts_json=json.dumps(response.unsupported_parts, ensure_ascii=False),
            citation_ids_json=json.dumps([str(item.citation_id) for item in response.citations]),
            model_name=settings.deepseek_model,
            prompt_version=settings.prompt_version,
            selected_document_ids_json=json.dumps([str(document_id) for document_id in selected_ids]),
            query_plan_json=json.dumps(query_plan.model_dump(mode="json") if query_plan else {}, ensure_ascii=False),
            entity_links_json=json.dumps(response.entity_links or [], ensure_ascii=False),
            rewrite_source=rewrite_source,
            rewrite_error_json=json.dumps(rewrite_error or {}, ensure_ascii=False),
            document_genres_json=json.dumps({str(document.document_id): document.genre for document in scope_documents}, ensure_ascii=False),
            generation_attempts_json=json.dumps(generation_attempts or [], ensure_ascii=False),
            semantic_audit_json=json.dumps(semantic_audits or [], ensure_ascii=False),
        )
    )
    session.commit()


@router.post("/session/deepseek-key", status_code=status.HTTP_204_NO_CONTENT)
async def set_deepseek_key(
    request: SetApiKeyRequest,
    key_store: DeepSeekSessionKeyStore = Depends(get_key_store),
) -> None:
    try:
        key_store.set_key(request.session_id, request.api_key.get_secret_value())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.delete("/session/deepseek-key/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_deepseek_key(
    session_id: str,
    key_store: DeepSeekSessionKeyStore = Depends(get_key_store),
) -> None:
    key_store.clear_key(session_id)


@router.post("/chat", response_model=AnswerResponse)
async def ask_question(
    request: ChatRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    key_store: DeepSeekSessionKeyStore = Depends(get_key_store),
) -> AnswerResponse:
    async def ignore_progress(event: str, payload: dict) -> None:
        pass

    return await _answer_question(request, session, settings, key_store, ignore_progress)


async def _answer_question(
    request: ChatRequest,
    session: Session,
    settings: Settings,
    key_store: DeepSeekSessionKeyStore,
    emit: Callable[[str, dict], Awaitable[None]],
) -> AnswerResponse:
    request = resolve_conversation_request(session, request)
    document_scope = _resolve_document_scope(request, session)
    if document_scope is None:
        scope_records = list(
            session.scalars(
                select(Document)
                .where(
                    Document.status == DocumentStatus.COMPLETED,
                    Document.domain_status.in_(
                        [DomainStatus.ACCEPTED, DomainStatus.MANUAL_APPROVED]
                    ),
                )
                .order_by(Document.created_at.desc())
            )
        )
    else:
        scope_records = list(session.scalars(select(Document).where(Document.id.in_(document_scope))))
    scope_documents = [ScopeDocument(document.id, document.original_filename, document.document_genre) for document in scope_records]
    document_scope = [document.document_id for document in scope_documents]
    query_plan = None
    linked_entities = []
    reference_resolutions = ()
    rewrite_source = "deepseek_rewrite"
    rewrite_error_context = {}
    generation_attempt_records = []
    semantic_audit_records = []
    provider = get_embedding_provider(settings)
    question_embedding = provider.embed_query(request.question) if request.conversation_id else None
    conversation_context = build_conversation_context(
        session,
        request.conversation_id,
        query_embedding=question_embedding,
    )

    def save_current(response: AnswerResponse) -> None:
        _save_audit(
            session,
            request,
            response,
            settings,
            scope_documents,
            query_plan,
            linked_entities,
            rewrite_source,
            rewrite_error_context,
            generation_attempt_records,
            semantic_audit_records,
        )
        complete_conversation_turn(
            session,
            request,
            response,
            query_plan,
            linked_entities,
            question_embedding,
        )
    await emit(
        "scope",
        {
            "message": "已确定检索范围",
            "scope": request.scope,
            "document_count": len(document_scope),
        },
    )
    api_key = key_store.get_key(request.session_id)
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="DeepSeek API key is missing or expired")
    turn_start = begin_conversation_turn(session, request)
    if turn_start.replayed_response is not None:
        await emit("replayed", {"message": "已返回该提问的已完成结果"})
        return turn_start.replayed_response
    await emit("rewrite", {"message": "正在理解问题并构建检索计划", "model": settings.deepseek_model})
    try:
        query_plan = await rewrite_query(
            api_key,
            request.question,
            scope_documents,
            request.scope,
            settings,
            conversation_context,
        )
    except QueryRewriteSchemaError as first_error:
        rewrite_error_context = {
            "initial_raw_sha256": first_error.raw_sha256,
            "initial_validation_errors": first_error.validation_errors,
        }
        await emit("rewrite_repair", {"message": "检索计划结构不合法，正在执行 Schema 修复", "validation_error_count": len(first_error.validation_errors)})
        try:
            query_plan = await repair_query_plan(api_key, request.question, first_error, settings)
            rewrite_source = "deepseek_schema_repair"
        except QueryRewriteSchemaError as repair_error:
            rewrite_error_context.update({
                "repair_raw_sha256": repair_error.raw_sha256,
                "repair_validation_errors": repair_error.validation_errors,
            })
            query_plan = build_safe_fallback_plan(request.question, provider, scope_documents)
            rewrite_source = "bge_safe_fallback"
            await emit("rewrite_fallback", {"message": "Schema 修复失败，已切换到本地语义安全计划"})
        except QueryRewriteProviderError:
            query_plan = build_safe_fallback_plan(request.question, provider, scope_documents)
            rewrite_source = "bge_safe_fallback"
            await emit("rewrite_fallback", {"message": "Schema 修复服务不可用，已切换到本地语义安全计划"})
    except QueryRewriteProviderError:
        response = make_refusal("问题理解模型服务不可用", "rewrite_provider_failure", "error")
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    if query_plan.intent == QueryIntent.GENERAL:
        semantic_fallback = build_safe_fallback_plan(request.question, provider, scope_documents)
        if semantic_fallback.intent != QueryIntent.GENERAL and semantic_fallback.confidence >= 0.45:
            query_plan = semantic_fallback
            rewrite_source = "bge_general_intent_correction"
            await emit("rewrite_fallback", {"message": "通用意图与本地语义候选冲突，已切换到领域检索计划"})
    query_plan = normalize_query_plan(query_plan, scope_documents, provider)
    intent_result = QueryIntentResult(query_plan.intent, query_plan.confidence, 1.0, rewrite_source)
    await emit("intent", {"message": "已生成结构化检索计划", "intent": query_plan.intent, "confidence": query_plan.confidence, "source": rewrite_source, "retrieval_query_count": len(query_plan.retrieval_queries)})
    fidelity_input = request.question
    if conversation_context and conversation_context["recent_messages_untrusted_data"]:
        recent_context = conversation_context["recent_messages_untrusted_data"][-6:]
        fidelity_input = json.dumps(
            {"recent_context": recent_context, "current_question": request.question},
            ensure_ascii=False,
        )
    fidelity_score = rewrite_fidelity_score(provider, fidelity_input, query_plan.standalone_question)
    await emit("rewrite_validation", {"message": "已校验改写与原问题的一致性", "fidelity_score": fidelity_score})
    if fidelity_score < settings.rewrite_min_fidelity_score:
        response = _attach_query_context(make_refusal("问题改写与原始问题的语义一致性不足", "rewrite_fidelity_failure", "clarify"), query_plan)
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    scope_error = validate_plan_scope(query_plan, scope_documents)
    if scope_error:
        response = _attach_query_context(make_refusal(scope_error, action="clarify"), query_plan)
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    prepared = prepare_reference_control(
        session,
        request.question,
        query_plan,
        document_scope,
    )
    reference_resolutions = prepared.resolutions
    resolved_formula_ids = [
        target_id
        for resolution in reference_resolutions
        if resolution.reference.kind == ReferenceKind.FORMULA
        and resolution.status == ResolutionStatus.RESOLVED
        for target_id in resolution.target_ids
    ]
    linked_entities = list(prepared.soft_entities)
    repair_job_ids = (
        enqueue_reference_repairs(session, reference_resolutions)
        if not prepared.decision.proceed
        else ()
    )
    if prepared.references:
        await emit(
            "reference_resolution",
            {
                "message": (
                    "强标识已解析"
                    if prepared.decision.proceed
                    else prepared.decision.reason
                ),
                "resolutions": [
                    item.as_dict() for item in reference_resolutions
                ],
                "repair_job_ids": [str(item) for item in repair_job_ids],
            },
        )
    if not prepared.decision.proceed:
        response = _attach_query_context(
            make_refusal(
                prepared.decision.reason or "强标识解析失败",
                prepared.decision.audit_result,
                prepared.decision.action,
            ),
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    await emit("entity_link", {"message": "已完成实体候选抽取，等待证据语义验证", "entity_count": len(linked_entities), "linked_count": sum(entity.linked for entity in linked_entities)})
    if query_plan.intent == QueryIntent.OUT_OF_SCOPE:
        response = _attach_query_context(
            make_refusal("问题意图超出当前论文库范围"),
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    await emit("retrieval", {"message": "正在按 QueryPlan 检索库内证据"})
    profile_hints = get_profile_retrieval_hints(
        session,
        document_scope,
        query_plan.required_evidence,
    )
    if profile_hints:
        await emit(
            "profile_navigation",
            {"message": "已读取论文结构地图并定位证据角色", "role_count": len(profile_hints)},
        )
    try:
        evidence = run_synced_chroma_query(
            session,
            settings,
            provider,
            document_scope,
            lambda collection: retrieve_planned_evidence(
                collection, provider, query_plan, request.top_n, document_scope, profile_hints
            ),
        )
    except VectorIndexUnavailableError as exc:
        response = _attach_query_context(
            make_refusal(str(exc), "vector_index_unavailable", "error"),
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit("error", {"message": response.refusal_reason, "code": "vector_index_unavailable"})
        return response
    evidence = merge_resolved_reference_evidence(
        session,
        evidence,
        reference_resolutions,
    )
    linked_entities = resolve_linked_entities_with_evidence(linked_entities, evidence, provider)
    required_entities = [entity for entity in linked_entities if entity.must_link]
    unresolved = [entity.surface for entity in required_entities if not entity.linked]
    resolved_required = [entity for entity in required_entities if entity.linked]
    await emit("entity_validation", {"message": "已使用召回证据验证关键实体", "linked_count": len(resolved_required), "unresolved": unresolved})
    if unresolved and not resolved_required:
        response = _attach_query_context(
            make_refusal(
                f"问题中的关键实体未能链接到所选论文：{'、'.join(unresolved)}"
            ),
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response
    await emit(
        "retrieved",
        {
            "message": "证据召回完成",
            "evidence_count": len(evidence),
            "document_count": len({item.document_id for item in evidence}),
            "pages": sorted({page for item in evidence for page in range(item.page_start, item.page_end + 1)}),
        },
    )
    await emit("gate", {"message": "正在检查证据充分性"})
    document_genres = [document.genre for document in scope_documents]
    decision = evaluate_evidence(
        session,
        query_plan.standalone_question,
        evidence,
        settings,
        intent_result,
        query_plan,
        document_genres,
    )
    if not decision.sufficient:
        response = _attach_query_context(
            make_refusal(decision.reason or "证据不足"),
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit("refused", {"message": response.refusal_reason})
        return response

    formula_diagnostics = _formula_diagnostics(session, evidence)
    if query_plan.intent == QueryIntent.FORMULA and query_plan.answer_mode != AnswerMode.EXTRACT:
        route = route_formula_query(query_plan.answer_mode)
        if resolved_formula_ids:
            guarded_formulas = load_formula_records_by_ids(
                session,
                resolved_formula_ids,
            )
        elif route.value == "compare":
            guarded_formulas = load_formula_records_for_evidence(session, evidence)
        else:
            guarded_formulas = select_relevant_formula_records(
                session,
                query_plan.standalone_question,
                evidence,
            )
        readiness = guard_formula_query(
            session,
            guarded_formulas,
            route,
            repair_pages=repair_pages_from_evidence(evidence),
        )
        if not readiness.ready:
            response = _attach_query_context(
                make_refusal(readiness.reason, readiness.audit_result, "refuse"),
                query_plan,
                linked_entities,
                reference_resolutions,
            )
            save_current(response)
            await emit("refused", {"message": response.refusal_reason})
            return response
    direct_formula_response = build_direct_formula_response(
        session,
        query_plan.standalone_question,
        evidence,
        query_plan,
        resolved_formula_ids=resolved_formula_ids,
    )
    if direct_formula_response is not None:
        response = _attach_query_context(
            direct_formula_response,
            query_plan,
            linked_entities,
            reference_resolutions,
        )
        save_current(response)
        await emit(
            "formula_source",
            {
                "message": "已从原始 PDF 坐标区域提取公式",
                "formula_count": len(response.formula_assets),
                "pages": sorted({item.page_number for item in response.formula_assets}),
            },
        )
        await emit("complete", {"message": "公式原文与引用校验通过"})
        return response

    await emit("generation", {"message": "证据门控通过，正在生成回答", "model": settings.deepseek_model})
    response: AnswerResponse | None = None
    previous_answer = None
    audit_feedback = None
    max_generation_attempts = 2 if query_plan.answer_mode == AnswerMode.HYPOTHESIZE else 3
    for generation_attempt in range(max_generation_attempts):
        attempt_started = time.perf_counter()
        try:
            if query_plan.answer_mode == AnswerMode.HYPOTHESIZE:
                hypotheses = await generate_evidence_bounded_hypotheses(
                    api_key, query_plan.standalone_question, evidence, settings
                )
                hypothesis_citations = list(dict.fromkeys(
                    citation_id
                    for hypothesis in hypotheses
                    for premise in hypothesis.premises
                    for citation_id in premise.citation_ids
                ))
                model_answer = ModelAnswer(
                    answer="",
                    citation_ids=hypothesis_citations,
                    hallucination_risk="medium",
                    hypotheses=hypotheses,
                )
            else:
                model_answer = await generate_grounded_answer(
                    api_key,
                    query_plan.standalone_question,
                    evidence,
                    settings,
                    query_plan,
                    document_genres,
                    audit_feedback,
                    previous_answer,
                    unresolved,
                )
                if intent_result.intent != QueryIntent.NOVELTY:
                    if (
                        query_plan.intent == QueryIntent.FORMULA
                        and query_plan.answer_mode != AnswerMode.EXTRACT
                        and not model_answer.formula_claims
                    ):
                        raise MissingFormulaClaimError(
                            "Structured formula claims are required for formula explanation"
                        )
                    if query_plan.answer_mode in {AnswerMode.EXTRACT, AnswerMode.SYNTHESIZE, AnswerMode.COMPARE} and not model_answer.claims:
                        raise DeepSeekSchemaError(
                            "Structured claims are required for this answer mode",
                            validation_errors=["claims: Field required"],
                        )
                    if query_plan.answer_mode == AnswerMode.DERIVE and not model_answer.derivations:
                        raise DeepSeekSchemaError(
                            "Structured derivations are required for derive mode",
                            validation_errors=["derivations: Field required"],
                        )
            generation_attempt_records.append({
                "attempt": generation_attempt + 1,
                "answer": model_answer.answer,
                "citation_ids": [str(item) for item in model_answer.citation_ids],
                "hallucination_risk": model_answer.hallucination_risk,
                "novelty_claims": [item.model_dump(mode="json") for item in model_answer.novelty_claims],
                "claims": [item.model_dump(mode="json") for item in model_answer.claims],
                "derivations": [item.model_dump(mode="json") for item in model_answer.derivations],
                "hypotheses": [item.model_dump(mode="json") for item in model_answer.hypotheses],
            })
            if intent_result.intent != QueryIntent.NOVELTY:
                valid_ids = {item.chunk_id for item in evidence}
                submitted_ids = {
                    *model_answer.citation_ids,
                    *(citation_id for claim in model_answer.claims for citation_id in claim.citation_ids),
                    *(citation_id for item in model_answer.derivations for citation_id in item.citation_ids),
                    *(
                        citation_id
                        for hypothesis in model_answer.hypotheses
                        for premise in hypothesis.premises
                        for citation_id in premise.citation_ids
                    ),
                }
                unknown_ids = submitted_ids - valid_ids
                if unknown_ids:
                    raise UnknownCitationError(
                        f"Model returned {len(unknown_ids)} citation IDs outside the whitelist"
                    )
                valid_claims = [item for item in model_answer.claims if set(item.citation_ids).issubset(valid_ids)]
                valid_derivations = [item for item in model_answer.derivations if set(item.citation_ids).issubset(valid_ids)]
                valid_hypotheses = [
                    item for item in model_answer.hypotheses
                    if all(set(premise.citation_ids).issubset(valid_ids) for premise in item.premises)
                ]
                model_answer = model_answer.model_copy(update={
                    "citation_ids": [item for item in model_answer.citation_ids if item in valid_ids],
                    "claims": valid_claims,
                    "derivations": valid_derivations,
                    "hypotheses": valid_hypotheses,
                })
            citations = map_citation_ids(session, model_answer, evidence)
        except (DeepSeekSchemaError, MissingFormulaClaimError, UnknownCitationError) as exc:
            generation_attempt_records.append(
                _generation_failure_record(
                    generation_attempt + 1,
                    exc,
                    evidence,
                    latency_ms=round((time.perf_counter() - attempt_started) * 1000),
                    formula_diagnostics=formula_diagnostics,
                )
            )
            if generation_attempt < max_generation_attempts - 1:
                continue
            code = _generation_failure_record(generation_attempt + 1, exc, evidence)["error_code"]
            response = make_refusal("模型结构或引用未通过校验", code, "error")
            break
        except DeepSeekProviderError as exc:
            generation_attempt_records.append(
                _generation_failure_record(
                    generation_attempt + 1,
                    exc,
                    evidence,
                    latency_ms=round((time.perf_counter() - attempt_started) * 1000),
                    formula_diagnostics=formula_diagnostics,
                )
            )
            response = make_refusal("模型服务不可用", "provider_failure", "error")
            break

        citation_audit = verify_citations_exist(citations, evidence)
        await emit("audit", {"message": "正在校验引用、数值和公式"})
        claim_audit = AuditResult(True)
        formula_audit = verify_formula_claims(session, model_answer, citations, evidence)
        cross_document_audit = verify_cross_document_citations(citations) if intent_result.intent == QueryIntent.CROSS_DOCUMENT else None
        novelty_partial = (
            intent_result.intent == QueryIntent.NOVELTY
            and not model_answer.novelty_claims
            and bool(model_answer.answer.strip())
            and bool(citations)
            and any(item.value == "chapter_result" for item in query_plan.required_evidence)
        )
        novelty_audit = verify_novelty_answer(model_answer, citations, evidence, document_genres) if intent_result.intent == QueryIntent.NOVELTY and not novelty_partial else None
        novelty_entailment_audit = None
        novelty_semantic_partial = False
        novelty_unsupported_parts: list[str] = []
        novelty_terminal_response: AnswerResponse | None = None
        hypothesis_terminal_response: AnswerResponse | None = None
        general_entailment_audit = None
        structured_answer: str | None = None
        structured_partial = False
        structured_unsupported_parts: list[str] = []
        structured_claim_details: list[dict] = []
        if intent_result.intent == QueryIntent.NOVELTY and novelty_audit and novelty_audit.passed:
            await emit("semantic_audit", {"message": "正在执行跨语言创新主张蕴含审计", "claim_count": len(model_answer.novelty_claims)})
            try:
                entailment_report = await audit_novelty_entailment(
                    api_key,
                    model_answer.answer,
                    model_answer.novelty_claims,
                    evidence,
                    settings,
                )
                semantic_audit_records.append(entailment_report.model_dump(mode="json"))
                entailed_indexes = {
                    item.claim_index for item in entailment_report.results if item.verdict == "entailed"
                }
                if entailed_indexes and len(entailed_indexes) < len(model_answer.novelty_claims):
                    novelty_unsupported_parts = [
                        f"主张{item.claim_index + 1}（{item.verdict}）：{item.reason}"
                        for item in entailment_report.results if item.verdict != "entailed"
                    ]
                    model_answer = model_answer.model_copy(update={
                        "novelty_claims": [
                            claim for index, claim in enumerate(model_answer.novelty_claims)
                            if index in entailed_indexes
                        ]
                    })
                    novelty_semantic_partial = True
                    novelty_entailment_audit = AuditResult(True)
                elif not entailed_indexes:
                    unavailable = [item for item in entailment_report.results if item.verdict == "audit_unavailable"]
                    if unavailable:
                        novelty_terminal_response = make_refusal(
                            "创新主张审计暂不可用，未展示未经审计的内容",
                            "claim_audit_unavailable",
                            "error",
                        )
                    else:
                        detail = "、".join(item.reason for item in entailment_report.results)
                        novelty_terminal_response = make_refusal(
                            f"检索到的原文未完整蕴含生成的创新主张：{detail}",
                            "novelty_not_entailed",
                            "refuse",
                        )
                else:
                    novelty_entailment_audit = verify_novelty_entailment_audit(
                        entailment_report,
                        len(model_answer.novelty_claims),
                    )
            except DeepSeekSchemaError:
                novelty_entailment_audit = AuditResult(False, "创新语义审计返回结构不合法")
            except DeepSeekProviderError:
                novelty_entailment_audit = AuditResult(False, "创新语义审计服务不可用")
        elif intent_result.intent != QueryIntent.NOVELTY and citation_audit.passed and query_plan.answer_mode == AnswerMode.HYPOTHESIZE:
            await emit("semantic_audit", {"message": "正在审计推理前提、条件和反证", "hypothesis_count": len(model_answer.hypotheses)})
            accepted_hypotheses = []
            original_hypothesis_count = len(model_answer.hypotheses)
            for hypothesis_index, hypothesis in enumerate(model_answer.hypotheses):
                premise_claims = [
                    GroundedClaim(text=premise.claim, citation_ids=premise.citation_ids, claim_type="direct_fact", label="推理前提")
                    for premise in hypothesis.premises
                ]
                premise_report = await audit_grounded_claims(api_key, premise_claims, evidence, settings)
                semantic_audit_records.append({"hypothesis_index": hypothesis_index, "premises": premise_report.model_dump(mode="json")})
                failed_premises = [item for item in premise_report.results if item.verdict != "entailed"]
                if failed_premises:
                    salvaged_hypothesis = salvage_partially_entailed_premises(hypothesis, premise_report)
                    if salvaged_hypothesis is None:
                        structured_unsupported_parts.append(
                            f"假设{hypothesis_index + 1}前提未通过：" + "；".join(item.reason for item in failed_premises)
                        )
                        continue
                    salvaged_claims = [
                        GroundedClaim(text=premise.claim, citation_ids=premise.citation_ids, claim_type="direct_fact", label="裁剪后推理前提")
                        for premise in salvaged_hypothesis.premises
                    ]
                    salvaged_report = await audit_grounded_claims(api_key, salvaged_claims, evidence, settings)
                    semantic_audit_records.append({"hypothesis_index": hypothesis_index, "salvaged_premises": salvaged_report.model_dump(mode="json")})
                    if any(item.verdict != "entailed" for item in salvaged_report.results):
                        structured_unsupported_parts.append(f"假设{hypothesis_index + 1}裁剪后前提仍未通过")
                        continue
                    hypothesis = salvaged_hypothesis
                try:
                    hypothesis_report = await audit_hypothesis(api_key, hypothesis, evidence, settings)
                    semantic_audit_records.append({"hypothesis_index": hypothesis_index, "inference": hypothesis_report.model_dump(mode="json")})
                except (DeepSeekSchemaError, DeepSeekProviderError) as exc:
                    structured_unsupported_parts.append(f"假设{hypothesis_index + 1}审计不可用：{type(exc).__name__}")
                    continue
                if hypothesis_report.verdict != "supported_hypothesis":
                    try:
                        repaired_hypothesis = await repair_evidence_bounded_hypothesis(
                            api_key,
                            request.question,
                            hypothesis,
                            hypothesis_report,
                            evidence,
                            settings,
                        )
                        repaired_premises = [
                            GroundedClaim(text=premise.claim, citation_ids=premise.citation_ids, claim_type="direct_fact", label="修复后推理前提")
                            for premise in repaired_hypothesis.premises
                        ]
                        repaired_premise_report = await audit_grounded_claims(api_key, repaired_premises, evidence, settings)
                        repaired_report = await audit_hypothesis(api_key, repaired_hypothesis, evidence, settings)
                        semantic_audit_records.append({
                            "hypothesis_index": hypothesis_index,
                            "repair": {
                                "premises": repaired_premise_report.model_dump(mode="json"),
                                "inference": repaired_report.model_dump(mode="json"),
                            },
                        })
                        if all(item.verdict == "entailed" for item in repaired_premise_report.results) and repaired_report.verdict == "supported_hypothesis":
                            hypothesis = repaired_hypothesis
                        else:
                            structured_unsupported_parts.append(f"假设{hypothesis_index + 1}越界：{repaired_report.reason}")
                            continue
                    except (DeepSeekSchemaError, DeepSeekProviderError) as exc:
                        structured_unsupported_parts.append(f"假设{hypothesis_index + 1}修复不可用：{type(exc).__name__}")
                        continue
                accepted_hypotheses.append(hypothesis)
                structured_claim_details.append({"type": "hypothesis", **hypothesis.model_dump(mode="json")})
            if accepted_hypotheses:
                model_answer = model_answer.model_copy(update={"hypotheses": accepted_hypotheses})
                structured_answer = render_hypotheses(accepted_hypotheses)
                structured_partial = len(accepted_hypotheses) < original_hypothesis_count or bool(structured_unsupported_parts)
                general_entailment_audit = AuditResult(True)
                claim_audit = AuditResult(True)
            else:
                hypothesis_terminal_response = make_hypothesis_refusal(structured_unsupported_parts)
                general_entailment_audit = AuditResult(True)
                claim_audit = AuditResult(True)
        elif intent_result.intent != QueryIntent.NOVELTY and citation_audit.passed and query_plan.answer_mode == AnswerMode.DERIVE:
            await emit("semantic_audit", {"message": "正在校验推导输入、运算和结果", "derivation_count": len(model_answer.derivations)})
            derivation_claims = [
                GroundedClaim(text=item.statement, citation_ids=item.citation_ids, claim_type="synthesized_fact", label="确定性推导")
                for item in model_answer.derivations
            ]
            report = await audit_grounded_claims(api_key, derivation_claims, evidence, settings) if derivation_claims else None
            if report:
                semantic_audit_records.append({"derivations": report.model_dump(mode="json")})
            entailed = {item.claim_index for item in report.results if item.verdict == "entailed"} if report else set()
            accepted = [item for index, item in enumerate(model_answer.derivations) if index in entailed]
            structured_unsupported_parts.extend(
                f"推导{item.claim_index + 1}（{item.verdict}）：{item.reason}"
                for item in (report.results if report else []) if item.verdict != "entailed"
            )
            if accepted:
                model_answer = model_answer.model_copy(update={"derivations": accepted})
                structured_answer = render_derivations(accepted)
                structured_partial = len(accepted) < len(derivation_claims)
                structured_claim_details = [{"type": "deterministic_derivation", **item.model_dump(mode="json")} for item in accepted]
                audit_copy = model_answer.model_copy(update={"answer": structured_answer})
                claim_audit = verify_claim_tokens_against_evidence(audit_copy, citations, evidence)
                general_entailment_audit = AuditResult(True)
            else:
                claim_audit = AuditResult(False, "缺少可审计的确定性推导")
                general_entailment_audit = AuditResult(True)
        elif intent_result.intent != QueryIntent.NOVELTY and citation_audit.passed and model_answer.claims:
            await emit("semantic_audit", {"message": "正在逐条审计精炼回答主张", "claim_count": len(model_answer.claims)})
            report = await audit_grounded_claims(api_key, model_answer.claims, evidence, settings)
            semantic_audit_records.append({"grounded_claims": report.model_dump(mode="json")})
            entailed = {item.claim_index for item in report.results if item.verdict == "entailed"}
            original_count = len(model_answer.claims)
            accepted = [item for index, item in enumerate(model_answer.claims) if index in entailed]
            structured_unsupported_parts.extend(
                f"主张{item.claim_index + 1}（{item.verdict}）：{item.reason}"
                for item in report.results if item.verdict != "entailed"
            )
            if accepted:
                model_answer = model_answer.model_copy(update={"claims": accepted})
                structured_answer = render_grounded_claims(accepted)
                structured_partial = len(accepted) < original_count
                structured_claim_details = [{"type": item.claim_type, **item.model_dump(mode="json")} for item in accepted]
                audit_copy = model_answer.model_copy(update={"answer": structured_answer})
                claim_audit = verify_claim_tokens_against_evidence(audit_copy, citations, evidence)
                general_entailment_audit = AuditResult(True)
            else:
                claim_audit = AuditResult(False, "没有通过逐条证据审计的回答主张")
                general_entailment_audit = AuditResult(True)
        elif (intent_result.intent != QueryIntent.NOVELTY or novelty_partial) and citation_audit.passed:
            await emit("semantic_audit", {"message": "正在执行回答与引用证据蕴含审计"})
            try:
                entailment_report = await audit_grounded_answer(api_key, model_answer, evidence, settings)
                semantic_audit_records.append({"general_answer": entailment_report.model_dump(mode="json")})
                if entailment_report.verdict != "entailed":
                    detail = "、".join(entailment_report.unsupported_parts) or entailment_report.reason
                    general_entailment_audit = AuditResult(False, f"回答存在引用证据未蕴含的内容：{detail}")
                else:
                    general_entailment_audit = AuditResult(True)
                    claim_audit = verify_claim_tokens_against_evidence(model_answer, citations, evidence)
            except DeepSeekSchemaError:
                general_entailment_audit = AuditResult(False, "回答语义审计返回结构不合法")
            except DeepSeekProviderError:
                general_entailment_audit = AuditResult(False, "回答语义审计服务不可用")
        used_citation_ids = None
        if structured_answer is not None:
            if query_plan.answer_mode in {AnswerMode.EXTRACT, AnswerMode.SYNTHESIZE, AnswerMode.COMPARE}:
                used_citation_ids = {citation_id for item in model_answer.claims for citation_id in item.citation_ids}
            elif query_plan.answer_mode == AnswerMode.DERIVE:
                used_citation_ids = {citation_id for item in model_answer.derivations for citation_id in item.citation_ids}
            elif query_plan.answer_mode == AnswerMode.HYPOTHESIZE:
                used_citation_ids = {
                    citation_id
                    for item in model_answer.hypotheses
                    for premise in item.premises
                    for citation_id in premise.citation_ids
                }
        elif intent_result.intent == QueryIntent.NOVELTY and model_answer.novelty_claims:
            used_citation_ids = {item.citation_id for item in model_answer.novelty_claims}
        if used_citation_ids is not None:
            citations = [item for item in citations if item.citation_id in used_citation_ids]
            citation_audit = verify_citations_exist(citations, evidence)
            if intent_result.intent == QueryIntent.CROSS_DOCUMENT:
                cross_document_audit = verify_cross_document_citations(citations)
        if novelty_terminal_response is not None:
            response = novelty_terminal_response
            break
        if hypothesis_terminal_response is not None:
            response = hypothesis_terminal_response
            break
        audit_failure = (
            citation_audit.reason
            or claim_audit.reason
            or formula_audit.reason
            or (cross_document_audit.reason if cross_document_audit and not cross_document_audit.passed else None)
            or (novelty_audit.reason if novelty_audit and not novelty_audit.passed else None)
            or (novelty_entailment_audit.reason if novelty_entailment_audit and not novelty_entailment_audit.passed else None)
            or (general_entailment_audit.reason if general_entailment_audit and not general_entailment_audit.passed else None)
        )
        if audit_failure:
            if generation_attempt < max_generation_attempts - 1:
                previous_answer = model_answer
                audit_feedback = (
                    f"上一次答案未通过审计：{audit_failure}。"
                    f"只能使用这些 citation_id：{[str(item.chunk_id) for item in evidence]}。"
                    "请修复引用覆盖和措辞，不得增加任何新事实。"
                )
                await emit("answer_repair", {"message": "答案审计未通过，正在修复引用覆盖", "reason": audit_failure})
                continue
            response = make_refusal(audit_failure, audit_result="failed_after_generation", action="error" if "结构" in audit_failure or "服务" in audit_failure else "refuse")
        else:
            final_answer = (
                render_novelty_claims(model_answer)
                if intent_result.intent == QueryIntent.NOVELTY and not novelty_partial
                else structured_answer or model_answer.answer
            )
            epistemic_level = {
                AnswerMode.EXTRACT: "source_fact",
                AnswerMode.SYNTHESIZE: "evidence_synthesis",
                AnswerMode.COMPARE: "evidence_synthesis",
                AnswerMode.DERIVE: "deterministic_derivation",
                AnswerMode.HYPOTHESIZE: "evidence_bounded_hypothesis",
            }[query_plan.answer_mode]
            response = AnswerResponse(
                answer=final_answer,
                citations=citations,
                evidence_status="sufficient",
                refused=False,
                refusal_reason=None,
                hallucination_risk=model_answer.hallucination_risk,
                audit_result="passed",
                action="answer",
                unsupported_parts=unresolved,
                answer_mode=query_plan.answer_mode,
                epistemic_level=epistemic_level,
                claim_details=structured_claim_details,
            )
            if unresolved:
                response.action = "partial"
            if novelty_partial:
                response.action = "partial"
                response.unsupported_parts = ["创新点缺少明确原文证据"]
            if novelty_semantic_partial:
                response.action = "partial"
                response.unsupported_parts = novelty_unsupported_parts
            if structured_partial or structured_unsupported_parts:
                response.action = "partial"
                response.unsupported_parts = [*response.unsupported_parts, *structured_unsupported_parts]
        break
    if response is None:
        response = make_refusal("模型未返回可审计结果", "generation_exhausted", "error")
    _attach_query_context(
        response,
        query_plan,
        linked_entities,
        reference_resolutions,
    )
    save_current(response)
    await emit(
        "complete" if not response.refused else "refused",
        {"message": "回答审计通过" if not response.refused else response.refusal_reason},
    )
    return response


@router.post("/chat/stream")
async def stream_question(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    key_store: DeepSeekSessionKeyStore = Depends(get_key_store),
) -> StreamingResponse:
    async def event_stream():
        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

        async def emit(event: str, payload: dict) -> None:
            await queue.put((event, payload))
            await asyncio.sleep(0)

        async def run_answer() -> AnswerResponse:
            with SessionLocal() as session:
                try:
                    return await _answer_question(request, session, settings, key_store, emit)
                except Exception:
                    session.rollback()
                    try:
                        fail_running_conversation_turn(session, request)
                    except Exception:
                        session.rollback()
                    raise

        task = asyncio.create_task(run_answer())
        _retain_answer_task(task)
        try:
            while not task.done() or not queue.empty():
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            try:
                response = await task
                yield f"event: result\ndata: {json.dumps(response.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            except HTTPException as exc:
                payload = {"status_code": exc.status_code, "detail": exc.detail}
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception:
                yield "event: error\ndata: {\"status_code\": 500, \"detail\": \"问答服务异常\"}\n\n"
        finally:
            # The answer owns a persistent turn. Let it reach a terminal state even if
            # the browser disconnects; the same client_turn_id can replay it later.
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
