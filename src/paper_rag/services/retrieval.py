from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from uuid import UUID

from paper_rag.config import Settings
from paper_rag.services.embeddings import EmbeddingProvider
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult
from paper_rag.schemas.query_plan import AnswerMode, EvidenceType, QueryPlan


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    page_start: int
    page_end: int
    section_path: str | None
    formula_ids: list[str]
    score: float
    quality_score: float = 1.0
    has_low_confidence_ocr: bool = False
    retrieval_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceDecision:
    sufficient: bool
    reason: str | None


STOPWORDS = {
    "a", "an", "and", "are", "at", "authors", "did", "does", "for", "how",
    "in", "is", "of", "paper", "the", "their", "this", "to", "what", "which",
    "论文", "研究", "结果", "作者", "本文", "该论", "这篇", "主要", "什么",
    "如何", "怎么", "内容", "工作", "问题", "进行", "通过", "解释",
}

OVERVIEW_QUERY = "论文摘要 研究主题 主要内容 研究范围 分类 关键进展 abstract overview"
CONCLUSION_QUERY = "论文结论 主要贡献 局限 发展趋势 总结 conclusion contribution outlook"
METHOD_QUERY = "研究方法 结构设计 工作原理 物理机制 实验设置 method structure design mechanism setup"
RESULT_QUERY = "主要结果 性能参数 实验数据 趋势 影响 result performance parameter measurement trend"
FORMULA_QUERY = "公式 方程 变量定义 物理含义 equation formula variable physical meaning"
LIMITATION_QUERY = "局限 不足 挑战 未来工作 发展方向 limitation drawback challenge future work outlook"

FACTUAL_EVIDENCE_ROLES = {
    EvidenceType.OVERVIEW.value,
    EvidenceType.NOVELTY_CLAIM.value,
    EvidenceType.METHOD_OR_STRUCTURE.value,
    EvidenceType.RESULT_OR_ADVANTAGE.value,
    EvidenceType.FORMULA_CONTEXT.value,
    EvidenceType.EXPERIMENT.value,
    EvidenceType.CONCLUSION.value,
    EvidenceType.OPERATING_CONDITIONS.value,
    EvidenceType.PREMISE_FOR_MATERIAL.value,
    EvidenceType.PREMISE_FOR_STRUCTURE.value,
}


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?", text.lower()))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.add(sequence)
        else:
            tokens.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return tokens


def _numeric_unit_anchors(question: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?\s*(?:ghz|thz|mhz|hz|nm|um|μm|mm|%|v|ohm|db)", question.lower()))


def _is_reference_dense(content: str, section_path: str | None = None) -> bool:
    if section_path and re.search(r"\b(?:references|bibliography)\b|参考文献", section_path, re.IGNORECASE):
        return True
    numbered_entries = len(re.findall(r"(?:^|\n)\s*\[\d+\]", content))
    if numbered_entries >= 5:
        return True
    marker = re.search(r"(?:^|\n)\s*(?:references|bibliography|参考文献)\s*(?:\n|$)", content, re.IGNORECASE)
    if not marker:
        return False
    reference_tail = content[marker.end():]
    tail_entries = len(re.findall(r"(?:^|\n)\s*\[?\d+\]?[。.\s]", reference_tail))
    return tail_entries >= 5


def _document_where(document_scope: UUID | list[UUID] | None):
    if isinstance(document_scope, UUID):
        return {"document_id": str(document_scope)}
    if document_scope:
        return {"document_id": {"$in": [str(item) for item in document_scope]}}
    return None


def retrieve_candidates(collection, provider: EmbeddingProvider, question: str, top_n: int, document_id: UUID | list[UUID] | None = None, retrieval_role: str | None = None) -> list[RetrievedChunk]:
    where = _document_where(document_id)
    result = collection.query(query_embeddings=[provider.embed_query(question)], n_results=max(top_n * 3, top_n), where=where, include=["documents", "metadatas", "distances"])
    candidates: list[RetrievedChunk] = []
    for content, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        metric = (collection.metadata or {}).get("hnsw:space", "l2")
        similarity = 1.0 - float(distance) if metric == "cosine" else 1.0 - (float(distance) / 2.0)
        candidates.append(RetrievedChunk(UUID(metadata["chunk_id"]), UUID(metadata["document_id"]), content, int(metadata["page_start"]), int(metadata["page_end"]), metadata["section_path"] or None, json.loads(metadata["formula_ids"]), similarity, float(metadata.get("quality_score", 1.0)), bool(metadata.get("has_low_confidence_ocr", False)), (retrieval_role,) if retrieval_role else ()))
    query_tokens = _tokens(question)
    def score(candidate: RetrievedChunk) -> float:
        overlap = len(query_tokens & _tokens(candidate.content))
        reference_penalty = (
            1.0
            if retrieval_role in FACTUAL_EVIDENCE_ROLES
            and _is_reference_dense(candidate.content, candidate.section_path)
            else 0.0
        )
        return candidate.score + min(0.5, overlap * 0.08) - reference_penalty
    ranked = sorted(candidates, key=score, reverse=True)
    return [
        candidate
        for candidate in ranked
        if not _is_reference_dense(candidate.content, candidate.section_path)
    ][:top_n]


def retrieve_question_evidence(
    collection,
    provider: EmbeddingProvider,
    question: str,
    top_n: int,
    document_id: UUID | list[UUID] | None = None,
    intent_result: QueryIntentResult | None = None,
) -> list[RetrievedChunk]:
    intent = intent_result.intent if intent_result else QueryIntent.GENERAL
    if intent == QueryIntent.CROSS_DOCUMENT and isinstance(document_id, list) and len(document_id) > 1:
        per_document = max(2, (top_n + len(document_id) - 1) // len(document_id))
        ranked_sets = [
            retrieve_candidates(collection, provider, question, per_document, current_document)
            for current_document in document_id
        ]
        return _round_robin_unique(ranked_sets, top_n)
    expansion = {
        QueryIntent.OVERVIEW: (OVERVIEW_QUERY, CONCLUSION_QUERY),
        QueryIntent.METHOD: (question, METHOD_QUERY),
        QueryIntent.RESULT: (question, RESULT_QUERY),
        QueryIntent.FORMULA: (question, FORMULA_QUERY),
        QueryIntent.LIMITATION: (question, LIMITATION_QUERY),
    }.get(intent)
    if expansion is None:
        return retrieve_candidates(collection, provider, question, top_n, document_id)
    per_query = max(3, (top_n + 1) // 2)
    ranked_sets = [retrieve_candidates(collection, provider, query, per_query, document_id) for query in expansion]
    return _round_robin_unique(ranked_sets, top_n)


def _round_robin_unique(ranked_sets: list[list[RetrievedChunk]], top_n: int) -> list[RetrievedChunk]:
    deduplicated = []
    seen: dict[UUID, int] = {}
    max_rank = max((len(candidates) for candidates in ranked_sets), default=0)
    for rank in range(max_rank):
        for candidates in ranked_sets:
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            if candidate.chunk_id in seen:
                existing_index = seen[candidate.chunk_id]
                existing = deduplicated[existing_index]
                merged_roles = tuple(dict.fromkeys((*existing.retrieval_roles, *candidate.retrieval_roles)))
                deduplicated[existing_index] = replace(existing, retrieval_roles=merged_roles)
                continue
            seen[candidate.chunk_id] = len(deduplicated)
            deduplicated.append(candidate)
            if len(deduplicated) >= top_n:
                return deduplicated
    return deduplicated[:top_n]


def retrieve_planned_evidence(
    collection,
    provider: EmbeddingProvider,
    plan: QueryPlan,
    top_n: int,
    document_scope: list[UUID],
    profile_hints: list[tuple[str, str]] | None = None,
) -> list[RetrievedChunk]:
    query_specs = [(plan.standalone_question, "primary")]
    query_specs.extend((item.query, item.evidence_type.value) for item in plan.retrieval_queries)
    query_specs.extend(profile_hints or [])
    per_query = max(2, (top_n + len(query_specs) - 1) // len(query_specs))
    ranked_sets = []
    if plan.intent == QueryIntent.CROSS_DOCUMENT and len(document_scope) > 1:
        for query, role in query_specs:
            for document_id in document_scope:
                ranked_sets.append(
                    retrieve_candidates(collection, provider, query, per_query, document_id, role)
                )
    else:
        for query, role in query_specs:
            ranked_sets.append(
                retrieve_candidates(collection, provider, query, per_query, document_scope, role)
            )
    ranked = _round_robin_unique(ranked_sets, top_n)
    if plan.answer_mode == AnswerMode.HYPOTHESIZE:
        ranked = [
            candidate
            for candidate in ranked
            if not _is_reference_dense(candidate.content, candidate.section_path)
        ]
    return ranked


def has_sufficient_retrieval_evidence(question: str, candidates: list[RetrievedChunk], settings: Settings, intent_result: QueryIntentResult | None = None, query_plan: QueryPlan | None = None, document_genres: list[str] | None = None) -> EvidenceDecision:
    if not candidates:
        return EvidenceDecision(False, "未检索到库内证据")
    intent = intent_result.intent if intent_result else QueryIntent.GENERAL
    if intent == QueryIntent.OUT_OF_SCOPE:
        return EvidenceDecision(False, "问题意图超出当前论文库范围")
    if query_plan and query_plan.answer_mode == AnswerMode.HYPOTHESIZE:
        reliable_roles = {
            role
            for candidate in candidates
            if candidate.score >= settings.retrieval_score_floor
            and not _is_reference_dense(candidate.content, candidate.section_path)
            for role in candidate.retrieval_roles
        }
        premise_roles = {
            EvidenceType.PREMISE_FOR_MATERIAL.value,
            EvidenceType.PREMISE_FOR_STRUCTURE.value,
        }
        if not premise_roles.issubset(reliable_roles):
            return EvidenceDecision(False, "证据不足以同时建立材料与结构推理前提")
        if EvidenceType.COUNTEREVIDENCE.value not in reliable_roles:
            return EvidenceDecision(False, "尚未检索到可用于约束推理的限制或反证证据")
        return EvidenceDecision(True, None)
    if intent == QueryIntent.CROSS_DOCUMENT and len({candidate.document_id for candidate in candidates}) < 2:
        return EvidenceDecision(False, "跨论文综合未召回至少两篇论文的证据")
    if intent == QueryIntent.OVERVIEW:
        if len({candidate.document_id for candidate in candidates}) != 1:
            return EvidenceDecision(False, "概述问题需要明确指定一篇论文")
        if len(candidates) < 2:
            return EvidenceDecision(False, "论文可用于概述的摘要或结论证据不足")
        return EvidenceDecision(True, None)
    if intent == QueryIntent.NOVELTY:
        reliable_roles = {
            role
            for candidate in candidates
            if candidate.score >= settings.retrieval_score_floor
            for role in candidate.retrieval_roles
        }
        genre = document_genres[0] if document_genres and len(set(document_genres)) == 1 else "research_paper"
        if genre == "review_paper":
            novelty_roles = {EvidenceType.SYNTHESIS_OR_TAXONOMY.value}
            supporting_roles = {EvidenceType.COMPARISON_BASELINE.value, EvidenceType.TREND_OR_OUTLOOK.value}
        elif genre == "thesis":
            novelty_roles = {EvidenceType.EXPLICIT_INNOVATION.value}
            supporting_roles = {EvidenceType.METHOD_OR_STRUCTURE.value, EvidenceType.CHAPTER_RESULT.value, EvidenceType.RESULT_OR_ADVANTAGE.value}
        else:
            novelty_roles = {EvidenceType.NOVELTY_CLAIM.value, EvidenceType.METHOD_OR_STRUCTURE.value}
            supporting_roles = {EvidenceType.PROBLEM_OR_GAP.value, EvidenceType.RESULT_OR_ADVANTAGE.value, EvidenceType.COMPARISON_BASELINE.value}
        if not reliable_roles & novelty_roles or not reliable_roles & supporting_roles:
            return EvidenceDecision(False, "创新点问题缺少新方法声明及其问题背景或结果优势的组合证据")
        if query_plan and query_plan.required_evidence:
            required = {item.value for item in query_plan.required_evidence}
            if len(required & reliable_roles) < min(2, len(required)):
                return EvidenceDecision(False, "创新点问题未覆盖 QueryPlan 要求的多类证据")
        return EvidenceDecision(True, None)
    best_score = max(candidate.score for candidate in candidates)
    informative_tokens = _tokens(question) - STOPWORDS
    lexical_support = max(
        (len(informative_tokens & _tokens(candidate.content)) for candidate in candidates),
        default=0,
    )
    domain_role_support = intent in {QueryIntent.METHOD, QueryIntent.RESULT} and best_score >= settings.retrieval_score_floor
    if not domain_role_support and best_score < settings.retrieval_min_score and not (
        best_score >= settings.retrieval_score_floor
        and lexical_support >= settings.retrieval_lexical_min_terms
    ):
        return EvidenceDecision(False, "已召回候选证据，但相关性不足以支持可靠回答")
    anchors = _numeric_unit_anchors(question)
    if anchors:
        evidence_text = " ".join(candidate.content.lower() for candidate in candidates)
        if not all(anchor.replace(" ", "") in evidence_text.replace(" ", "") for anchor in anchors):
            return EvidenceDecision(False, "问题中的数值或单位未出现在检索证据中")
        reliable_text = " ".join(
            candidate.content.lower()
            for candidate in candidates
            if not candidate.has_low_confidence_ocr
            and candidate.quality_score >= settings.ocr_numeric_min_confidence
        )
        if not all(
            anchor.replace(" ", "") in reliable_text.replace(" ", "")
            for anchor in anchors
        ):
            return EvidenceDecision(False, "精确数值仅见于低置信 OCR 证据")
    return EvidenceDecision(True, None)
