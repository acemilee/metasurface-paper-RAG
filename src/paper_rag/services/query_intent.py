from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from paper_rag.services.embeddings import EmbeddingProvider


class QueryIntent(StrEnum):
    OVERVIEW = "overview"
    METHOD = "method_mechanism"
    RESULT = "result_parameter"
    FORMULA = "formula_explanation"
    COMPARISON = "comparison"
    LIMITATION = "limitation_outlook"
    NOVELTY = "novelty_contribution"
    CROSS_DOCUMENT = "cross_document_synthesis"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL = "general"


@dataclass(frozen=True)
class QueryIntentResult:
    intent: QueryIntent
    confidence: float
    margin: float
    source: str
    candidate_intent: QueryIntent | None = None


INTENT_ANCHORS = {
    QueryIntent.OVERVIEW: "概述论文的研究目标、核心内容、方法、贡献和结论。 Summarize the paper objective, scope, method, contribution and conclusion.",
    QueryIntent.METHOD: "解释论文的方法、结构设计、工作原理、机制和实验设置。 Explain the method, structure, design, mechanism and experimental setup.",
    QueryIntent.RESULT: "查询论文报告的结果、性能、参数、数值、趋势和影响。 Ask for reported results, performance, parameters, values, trends and effects.",
    QueryIntent.FORMULA: "解释论文公式、方程、变量和物理含义。 Explain an equation, formula, variables and physical meaning in the paper.",
    QueryIntent.COMPARISON: "比较论文中的两种方法、结构、参数或结果。 Compare methods, structures, parameters or results within a paper.",
    QueryIntent.LIMITATION: "查询论文局限、缺点、未来工作和发展方向。 Ask about limitations, drawbacks, future work and outlook.",
    QueryIntent.NOVELTY: "分析论文的创新点、新颖性、技术亮点和主要贡献。 Analyze novelty, innovations, technical advances and contributions of the paper.",
    QueryIntent.CROSS_DOCUMENT: "综合或比较多篇论文的观点、方法和结果。 Synthesize or compare evidence across multiple papers.",
    QueryIntent.OUT_OF_SCOPE: "与论文内容无关的天气、生活、娱乐、金融、法律、医疗建议或库外事实。 Questions unrelated to the supplied papers or asking for outside facts.",
    QueryIntent.GENERAL: "针对论文内容提出一般事实问题。 A general factual question grounded in the paper.",
}

_anchor_cache: dict[tuple[str, int], tuple[list[QueryIntent], list[list[float]]]] = {}
_anchor_lock = Lock()


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _get_anchor_vectors(provider: EmbeddingProvider) -> tuple[list[QueryIntent], list[list[float]]]:
    cache_key = (provider.model_id, provider.dimension)
    with _anchor_lock:
        cached = _anchor_cache.get(cache_key)
    if cached is not None:
        return cached
    intents = list(INTENT_ANCHORS)
    vectors = provider.embed_documents([INTENT_ANCHORS[intent] for intent in intents])
    with _anchor_lock:
        return _anchor_cache.setdefault(cache_key, (intents, vectors))


def classify_query_intent(question: str, provider: EmbeddingProvider) -> QueryIntentResult:
    question_vector = provider.embed_query(question)
    intents, vectors = _get_anchor_vectors(provider)
    ranked = sorted(
        ((_dot(question_vector, vector), intent) for intent, vector in zip(intents, vectors, strict=True)),
        reverse=True,
        key=lambda item: item[0],
    )
    best_score, best_intent = ranked[0]
    margin = best_score - ranked[1][0]
    if best_score < 0.45 or margin < 0.03:
        return QueryIntentResult(QueryIntent.GENERAL, best_score, margin, "fallback", best_intent)
    if best_intent == QueryIntent.OUT_OF_SCOPE and (best_score < 0.52 or margin < 0.05):
        return QueryIntentResult(QueryIntent.GENERAL, best_score, margin, "fallback", best_intent)
    return QueryIntentResult(best_intent, best_score, margin, "bge_semantic", best_intent)
