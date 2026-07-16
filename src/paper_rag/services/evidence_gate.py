from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.config import Settings
from paper_rag.models.formula import Formula
from paper_rag.services.retrieval import EvidenceDecision, RetrievedChunk, has_sufficient_retrieval_evidence
from paper_rag.services.query_intent import QueryIntent, QueryIntentResult
from paper_rag.schemas.query_plan import AnswerMode, QueryPlan


FORMULA_TERMS = ("formula", "equation", "公式", "方程", "表达式", "物理含义")


def must_refuse_for_formula_claim(
    session: Session,
    question: str,
    evidence: list[RetrievedChunk],
    force: bool = False,
) -> bool:
    if not force and not any(term in question.lower() for term in FORMULA_TERMS):
        return False
    formula_ids = {UUID(formula_id) for item in evidence for formula_id in item.formula_ids}
    if not formula_ids:
        return True
    statuses = list(
        session.scalars(
            select(Formula.semantic_status).where(Formula.id.in_(formula_ids))
        )
    )
    return not statuses or not any(status == "grounded" for status in statuses)


def evaluate_evidence(
    session: Session,
    question: str,
    evidence: list[RetrievedChunk],
    settings: Settings,
    intent_result: QueryIntentResult | None = None,
    query_plan: QueryPlan | None = None,
    document_genres: list[str] | None = None,
) -> EvidenceDecision:
    decision = has_sufficient_retrieval_evidence(question, evidence, settings, intent_result, query_plan, document_genres)
    if not decision.sufficient:
        return decision
    direct_formula_extract = bool(
        query_plan
        and query_plan.intent == QueryIntent.FORMULA
        and query_plan.answer_mode == AnswerMode.EXTRACT
    )
    if not direct_formula_extract and (
        (intent_result and intent_result.intent == QueryIntent.FORMULA)
        or any(term in question.lower() for term in FORMULA_TERMS)
    ):
        if must_refuse_for_formula_claim(session, question, evidence, force=True):
            return EvidenceDecision(False, "检索到的公式缺少可验证物理含义")
    return EvidenceDecision(True, None)
