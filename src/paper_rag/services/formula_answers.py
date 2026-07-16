from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Collection
from pathlib import Path
from uuid import UUID

import fitz
from latex2mathml.converter import convert as latex_to_mathml
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.schemas.chat import AnswerResponse, Citation, FormulaAsset
from paper_rag.schemas.query_plan import AnswerMode, QueryPlan
from paper_rag.services.query_intent import QueryIntent
from paper_rag.services.retrieval import RetrievedChunk
from paper_rag.services.answer_audit import make_refusal
from paper_rag.services.formula_dependencies import FormulaQueryRoute
from paper_rag.services.formula_query_guard import guard_formula_query, repair_pages_from_evidence


FORMULA_QUERY_STOPWORDS = {
    "formula",
    "equation",
    "what",
    "give",
    "show",
    "directly",
    "please",
}
FORMULA_NUMBER_QUERY = re.compile(r"(?:公式|方程|式|equation)\s*\(?\s*(\d{1,3}[a-z]?)\s*\)?", re.IGNORECASE)
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"


def _formula_query_terms(question: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question)
        if token.lower() not in FORMULA_QUERY_STOPWORDS
    }


def _formula_score(formula: Formula, terms: set[str], requested_number: str | None) -> int:
    haystack = " ".join(
        filter(
            None,
            (
                formula.raw_text,
                formula.normalized_text,
                formula.context_before,
                formula.context_after,
                formula.physical_meaning,
            ),
        )
    ).lower()
    score = sum(3 for term in terms if term in haystack)
    if requested_number and formula.formula_number == requested_number:
        score += 10
    return score


def _valid_bbox_json(value: str) -> bool:
    try:
        bbox = [float(item) for item in json.loads(value)]
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(bbox) == 4
        and all(math.isfinite(item) for item in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _public_fidelity_status(value: str) -> str:
    if value in {"source_exact", "needs_review", "unusable"}:
        return value
    return "unusable"


def _render_verified_mathml(formula: Formula) -> tuple[str | None, str]:
    status = formula.latex_verification_status or "absent"
    if status != "verified" or not formula.latex_text:
        return None, status if status in {"absent", "unverified"} else "invalid"
    latex = formula.latex_text.strip()
    if not latex or len(latex) > 20_000 or any(
        marker in latex.lower() for marker in ("<", ">", "script", "javascript:")
    ):
        return None, "invalid"
    try:
        mathml = latex_to_mathml(latex)
        root = ET.fromstring(mathml)
    except Exception:
        return None, "invalid"
    if root.tag != f"{{{MATHML_NAMESPACE}}}math":
        return None, "invalid"
    if any(
        not element.tag.startswith(f"{{{MATHML_NAMESPACE}}}")
        or any(name.lower().startswith("on") or name.lower().endswith("href") for name in element.attrib)
        for element in root.iter()
    ):
        return None, "invalid"
    return mathml, "verified"


def _source_regions_valid(path: Path, formulas: list[Formula]) -> bool:
    try:
        with fitz.open(path) as pdf:
            for formula in formulas:
                if not 1 <= formula.page_number <= len(pdf):
                    return False
                bbox = tuple(float(item) for item in json.loads(formula.bbox_json))
                clip = fitz.Rect(*bbox) & pdf[formula.page_number - 1].rect
                if clip.is_empty or clip.width <= 1 or clip.height <= 1:
                    return False
    except (TypeError, ValueError, json.JSONDecodeError, fitz.FileDataError):
        return False
    return True


def load_formula_records_for_evidence(
    session: Session,
    evidence: list[RetrievedChunk],
) -> list[Formula]:
    document_ids = list(dict.fromkeys(item.document_id for item in evidence))
    evidence_pages = {
        (item.document_id, page_number)
        for item in evidence
        for page_number in range(item.page_start, item.page_end + 1)
    }
    if not document_ids:
        return []
    formulas = list(
        session.scalars(
            select(Formula)
            .where(Formula.document_id.in_(document_ids))
            .order_by(Formula.document_id, Formula.page_number, Formula.group_key, Formula.part_index)
        )
    )
    return [
        formula
        for formula in formulas
        if (formula.document_id, formula.page_number) in evidence_pages
    ]


def select_relevant_formula_records(
    session: Session,
    question: str,
    evidence: list[RetrievedChunk],
) -> list[Formula]:
    formulas = load_formula_records_for_evidence(session, evidence)
    terms = _formula_query_terms(question)
    number_match = FORMULA_NUMBER_QUERY.search(question)
    requested_number = number_match.group(1).lower() if number_match else None
    scores = {formula.id: _formula_score(formula, terms, requested_number) for formula in formulas}
    best = max(formulas, key=lambda item: scores[item.id], default=None)
    if best is None or scores[best.id] <= 0:
        return []
    selected = [
        formula
        for formula in formulas
        if formula.document_id == best.document_id
        and (
            (best.group_key and formula.group_key == best.group_key)
            or (not best.group_key and formula.id == best.id)
        )
    ]
    selected.sort(key=lambda item: (item.part_index, item.formula_number or ""))
    return selected


def load_formula_records_by_ids(
    session: Session,
    formula_ids: Collection[UUID],
) -> list[Formula]:
    selected = list(
        session.scalars(
            select(Formula).where(
                Formula.id.in_(sorted(set(formula_ids), key=str))
            )
        )
    )
    selected.sort(
        key=lambda item: (
            item.part_index,
            item.formula_number or "",
            str(item.id),
        )
    )
    return selected


def build_direct_formula_response(
    session: Session,
    question: str,
    evidence: list[RetrievedChunk],
    query_plan: QueryPlan,
    *,
    resolved_formula_ids: Collection[UUID] | None = None,
) -> AnswerResponse | None:
    if (
        not evidence
        or query_plan.intent != QueryIntent.FORMULA
        or query_plan.answer_mode != AnswerMode.EXTRACT
    ):
        return None
    if resolved_formula_ids:
        selected = load_formula_records_by_ids(session, resolved_formula_ids)
    else:
        selected = select_relevant_formula_records(session, question, evidence)
    if not selected:
        readiness = guard_formula_query(
            session,
            [],
            FormulaQueryRoute.SOURCE_RENDER,
            repair_pages=repair_pages_from_evidence(evidence),
        )
        return make_refusal(
            readiness.reason,
            readiness.audit_result,
            "refuse",
        )
    best = selected[0]
    readiness = guard_formula_query(
        session,
        selected,
        FormulaQueryRoute.SOURCE_RENDER,
        repair_pages=repair_pages_from_evidence(evidence),
    )
    if not readiness.ready:
        return make_refusal(
            readiness.reason,
            readiness.audit_result,
            "refuse",
        )
    cited_evidence = next(
        (
            item
            for item in evidence
            if item.document_id == best.document_id
            and item.page_start <= best.page_number <= item.page_end
        ),
        None,
    )
    document = session.get(Document, best.document_id)
    if cited_evidence is None or document is None:
        return make_refusal(
            "公式记录缺少可核验的论文或引用片段",
            "formula_text_corrupted",
            "refuse",
        )
    if any(not _valid_bbox_json(item.bbox_json) for item in selected):
        return make_refusal(
            "公式已定位但原始区域坐标损坏，无法可靠还原",
            "formula_text_corrupted",
            "refuse",
        )
    if not Path(document.stored_path).is_file():
        return make_refusal(
            "公式已定位但原始 PDF 不可用，无法可靠还原",
            "formula_text_corrupted",
            "refuse",
        )
    if not _source_regions_valid(Path(document.stored_path), selected):
        return make_refusal(
            "公式已定位但原始区域不在有效 PDF 页面内，无法可靠还原",
            "formula_text_corrupted",
            "refuse",
        )
    citation = Citation(
        citation_id=cited_evidence.chunk_id,
        document_id=cited_evidence.document_id,
        paper_title=document.original_filename,
        page_start=cited_evidence.page_start,
        page_end=cited_evidence.page_end,
        section_path=cited_evidence.section_path,
        quoted_snippet=cited_evidence.content[:600],
    )
    numbers = "、".join(f"({formula.formula_number})" for formula in selected if formula.formula_number)
    rendered = {formula.id: _render_verified_mathml(formula) for formula in selected}
    all_verified = all(rendered[formula.id][0] for formula in selected)
    needs_image = not all_verified and any(
        _public_fidelity_status(formula.fidelity_status) != "source_exact" for formula in selected
    )
    normalized_lines = [
        formula.normalized_text
        for formula in selected
        if formula.normalized_text and _public_fidelity_status(formula.fidelity_status) == "source_exact"
    ]
    answer = (
        f"原文第 {best.page_number} 页给出的公式 {numbers} 如下；主显示为已验证 LaTeX，原 PDF 裁剪保留供审计。"
        if all_verified
        else
        f"已在原文第 {best.page_number} 页定位到公式 {numbers}。"
        "PDF 数学字体编码无法保证纯文本转写准确，以下展示原始公式图像；系统未使用库外知识补写符号。"
        if needs_image
        else f"原文第 {best.page_number} 页给出的公式 {numbers} 如下：\n" + "\n".join(normalized_lines)
    )
    return AnswerResponse(
        answer=answer,
        citations=[citation],
        evidence_status="sufficient",
        refused=False,
        refusal_reason=None,
        hallucination_risk="low",
        audit_result="formula_source_rendered",
        action="answer",
        answer_mode="extract",
        epistemic_level="source_fact",
        claim_details=[
            {
                "type": "formula_source",
                "formula_id": str(formula.id),
                "formula_number": formula.formula_number,
                "citation_ids": [str(citation.citation_id)],
            }
            for formula in selected
        ],
        formula_assets=[
            FormulaAsset(
                formula_id=formula.id,
                group_key=formula.group_key,
                formula_number=formula.formula_number,
                page_number=formula.page_number,
                image_url=f"/api/formulas/{formula.id}/image",
                normalized_text=formula.normalized_text,
                fidelity_status=_public_fidelity_status(formula.fidelity_status),
                latex_text=formula.latex_text if rendered[formula.id][1] == "verified" else None,
                rendered_mathml=rendered[formula.id][0],
                latex_verification_status=rendered[formula.id][1],
                source_crop_sha256=formula.source_crop_sha256,
            )
            for formula in selected
        ],
    )
