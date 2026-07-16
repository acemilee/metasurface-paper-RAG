from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.schemas.chat import AnswerResponse, Citation, DeterministicDerivation, EvidenceBoundedHypothesis, GroundedClaim, ModelAnswer, NoveltyEntailmentAudit
from paper_rag.services.retrieval import RetrievedChunk


NUMERIC_UNIT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ghz|thz|mhz|hz|nm|um|μm|mm|%|v|db|ohm(?:/|=)?sq|[Ωu](?:/|=)?sq)",
    re.I,
)


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    reason: str | None = None


class UnknownCitationError(ValueError):
    pass


class MissingFormulaClaimError(ValueError):
    pass


def map_citation_ids(
    session: Session,
    answer: ModelAnswer,
    evidence: list[RetrievedChunk],
) -> list[Citation]:
    evidence_by_id = {item.chunk_id: item for item in evidence}
    citations: list[Citation] = []
    for citation_id in dict.fromkeys(answer.citation_ids):
        item = evidence_by_id.get(citation_id)
        if item is None:
            raise UnknownCitationError("Model returned an unknown citation ID")
        document = session.get(Document, item.document_id)
        if document is None:
            raise ValueError("Citation document is missing")
        citations.append(
            Citation(
                citation_id=item.chunk_id,
                document_id=item.document_id,
                paper_title=document.original_filename,
                page_start=item.page_start,
                page_end=item.page_end,
                section_path=item.section_path,
                quoted_snippet=item.content[:600],
            )
        )
    return citations


def verify_citations_exist(citations: list[Citation], evidence: list[RetrievedChunk]) -> AuditResult:
    if not citations:
        return AuditResult(False, "回答没有引用任何证据")
    evidence_ids = {item.chunk_id for item in evidence}
    if any(citation.citation_id not in evidence_ids for citation in citations):
        return AuditResult(False, "回答包含未知引用")
    return AuditResult(True)


def verify_cross_document_citations(citations: list[Citation]) -> AuditResult:
    if len({citation.document_id for citation in citations}) < 2:
        return AuditResult(False, "跨论文回答未引用至少两篇论文")
    return AuditResult(True)


def verify_novelty_answer(
    answer: ModelAnswer,
    citations: list[Citation],
    evidence: list[RetrievedChunk],
    document_genres: list[str] | None = None,
) -> AuditResult:
    if not answer.novelty_claims:
        return AuditResult(False, "创新点回答未输出逐条主张与原文引用")
    citation_ids = {citation.citation_id for citation in citations}
    evidence_by_id = {item.chunk_id: item for item in evidence}
    claim_citation_ids = {claim.citation_id for claim in answer.novelty_claims}
    if not claim_citation_ids.issubset(citation_ids):
        return AuditResult(False, "创新主张引用未包含在答案 citation_ids 中")
    for claim in answer.novelty_claims:
        item = evidence_by_id.get(claim.citation_id)
        if item is None:
            return AuditResult(False, "创新主张引用了未知证据")
    return AuditResult(True)


def verify_novelty_entailment_audit(
    audit: NoveltyEntailmentAudit,
    claim_count: int,
) -> AuditResult:
    if not audit.answer_claims_fully_covered:
        detail = "、".join(audit.uncovered_answer_claims) or "存在未登记主张"
        return AuditResult(False, f"创新答案正文含未登记的创新主张：{detail}")
    by_index = {item.claim_index: item for item in audit.results}
    if len(by_index) != len(audit.results):
        return AuditResult(False, "创新语义审计包含重复的主张索引")
    if set(by_index) != set(range(claim_count)):
        return AuditResult(False, "创新语义审计未覆盖全部主张")
    failures = []
    for index in range(claim_count):
        item = by_index[index]
        if item.verdict != "entailed":
            detail = item.reason
            if item.unsupported_parts:
                detail += f"；不受支持部分：{'、'.join(item.unsupported_parts)}"
            failures.append(f"主张{index + 1}: {item.verdict} - {detail}")
    if failures:
        return AuditResult(False, "创新主张未被原文完整蕴含：" + " | ".join(failures))
    return AuditResult(True)


def verify_claim_tokens_against_evidence(
    answer: ModelAnswer,
    citations: list[Citation],
    evidence: list[RetrievedChunk],
) -> AuditResult:
    citation_ids = {item.citation_id for item in citations}
    cited_text = " ".join(
        item.content for item in evidence if item.chunk_id in citation_ids
    )
    direction_audit = _verify_bias_resistance_direction(answer.answer, cited_text)
    if not direction_audit.passed:
        return direction_audit
    anchors = _numeric_anchors(answer.answer)
    if not anchors:
        return AuditResult(True)
    evidence_anchors = _numeric_anchors(cited_text)
    missing = sorted(
        anchor for anchor in anchors - evidence_anchors
        if not _is_derived_difference(anchor, evidence_anchors)
    )
    if missing:
        return AuditResult(False, "回答中的数值或单位未被引用证据支持")
    return AuditResult(True)


def _is_derived_difference(anchor: str, evidence_anchors: set[str]) -> bool:
    match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[a-z/%]+)", anchor)
    if match is None:
        return False
    target = float(match.group("value"))
    unit = match.group("unit")
    values = []
    for candidate in evidence_anchors:
        candidate_match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[a-z/%]+)", candidate)
        if candidate_match and candidate_match.group("unit") == unit:
            values.append(float(candidate_match.group("value")))
    return any(
        abs(abs(left - right) - target) <= max(1e-6, abs(target) * 1e-4)
        for index, left in enumerate(values)
        for right in values[index + 1:]
    )


def _numeric_anchors(text: str) -> set[str]:
    text = re.sub(
        r"(?P<whole>\d+):(?P<fraction>\d+)(?=\s*(?:ghz|mhz|hz|nm|um|μm|mm|%|v|db|ohm|[Ωu]))",
        r"\g<whole>.\g<fraction>",
        text,
        flags=re.I,
    )
    anchors: set[str] = set()
    for match in NUMERIC_UNIT.finditer(text):
        value = match.group("value")
        unit = match.group("unit").lower().replace("μ", "u").replace("=", "/")
        if unit.startswith(("ω", "u", "ohm")) and unit.endswith("sq"):
            unit = "ohm/sq"
        anchors.add(f"{value}{unit}")
    return anchors


def _verify_bias_resistance_direction(answer: str, evidence_text: str) -> AuditResult:
    normalized_answer = answer.lower().replace("ω", "ohm").replace("Ω", "ohm")
    increasing_claim = bool(re.search(
        r"(?:rg|sheet resistance).{0,50}increas(?:e|es|ed|ing)?.{0,50}bias voltage",
        normalized_answer,
    ))
    decreasing_claim = bool(re.search(
        r"(?:rg|sheet resistance).{0,50}decreas(?:e|es|ed|ing)?.{0,50}bias voltage",
        normalized_answer,
    ))
    if not increasing_claim and not decreasing_claim:
        return AuditResult(True)
    pairs = [
        (float(voltage), float(resistance))
        for voltage, resistance in re.findall(
            r"(\d+(?:\.\d+)?)\s*v\s*\(\s*(\d+(?:\.\d+)?)\s*(?:u|ohm|Ω)\s*[=/]?\s*sq",
            evidence_text,
            flags=re.I,
        )
    ]
    if len(pairs) < 2:
        return AuditResult(True)
    pairs.sort()
    observed_increasing = pairs[-1][1] > pairs[0][1]
    if increasing_claim != observed_increasing:
        return AuditResult(False, "回答中的偏压与石墨烯片阻变化方向和引用证据冲突")
    return AuditResult(True)


def verify_formula_claims(
    session: Session,
    answer: ModelAnswer,
    citations: list[Citation],
    evidence: list[RetrievedChunk],
) -> AuditResult:
    if not answer.formula_claims:
        return AuditResult(True)
    citation_ids = {item.citation_id for item in citations}
    formula_ids = {
        UUID(formula_id)
        for item in evidence
        if item.chunk_id in citation_ids
        for formula_id in item.formula_ids
    }
    if not formula_ids:
        return AuditResult(False, "公式结论没有引用公式证据")
    statuses = list(
        session.scalars(select(Formula.semantic_status).where(Formula.id.in_(formula_ids)))
    )
    if not any(status == "grounded" for status in statuses):
        return AuditResult(False, "公式结论引用的物理含义未经原文证据确认")
    return AuditResult(True)


def render_novelty_claims(answer: ModelAnswer) -> str:
    return "\n".join(
        f"{index}. {claim.claim.rstrip('。')}。"
        for index, claim in enumerate(answer.novelty_claims, start=1)
    )


def render_grounded_claims(claims: list[GroundedClaim]) -> str:
    return "\n".join(
        f"{index}. {f'{claim.label}：' if claim.label else ''}{claim.text.rstrip('。')}。"
        for index, claim in enumerate(claims, start=1)
    )


def render_derivations(derivations: list[DeterministicDerivation]) -> str:
    blocks = []
    for index, item in enumerate(derivations, start=1):
        inputs = "；".join(item.inputs)
        blocks.append(
            f"{index}. {item.statement.rstrip('。')}。\n"
            f"   已知：{inputs}\n"
            f"   计算：{item.operation}\n"
            f"   结果：{item.result}"
        )
    return "\n".join(blocks)


def render_hypotheses(hypotheses: list[EvidenceBoundedHypothesis]) -> str:
    blocks = ["以下是基于库内证据的推测，不是论文已经验证的结论。"]
    for index, item in enumerate(hypotheses, start=1):
        premises = "；".join(premise.claim for premise in item.premises)
        assumptions = "；".join(item.assumptions)
        validation = "；".join(item.validation_needed)
        counterevidence = "；".join(item.counterevidence) or "当前召回证据未提供明确反证"
        blocks.append(
            f"{index}. 条件性推测：{item.claim.rstrip('。')}。\n"
            f"   库内前提：{premises}\n"
            f"   置信度：{item.confidence}\n"
            f"   关键假设：{assumptions}\n"
            f"   风险或反证：{counterevidence}\n"
            f"   建议验证：{validation}"
        )
    return "\n".join(blocks)


def salvage_partially_entailed_premises(
    hypothesis: EvidenceBoundedHypothesis,
    report: NoveltyEntailmentAudit,
) -> EvidenceBoundedHypothesis | None:
    results = {item.claim_index: item for item in report.results}
    if len(results) != len(hypothesis.premises):
        return None
    premises = []
    for index, premise in enumerate(hypothesis.premises):
        result = results.get(index)
        if result is None:
            return None
        if result.verdict == "entailed":
            premises.append(premise)
        elif result.verdict == "partially_entailed" and result.supported_scope.strip():
            premises.append(premise.model_copy(update={"claim": result.supported_scope.strip()}))
        else:
            return None
    return hypothesis.model_copy(update={"premises": premises})


def make_refusal(
    reason: str,
    audit_result: str = "refused_before_generation",
    action: str = "refuse",
) -> AnswerResponse:
    normalized_reason = reason.rstrip("。！？!? ")
    return AnswerResponse(
        answer=f"{normalized_reason}。系统不会使用库外知识补全答案。",
        citations=[],
        evidence_status="insufficient",
        refused=True,
        refusal_reason=reason,
        hallucination_risk="unknown",
        audit_result=audit_result,
        action=action,
    )


def make_hypothesis_refusal(unsupported_parts: list[str]) -> AnswerResponse:
    reason = "没有通过前提与推理审计的证据约束假设"
    response = make_refusal(reason, "hypothesis_not_supported", "refuse")
    response.answer_mode = "hypothesize"
    response.epistemic_level = "evidence_bounded_hypothesis"
    response.unsupported_parts = unsupported_parts
    return response
