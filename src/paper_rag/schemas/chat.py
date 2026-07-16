from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SecretStr


class SetApiKeyRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    api_key: SecretStr


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    conversation_id: UUID | None = None
    client_turn_id: str = Field(default_factory=lambda: str(uuid4()), min_length=8, max_length=128)
    question: str = Field(min_length=2, max_length=4000)
    document_id: UUID | None = None
    scope: Literal["all", "selected"] | None = None
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)
    top_n: int = Field(default=8, ge=1, le=20)


class ModelAnswer(BaseModel):
    answer: str
    citation_ids: list[UUID]
    hallucination_risk: Literal["low", "medium", "high"]
    formula_claims: list[str] = Field(default_factory=list)
    novelty_claims: list["NoveltyClaim"] = Field(default_factory=list)
    claims: list["GroundedClaim"] = Field(default_factory=list, max_length=12)
    derivations: list["DeterministicDerivation"] = Field(default_factory=list, max_length=8)
    hypotheses: list["EvidenceBoundedHypothesis"] = Field(default_factory=list, max_length=6)


class GroundedClaim(BaseModel):
    text: str = Field(min_length=2, max_length=1200)
    citation_ids: list[UUID] = Field(min_length=1, max_length=8)
    claim_type: Literal["direct_fact", "synthesized_fact"]
    label: str | None = Field(default=None, max_length=100)


class DeterministicDerivation(BaseModel):
    statement: str = Field(min_length=2, max_length=1200)
    inputs: list[str] = Field(min_length=1, max_length=10)
    operation: str = Field(min_length=1, max_length=500)
    result: str = Field(min_length=1, max_length=300)
    citation_ids: list[UUID] = Field(min_length=1, max_length=8)


class HypothesisPremise(BaseModel):
    claim: str = Field(min_length=2, max_length=1000)
    citation_ids: list[UUID] = Field(min_length=1, max_length=8)


class EvidenceBoundedHypothesis(BaseModel):
    claim: str = Field(min_length=2, max_length=1200)
    premises: list[HypothesisPremise] = Field(min_length=2, max_length=10)
    confidence: Literal["low", "medium", "high"]
    assumptions: list[str] = Field(min_length=1, max_length=10)
    validation_needed: list[str] = Field(min_length=1, max_length=10)
    counterevidence: list[str] = Field(default_factory=list, max_length=10)


class HypothesisGeneration(BaseModel):
    hypotheses: list[EvidenceBoundedHypothesis] = Field(min_length=1, max_length=6)


class HypothesisAudit(BaseModel):
    verdict: Literal["supported_hypothesis", "overreach", "uncertain"]
    reason: str = Field(min_length=1, max_length=1200)
    unsupported_premises: list[str] = Field(default_factory=list, max_length=10)
    missing_conditions: list[str] = Field(default_factory=list, max_length=10)
    counterevidence_ignored: list[str] = Field(default_factory=list, max_length=10)


class NoveltyClaim(BaseModel):
    claim: str = Field(min_length=2, max_length=1000)
    citation_id: UUID
    source_quote: str | None = Field(default=None, min_length=3, max_length=1200)
    claim_strength: Literal["explicit_strong", "explicit", "synthesized"]


class ClaimEntailmentResult(BaseModel):
    claim_index: int = Field(ge=0)
    verdict: Literal["entailed", "partially_entailed", "not_entailed", "uncertain", "audit_unavailable"]
    reason: str = Field(min_length=1, max_length=1000)
    supported_scope: str = Field(default="", max_length=1000)
    unsupported_parts: list[str] = Field(default_factory=list, max_length=20)
    attempt_count: int = Field(default=1, ge=0, le=3)
    error_code: str | None = Field(default=None, max_length=100)
    validation_errors: list[str] = Field(default_factory=list, max_length=10)
    raw_output_sha256: list[str] = Field(default_factory=list, max_length=3)
    latency_ms: int = Field(default=0, ge=0)
    cached: bool = False


class NoveltyEntailmentAudit(BaseModel):
    answer_claims_fully_covered: bool
    uncovered_answer_claims: list[str] = Field(default_factory=list, max_length=20)
    results: list[ClaimEntailmentResult] = Field(min_length=1, max_length=20)


class SingleClaimAudit(BaseModel):
    verdict: Literal["entailed", "partially_entailed", "not_entailed", "uncertain"]
    reason: str = Field(min_length=1, max_length=1000)
    supported_scope: str = Field(default="", max_length=1000)
    unsupported_parts: list[str] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    citation_id: UUID
    document_id: UUID
    paper_title: str
    page_start: int
    page_end: int
    section_path: str | None
    quoted_snippet: str


class FormulaAsset(BaseModel):
    formula_id: UUID
    group_key: str | None
    formula_number: str | None
    page_number: int
    image_url: str
    normalized_text: str | None
    fidelity_status: Literal["source_exact", "needs_review", "unusable"]
    latex_text: str | None = None
    rendered_mathml: str | None = None
    latex_verification_status: Literal["absent", "unverified", "verified", "invalid"] = "absent"
    source_crop_sha256: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    evidence_status: Literal["sufficient", "insufficient"]
    refused: bool
    refusal_reason: str | None
    hallucination_risk: Literal["low", "medium", "high", "unknown"]
    audit_result: str
    action: Literal["answer", "partial", "clarify", "refuse", "error"] = "answer"
    unsupported_parts: list[str] = Field(default_factory=list)
    query_plan: dict | None = None
    entity_links: list[dict] | None = None
    answer_mode: Literal["extract", "synthesize", "compare", "derive", "hypothesize"] = "synthesize"
    epistemic_level: Literal["source_fact", "evidence_synthesis", "deterministic_derivation", "evidence_bounded_hypothesis"] = "evidence_synthesis"
    claim_details: list[dict] = Field(default_factory=list)
    formula_assets: list[FormulaAsset] = Field(default_factory=list)
