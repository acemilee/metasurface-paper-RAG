from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from paper_rag.services.query_intent import QueryIntent


class EvidenceType(StrEnum):
    GENERAL = "general"
    OVERVIEW = "overview"
    PROBLEM_OR_GAP = "problem_or_gap"
    NOVELTY_CLAIM = "novelty_claim"
    METHOD_OR_STRUCTURE = "method_or_structure"
    RESULT_OR_ADVANTAGE = "result_or_advantage"
    COMPARISON_BASELINE = "comparison_baseline"
    LIMITATION = "limitation"
    FORMULA_CONTEXT = "formula_context"
    SYNTHESIS_OR_TAXONOMY = "synthesis_or_taxonomy"
    TREND_OR_OUTLOOK = "trend_or_outlook"
    EXPLICIT_INNOVATION = "explicit_innovation"
    CHAPTER_RESULT = "chapter_result"
    EXPERIMENT = "experiment"
    CONCLUSION = "conclusion"
    OPERATING_CONDITIONS = "operating_conditions"
    COUNTEREVIDENCE = "counterevidence"
    PREMISE_FOR_MATERIAL = "premise_for_material"
    PREMISE_FOR_STRUCTURE = "premise_for_structure"


class AnswerMode(StrEnum):
    EXTRACT = "extract"
    SYNTHESIZE = "synthesize"
    COMPARE = "compare"
    DERIVE = "derive"
    HYPOTHESIZE = "hypothesize"


class EntityType(StrEnum):
    DOCUMENT_REFERENCE = "document_reference"
    FORMULA = "formula"
    MATERIAL = "material"
    METHOD = "method"
    PARAMETER = "parameter"
    FIGURE_OR_TABLE = "figure_or_table"
    OTHER = "other"


class ScopeRequirement(StrEnum):
    CURRENT_SCOPE = "current_scope"
    SINGLE_DOCUMENT = "single_document"
    MULTIPLE_DOCUMENTS = "multiple_documents"
    ALL_DOCUMENTS = "all_documents"


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    evidence_type: EvidenceType


class QueryEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: str = Field(min_length=1, max_length=200)
    canonical: str | None = Field(default=None, max_length=200)
    entity_type: EntityType
    must_link: bool = False


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: QueryIntent
    answer_mode: AnswerMode = AnswerMode.SYNTHESIZE
    standalone_question: str = Field(min_length=2, max_length=1000)
    retrieval_queries: list[RetrievalQuery] = Field(min_length=1, max_length=8)
    entities: list[QueryEntity] = Field(default_factory=list, max_length=12)
    required_evidence: list[EvidenceType] = Field(default_factory=list, max_length=6)
    scope_requirement: ScopeRequirement = ScopeRequirement.CURRENT_SCOPE
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
