from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from paper_rag.config import Settings
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus, DomainStatus
from paper_rag.models.domain_admission import (
    DomainAssessment,
    DomainManualOverride,
)
from paper_rag.models.job import JobState
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.services.domain_admission import (
    CLASSIFIER_VERSION,
    DomainAdmissionResult,
)
from paper_rag.services.vector_store import delete_document_vectors


class DomainAssessmentConflict(RuntimeError):
    pass


class DomainArtifactCleanupError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionApplication:
    assessment_id: UUID
    may_index: bool
    applied: bool
    terminal_job_state: JobState | None


def admission_application_mode(
    document: Document,
) -> Literal["enforce", "shadow"]:
    return (
        "enforce"
        if document.domain_enforcement_version == CLASSIFIER_VERSION
        else "shadow"
    )


def _serialized_evidence(result: DomainAdmissionResult) -> str:
    evidence = [
        {
            "region_id": region.region_id,
            "page_numbers": list(region.page_numbers),
            "section_role": region.section_role,
            "concept_families": list(region.concept_families),
            "relations": list(region.relations),
            "semantic_support": region.semantic_support,
            "content_hash": region.content_hash,
            "excerpt": region.excerpt[:240],
        }
        for region in result.evidence_regions
    ]
    return json.dumps(evidence, ensure_ascii=False)


def record_domain_assessment(
    session: Session,
    document: Document,
    result: DomainAdmissionResult,
    *,
    trigger: str,
    applied: bool,
) -> DomainAssessment:
    assessment = DomainAssessment(
        document_id=document.id,
        trigger=trigger,
        decision=result.decision.value,
        decision_code=result.decision_code,
        classifier_version=result.classifier_version,
        embedding_model_id=result.embedding_model_id,
        config_fingerprint=result.config_fingerprint,
        evidence_json=_serialized_evidence(result),
        passed_requirements_json=json.dumps(list(result.passed_requirements)),
        failed_requirements_json=json.dumps(list(result.failed_requirements)),
        parse_quality=result.parse_quality,
        duration_ms=result.duration_ms,
        applied_to_document=applied,
        created_at=result.evaluated_at,
    )
    session.add(assessment)
    if applied:
        document.domain_status = result.decision
        document.domain_decision_code = result.decision_code
        document.domain_reasons_json = json.dumps(
            list(result.failed_requirements or result.passed_requirements),
            ensure_ascii=False,
        )
        document.domain_classifier_version = result.classifier_version
        document.domain_checked_at = result.evaluated_at
    session.flush()
    return assessment


def apply_domain_assessment(
    session: Session,
    document: Document,
    result: DomainAdmissionResult,
    *,
    trigger: str,
) -> AdmissionApplication:
    mode = admission_application_mode(document)
    applied = (
        mode == "enforce"
        and document.domain_status != DomainStatus.MANUAL_APPROVED
    )
    assessment = record_domain_assessment(
        session,
        document,
        result,
        trigger=trigger,
        applied=applied,
    )
    if document.domain_status == DomainStatus.MANUAL_APPROVED:
        return AdmissionApplication(assessment.id, True, False, None)
    if mode == "shadow":
        return AdmissionApplication(
            assessment.id,
            document.domain_status
            in {DomainStatus.ACCEPTED, DomainStatus.MANUAL_APPROVED},
            False,
            None,
        )
    if result.decision != DomainStatus.ACCEPTED:
        document.status = DocumentStatus.REVIEW_REQUIRED
        return AdmissionApplication(
            assessment.id,
            False,
            True,
            JobState.REVIEW_REQUIRED,
        )
    return AdmissionApplication(assessment.id, True, True, None)


def clear_knowledge_artifacts_for_review(
    session: Session,
    document: Document,
    collection,
    settings: Settings | None = None,
) -> None:
    chunks = list(
        session.scalars(select(Chunk).where(Chunk.document_id == document.id))
    )
    delete_document_vectors(collection, chunks, settings)
    session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    profiles = list(
        session.scalars(
            select(PaperProfile).where(PaperProfile.document_id == document.id)
        )
    )
    for profile in profiles:
        session.delete(profile)
    session.flush()
    chunk_count = session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == document.id)
    ) or 0
    profile_count = session.scalar(
        select(func.count(PaperProfile.id)).where(
            PaperProfile.document_id == document.id
        )
    ) or 0
    vector_count = len(
        collection.get(
            where={"document_id": str(document.id)}, include=[]
        )["ids"]
    )
    if chunk_count or profile_count or vector_count:
        raise DomainArtifactCleanupError(
            "review cleanup failed: "
            f"chunks={chunk_count} vectors={vector_count} profiles={profile_count}"
        )


def approve_domain_assessment(
    session: Session,
    document: Document,
    assessment_id: UUID,
    *,
    actor: str,
) -> DomainManualOverride:
    assessment = session.get(DomainAssessment, assessment_id)
    if assessment is None or assessment.document_id != document.id:
        raise DomainAssessmentConflict("assessment does not belong to document")
    if (
        not assessment.applied_to_document
        or assessment.decision != DomainStatus.REVIEW_REQUIRED
    ):
        raise DomainAssessmentConflict("assessment is not the active review decision")
    latest_applied_id = session.scalar(
        select(DomainAssessment.id)
        .where(
            DomainAssessment.document_id == document.id,
            DomainAssessment.applied_to_document.is_(True),
        )
        .order_by(
            DomainAssessment.created_at.desc(), DomainAssessment.id.desc()
        )
        .limit(1)
    )
    if latest_applied_id != assessment.id:
        raise DomainAssessmentConflict("assessment is stale")
    if document.domain_status != DomainStatus.REVIEW_REQUIRED:
        raise DomainAssessmentConflict("document does not require approval")
    override = DomainManualOverride(
        document_id=document.id,
        assessment_id=assessment.id,
        action="approve",
        actor=actor,
    )
    session.add(override)
    document.domain_status = DomainStatus.MANUAL_APPROVED
    document.domain_manual_override_at = datetime.now().astimezone()
    session.flush()
    return override
