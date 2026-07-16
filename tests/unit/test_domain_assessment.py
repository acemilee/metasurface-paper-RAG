from datetime import datetime, timezone

from sqlalchemy import select

from paper_rag.models.document import DomainStatus
from paper_rag.models.domain_admission import DomainAssessment
from paper_rag.services.domain_admission import DomainAdmissionResult
from paper_rag.services.domain_assessment import (
    approve_domain_assessment,
    record_domain_assessment,
)


def make_result(
    decision: DomainStatus, decision_code: str
) -> DomainAdmissionResult:
    return DomainAdmissionResult(
        decision=decision,
        decision_code=decision_code,
        evidence_regions=(),
        passed_requirements=(),
        failed_requirements=(
            ()
            if decision == DomainStatus.ACCEPTED
            else ("domain_relationship",)
        ),
        parse_quality=1.0,
        classifier_version="positive-admission-v2",
        embedding_model_id="assessment-test",
        config_fingerprint="f" * 64,
        duration_ms=12,
        evaluated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )


def accepted_result() -> DomainAdmissionResult:
    return make_result(DomainStatus.ACCEPTED, "positive_evidence_quorum")


def review_result() -> DomainAdmissionResult:
    return make_result(
        DomainStatus.REVIEW_REQUIRED, "missing_domain_relationship"
    )


def test_domain_assessments_are_append_only(session, document) -> None:
    first = record_domain_assessment(
        session,
        document,
        accepted_result(),
        trigger="upload",
        applied=True,
    )
    second = record_domain_assessment(
        session,
        document,
        review_result(),
        trigger="deletion_check",
        applied=False,
    )

    rows = {row.id: row for row in session.scalars(select(DomainAssessment))}
    assert set(rows) == {first.id, second.id}
    assert rows[first.id].decision == "accepted"
    assert rows[second.id].decision == "review_required"
    assert document.domain_status == DomainStatus.ACCEPTED


def test_manual_approval_records_linked_override(session, document) -> None:
    assessment = record_domain_assessment(
        session,
        document,
        review_result(),
        trigger="upload",
        applied=True,
    )

    override = approve_domain_assessment(
        session,
        document,
        assessment.id,
        actor="local_user",
    )

    assert override.assessment_id == assessment.id
    assert document.domain_status == DomainStatus.MANUAL_APPROVED
    assert document.domain_manual_override_at is not None
    assert assessment.decision == DomainStatus.REVIEW_REQUIRED
