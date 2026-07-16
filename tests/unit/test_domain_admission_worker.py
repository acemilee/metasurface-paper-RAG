from datetime import datetime, timezone

from sqlalchemy import func, select

from paper_rag.models.chunk import Chunk
from paper_rag.models.document import DocumentStatus, DomainStatus
from paper_rag.models.job import JobState
from paper_rag.models.paper_profile import PaperProfile
from paper_rag.services.domain_admission import (
    CLASSIFIER_VERSION,
    DomainAdmissionResult,
)
from paper_rag.services.domain_assessment import (
    apply_domain_assessment,
    clear_knowledge_artifacts_for_review,
)


class FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = set(ids)

    def delete(self, ids: list[str]) -> None:
        self.ids.difference_update(ids)

    def get(self, **_kwargs) -> dict:
        return {"ids": sorted(self.ids)}


def review_result() -> DomainAdmissionResult:
    return DomainAdmissionResult(
        decision=DomainStatus.REVIEW_REQUIRED,
        decision_code="missing_domain_relationship",
        evidence_regions=(),
        passed_requirements=(),
        failed_requirements=("domain_relationship",),
        parse_quality=1.0,
        classifier_version=CLASSIFIER_VERSION,
        embedding_model_id="worker-test",
        config_fingerprint="f" * 64,
        duration_ms=10,
        evaluated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )


def test_enforced_review_blocks_indexing(session, document) -> None:
    document.domain_enforcement_version = CLASSIFIER_VERSION

    outcome = apply_domain_assessment(
        session,
        document,
        review_result(),
        trigger="upload",
    )

    assert outcome.may_index is False
    assert outcome.applied is True
    assert document.status == DocumentStatus.REVIEW_REQUIRED
    assert document.domain_status == DomainStatus.REVIEW_REQUIRED
    assert outcome.terminal_job_state == JobState.REVIEW_REQUIRED


def test_review_cleanup_removes_all_knowledge_artifacts(session, document) -> None:
    chunks = [
        Chunk(
            document_id=document.id,
            vector_id=f"vector-{index}",
            content=f"content {index}",
            page_start=index,
            page_end=index,
            chunk_index=index,
        )
        for index in (1, 2)
    ]
    profile = PaperProfile(
        document_id=document.id,
        status="ready",
        profile_version=1,
        parser_version="test",
        prompt_version="test",
        source_sha256="a" * 64,
    )
    session.add_all([*chunks, profile])
    session.commit()
    collection = FakeCollection([chunk.vector_id for chunk in chunks])

    clear_knowledge_artifacts_for_review(session, document, collection)

    assert (
        session.scalar(
            select(func.count(Chunk.id)).where(Chunk.document_id == document.id)
        )
        == 0
    )
    assert (
        session.scalar(
            select(func.count(PaperProfile.id)).where(
                PaperProfile.document_id == document.id
            )
        )
        == 0
    )
    assert collection.get(
        where={"document_id": str(document.id)}, include=[]
    )["ids"] == []


def test_legacy_reindex_assessment_is_shadow_only(session, document) -> None:
    document.domain_enforcement_version = None
    document.domain_status = DomainStatus.ACCEPTED
    document.status = DocumentStatus.COMPLETED

    outcome = apply_domain_assessment(
        session,
        document,
        review_result(),
        trigger="reindex",
    )

    assert outcome.applied is False
    assert outcome.may_index is True
    assert document.domain_status == DomainStatus.ACCEPTED
    assert document.status == DocumentStatus.COMPLETED
