from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag import models as _models  # noqa: F401
from paper_rag.db import Base, get_db_session
from paper_rag.main import create_app
from paper_rag.models.document import Document, DocumentStatus, DomainStatus
from paper_rag.models.domain_admission import DomainAssessment
from paper_rag.models.page import Page
from paper_rag.services.domain_admission import DomainAdmissionResult
from paper_rag.services.domain_assessment import record_domain_assessment


class ApiEmbeddingProvider:
    model_id = "api-domain-stub"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if any(
                marker in text.lower()
                for marker in ("metasurface", "metamaterial", "超表面", "超材料")
            )
            else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class EmptyCollection:
    def get(self, **_kwargs) -> dict:
        return {"ids": []}


@pytest.fixture
def api() -> Iterator[tuple[TestClient, Session]]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        yield session

    app = create_app()
    app.dependency_overrides[get_db_session] = override_session
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.clear()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def add_review_document(
    session: Session, filename: str = "review.pdf"
) -> tuple[Document, object]:
    document = Document(
        original_filename=filename,
        stored_path=f"data/{uuid4()}.pdf",
        file_sha256=uuid4().hex + uuid4().hex,
        status=DocumentStatus.REVIEW_REQUIRED,
        domain_status=DomainStatus.REVIEW_REQUIRED,
    )
    session.add(document)
    session.flush()
    result = DomainAdmissionResult(
        decision=DomainStatus.REVIEW_REQUIRED,
        decision_code="missing_domain_relationship",
        evidence_regions=(),
        passed_requirements=("parse_quality", "semantic_support"),
        failed_requirements=("domain_relationship",),
        parse_quality=0.95,
        classifier_version="positive-admission-v2",
        embedding_model_id="api-test",
        config_fingerprint="f" * 64,
        duration_ms=10,
        evaluated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    assessment = record_domain_assessment(
        session,
        document,
        result,
        trigger="upload",
        applied=True,
    )
    session.commit()
    return document, assessment


def test_document_list_exposes_latest_applied_domain_assessment(api) -> None:
    client, session = api
    _document, assessment = add_review_document(session)

    response = client.get("/api/documents")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["domain_assessment_id"] == str(assessment.id)
    assert item["domain_decision_code"] == "missing_domain_relationship"
    assert item["domain_failed_requirements"] == ["domain_relationship"]
    assert item["domain_evidence"] == []
    assert item["chunk_count"] == 0
    assert item["profile_status"] is None


def test_review_required_document_is_rejected_from_chat_scope(api) -> None:
    client, session = api
    document, _assessment = add_review_document(session)
    document.status = DocumentStatus.COMPLETED
    session.commit()

    response = client.post(
        "/api/chat",
        json={
            "session_id": "scope-test-session",
            "question": "这篇论文的主要结论是什么？",
            "scope": "selected",
            "document_ids": [str(document.id)],
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Selected document is not ready"


def test_approval_rejects_assessment_from_another_document(api) -> None:
    client, session = api
    document, _assessment = add_review_document(session, "first.pdf")
    _other_document, other_assessment = add_review_document(
        session, "second.pdf"
    )

    response = client.post(
        f"/api/documents/{document.id}/approve",
        json={"assessment_id": str(other_assessment.id)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "assessment does not belong to document"


def test_approval_rejects_stale_assessment(api) -> None:
    client, session = api
    document, first = add_review_document(session)
    second_result = replace(
        DomainAdmissionResult(
            decision=DomainStatus.REVIEW_REQUIRED,
            decision_code="insufficient_independent_regions",
            evidence_regions=(),
            passed_requirements=("parse_quality",),
            failed_requirements=("independent_regions",),
            parse_quality=0.95,
            classifier_version="positive-admission-v2",
            embedding_model_id="api-test",
            config_fingerprint="f" * 64,
            duration_ms=11,
            evaluated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        ),
        evaluated_at=first.created_at + timedelta(seconds=1),
    )
    record_domain_assessment(
        session,
        document,
        second_result,
        trigger="reindex",
        applied=True,
    )
    session.commit()

    response = client.post(
        f"/api/documents/{document.id}/approve",
        json={"assessment_id": str(first.id)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "assessment is stale"


def test_deletion_check_appends_diagnostic_without_overwriting_status(
    api, monkeypatch
) -> None:
    client, session = api
    document, prior = add_review_document(session)
    accepted_result = DomainAdmissionResult(
        decision=DomainStatus.ACCEPTED,
        decision_code="positive_evidence_quorum",
        evidence_regions=(),
        passed_requirements=(
            "domain_identity",
            "domain_relationship",
            "independent_regions",
        ),
        failed_requirements=(),
        parse_quality=0.99,
        classifier_version="positive-admission-v2",
        embedding_model_id="api-test",
        config_fingerprint="f" * 64,
        duration_ms=10,
        evaluated_at=prior.created_at + timedelta(seconds=1),
    )
    applied = record_domain_assessment(
        session,
        document,
        accepted_result,
        trigger="upload",
        applied=True,
    )
    document.status = DocumentStatus.COMPLETED
    session.add_all(
        [
            Page(
                document_id=document.id,
                page_number=1,
                text="A metasurface controls electromagnetic absorption. " * 5,
                quality_score=0.99,
            ),
            Page(
                document_id=document.id,
                page_number=2,
                text="Metasurface unit cells tune terahertz reflection phase. " * 5,
                quality_score=0.99,
            ),
        ]
    )
    session.commit()
    before_count = session.scalar(select(func.count(DomainAssessment.id)))
    monkeypatch.setattr(
        "paper_rag.api.documents.get_embedding_provider",
        lambda _settings: ApiEmbeddingProvider(),
    )
    monkeypatch.setattr(
        "paper_rag.api.documents.get_chroma_collection",
        lambda _settings, _provider: EmptyCollection(),
    )

    response = client.post(f"/api/documents/{document.id}/deletion-check")

    assert response.status_code == 200
    payload = response.json()
    assert payload["fresh_assessment_id"] != str(applied.id)
    assert payload["fresh_decision_code"] == "positive_evidence_quorum"
    assert session.scalar(select(func.count(DomainAssessment.id))) == before_count + 1
    session.refresh(document)
    assert document.domain_status == DomainStatus.ACCEPTED
    assert document.domain_decision_code == "positive_evidence_quorum"


def test_latest_review_assessment_can_be_manually_approved(api) -> None:
    client, session = api
    document, assessment = add_review_document(session)

    response = client.post(
        f"/api/documents/{document.id}/approve",
        json={"assessment_id": str(assessment.id)},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["assessment_id"] == str(assessment.id)
    assert payload["override_id"]
    session.refresh(document)
    assert document.domain_status == DomainStatus.MANUAL_APPROVED
    assert document.status == DocumentStatus.QUEUED


def test_quarantined_document_cannot_use_domain_approval(api) -> None:
    client, session = api
    document, assessment = add_review_document(session)
    document.domain_status = DomainStatus.QUARANTINED
    document.status = DocumentStatus.QUARANTINED
    session.commit()

    response = client.post(
        f"/api/documents/{document.id}/approve",
        json={"assessment_id": str(assessment.id)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Document does not require approval"
