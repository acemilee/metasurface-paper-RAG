import pytest
from pydantic import ValidationError

from paper_rag.config import Settings
from paper_rag.models.document import DomainStatus
from paper_rag.services.domain_admission import (
    CLASSIFIER_VERSION,
    AdmissionPage,
    evaluate_domain_admission,
)
from paper_rag.services.domain_assessment import admission_application_mode
from paper_rag.services.ingestion import (
    create_or_get_document_job,
    create_reindex_job,
)
from paper_rag.services.storage import StoredUpload


class ExplodingProvider:
    model_id = "safe-mode-test"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("provider must not be called")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("provider must not be called")


def test_new_document_uses_current_admission_policy(session, tmp_path) -> None:
    upload = StoredUpload(
        "new.pdf",
        tmp_path / "new.pdf",
        "a" * 64,
        100,
    )

    document, _job, created = create_or_get_document_job(session, upload)

    assert created is True
    assert document.domain_enforcement_version == CLASSIFIER_VERSION


def test_reindex_preserves_legacy_shadow_policy(session, document) -> None:
    document.domain_enforcement_version = None
    document.domain_status = DomainStatus.ACCEPTED
    session.commit()

    create_reindex_job(session, document)

    assert document.domain_enforcement_version is None
    assert admission_application_mode(document) == "shadow"


@pytest.mark.parametrize("value", ["v1", "fallback", "disabled"])
def test_unsafe_domain_gate_modes_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(domain_gate_safe_mode=value)


def test_review_all_safe_mode_does_not_call_embedding_provider() -> None:
    settings = Settings(
        domain_gate_safe_mode="review_all",
        domain_region_min_chars=80,
    )
    pages = [
        AdmissionPage(
            1,
            "A metasurface controls electromagnetic absorption. " * 5,
            0.99,
            None,
        )
    ]

    result = evaluate_domain_admission(pages, ExplodingProvider(), settings)

    assert result.decision == DomainStatus.REVIEW_REQUIRED
    assert result.decision_code == "gate_safe_mode"
    assert result.evidence_regions == ()
