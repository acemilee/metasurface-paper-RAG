from uuid import uuid4

import pytest

from paper_rag.api.chat import _generation_failure_record
from paper_rag.services.answer_audit import MissingFormulaClaimError, UnknownCitationError
from paper_rag.services.deepseek import DeepSeekProviderError, DeepSeekSchemaError
from paper_rag.services.retrieval import RetrievedChunk


def _evidence() -> list[RetrievedChunk]:
    return [RetrievedChunk(uuid4(), uuid4(), "formula evidence", 4, 4, None, [], 0.9)]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DeepSeekSchemaError("bad schema", validation_errors=["claims missing"]), "model_schema_failure"),
        (MissingFormulaClaimError("formula claim missing"), "missing_formula_claim"),
        (UnknownCitationError("citation is outside whitelist"), "unknown_citation"),
        (DeepSeekProviderError("timeout"), "provider_failure"),
    ],
)
def test_generation_failure_taxonomy(error: Exception, expected: str) -> None:
    record = _generation_failure_record(1, error, _evidence(), latency_ms=17)

    assert record["error_code"] == expected
    assert record["latency_ms"] == 17
    assert record["allowed_citation_ids"]
    assert "raw_content" not in record
