import asyncio
from types import SimpleNamespace
from uuid import uuid4

from paper_rag.schemas.chat import NoveltyClaim
from paper_rag.services.deepseek import _audit_single_novelty_claim
from paper_rag.services.retrieval import RetrievedChunk


class FakeCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    async def create(self, **_kwargs):
        content = next(self.outputs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(outputs))


def claim_and_evidence():
    chunk_id = uuid4()
    document_id = uuid4()
    claim = NoveltyClaim(
        claim="本文提出了独立调谐结构",
        citation_id=chunk_id,
        claim_strength="explicit",
    )
    evidence = RetrievedChunk(
        chunk_id,
        document_id,
        "The proposed structure provides independent tuning capability.",
        1,
        1,
        None,
        [],
        0.8,
    )
    return claim, evidence


def test_single_claim_uses_minimal_contract_after_schema_repair_failure() -> None:
    claim, evidence = claim_and_evidence()
    client = FakeClient([
        "not-json",
        '{"unexpected":"wrapper"}',
        '{"verdict":"entailed","reason":"The context directly supports the claim."}',
    ])
    settings = SimpleNamespace(deepseek_model="deepseek-v4-flash")

    result = asyncio.run(_audit_single_novelty_claim(client, settings, claim, evidence, 2))

    assert result.verdict == "entailed"
    assert result.claim_index == 2
    assert result.attempt_count == 3
    assert len(result.raw_output_sha256) == 3
    assert len(result.validation_errors) == 2


def test_single_claim_schema_failures_are_isolated_as_unavailable() -> None:
    claim, evidence = claim_and_evidence()
    client = FakeClient(["bad", "still bad", "also bad"])
    settings = SimpleNamespace(deepseek_model="deepseek-v4-flash")

    result = asyncio.run(_audit_single_novelty_claim(client, settings, claim, evidence, 0))

    assert result.verdict == "audit_unavailable"
    assert result.error_code == "schema_failure"
    assert result.attempt_count == 3


def test_empty_initial_output_continues_to_repair() -> None:
    claim, evidence = claim_and_evidence()
    client = FakeClient([
        "",
        '{"verdict":"entailed","reason":"Supported after repair."}',
    ])
    settings = SimpleNamespace(deepseek_model="deepseek-v4-flash")

    result = asyncio.run(_audit_single_novelty_claim(client, settings, claim, evidence, 0))

    assert result.verdict == "entailed"
    assert result.attempt_count == 2
