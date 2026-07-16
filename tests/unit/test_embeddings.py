from pathlib import Path

import httpx
import pytest

from paper_rag.services import embeddings
from paper_rag.config import Settings
from paper_rag.services.embeddings import HashEmbeddingProvider, HttpEmbeddingProvider, get_embedding_provider


def test_hash_provider_remains_available_for_unit_tests() -> None:
    provider = get_embedding_provider(
        Settings(
            embedding_provider="hash",
            embedding_model_id="hash-test",
            embedding_dimension=32,
        )
    )

    assert isinstance(provider, HashEmbeddingProvider)
    assert len(provider.embed_query("graphene metasurface")) == 32


def test_sentence_transformer_requires_local_model_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_embedding_provider(
            Settings(
                embedding_provider="sentence_transformer",
                embedding_model_path=tmp_path / "missing-model",
            )
        )


def test_http_provider_batches_requests() -> None:
    class Response:
        def __init__(self, count: int) -> None:
            self.count = count

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"vectors": [[1.0, 0.0] for _ in range(self.count)]}

    class Client:
        def post(self, path: str, json: dict) -> Response:
            assert path == "/embed"
            return Response(len(json["texts"]))

    provider = HttpEmbeddingProvider.__new__(HttpEmbeddingProvider)
    provider.model_id = "test"
    provider.dimension = 2
    provider.max_batch_size = 2
    provider.ingestion_batch_size = 2
    provider._client = Client()

    assert len(provider.embed_documents(["a", "b", "c"])) == 3


def test_http_provider_retries_transient_health_503(monkeypatch) -> None:
    calls = 0
    client_options = {}

    class Client:
        def get(self, path: str):
            nonlocal calls
            calls += 1
            request = httpx.Request("GET", f"http://embedding{path}")
            if calls < 3:
                return httpx.Response(503, request=request)
            return httpx.Response(
                200,
                request=request,
                json={"model_id": "test", "dimension": 2},
            )

    def build_client(**kwargs):
        client_options.update(kwargs)
        return Client()

    monkeypatch.setattr(embeddings.httpx, "Client", build_client)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)

    provider = HttpEmbeddingProvider(
        "http://embedding",
        "test",
        2,
        10.0,
        8,
        4,
    )

    assert provider.model_id == "test"
    assert calls == 3
    assert client_options["trust_env"] is False
