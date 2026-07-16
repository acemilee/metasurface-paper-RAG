from __future__ import annotations

import hashlib
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import httpx

from paper_rag.config import Settings


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    def __init__(self, dimension: int, model_id: str) -> None:
        self.dimension = dimension
        self.model_id = model_id

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[A-Za-z0-9.]+|[\u4e00-\u9fff]+", text.lower()):
            slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_path: Path,
        model_id: str,
        dimension: int,
        device: str,
        batch_size: int,
        max_seq_length: int,
    ) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local embedding model is missing: {model_path}")

        import torch
        from sentence_transformers import SentenceTransformer

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self._model = SentenceTransformer(
            str(model_path),
            local_files_only=True,
            device=resolved_device,
        )
        self._model.max_seq_length = max_seq_length
        actual_dimension = self._model.get_embedding_dimension()
        if actual_dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: configured={dimension}, model={actual_dimension}"
            )
        self.model_id = model_id
        self.dimension = dimension
        self.device = resolved_device
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HttpEmbeddingProvider:
    def __init__(
        self,
        service_url: str,
        model_id: str,
        dimension: int,
        timeout_seconds: float,
        max_batch_size: int,
        ingestion_batch_size: int,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.max_batch_size = max_batch_size
        self.ingestion_batch_size = ingestion_batch_size
        self._client = httpx.Client(
            base_url=service_url.rstrip("/"), timeout=timeout_seconds, trust_env=False
        )
        response = None
        for attempt in range(5):
            try:
                response = self._client.get("/health")
                response.raise_for_status()
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                transient = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {503, 504}
                )
                if not transient or attempt == 4:
                    raise
                time.sleep(1.0)
        assert response is not None
        metadata = response.json()
        if metadata.get("model_id") != model_id or metadata.get("dimension") != dimension:
            raise ValueError("Embedding service model metadata does not match configuration")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.ingestion_batch_size):
            response = self._client.post(
                "/embed",
                json={
                    "texts": texts[start:start + self.ingestion_batch_size],
                    "priority": "ingestion",
                },
            )
            response.raise_for_status()
            vectors.extend(response.json()["vectors"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.post(
            "/embed", json={"texts": [text], "priority": "query"}
        )
        response.raise_for_status()
        return response.json()["vectors"][0]


@lru_cache(maxsize=4)
def _get_cached_provider(
    provider_name: str,
    model_path: str,
    model_id: str,
    dimension: int,
    device: str,
    batch_size: int,
    max_seq_length: int,
    service_url: str,
    timeout_seconds: float,
    max_batch_size: int,
    ingestion_batch_size: int,
) -> EmbeddingProvider:
    if provider_name == "hash":
        return HashEmbeddingProvider(dimension, model_id)
    if provider_name == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(
            model_path=Path(model_path),
            model_id=model_id,
            dimension=dimension,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
    if provider_name == "http":
        return HttpEmbeddingProvider(
            service_url, model_id, dimension, timeout_seconds, max_batch_size,
            ingestion_batch_size,
        )
    raise ValueError(f"Unsupported embedding provider: {provider_name}")


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return _get_cached_provider(
        settings.embedding_provider,
        str(settings.embedding_model_path.resolve()),
        settings.embedding_model_id,
        settings.embedding_dimension,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.embedding_max_seq_length,
        settings.embedding_service_url,
        settings.embedding_service_timeout_seconds,
        settings.embedding_service_max_batch_size,
        settings.embedding_ingestion_batch_size,
    )
