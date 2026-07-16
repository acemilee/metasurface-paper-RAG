from __future__ import annotations

from contextlib import asynccontextmanager
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_rag.config import get_settings
from paper_rag.services.embeddings import SentenceTransformerEmbeddingProvider


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    priority: Literal["query", "ingestion"] = "query"


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


_provider: SentenceTransformerEmbeddingProvider | None = None


class PriorityInferenceScheduler:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._query_waiting = 0
        self._ingestion_waiting = 0

    def run(self, priority: str, operation):
        with self._condition:
            if priority == "query":
                self._query_waiting += 1
            else:
                self._ingestion_waiting += 1
            self._condition.wait_for(
                lambda: not self._active
                and (priority == "query" or self._query_waiting == 0)
            )
            if priority == "query":
                self._query_waiting -= 1
            else:
                self._ingestion_waiting -= 1
            self._active = True
        try:
            return operation()
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def status(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "active": self._active,
                "query_waiting": self._query_waiting,
                "ingestion_waiting": self._ingestion_waiting,
            }


_scheduler = PriorityInferenceScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider
    settings = get_settings()
    _provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_path,
        settings.embedding_model_id,
        settings.embedding_dimension,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.embedding_max_seq_length,
    )
    yield
    _provider = None


app = FastAPI(title="Paper RAG Embedding Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    if _provider is None:
        raise HTTPException(status_code=503, detail="Embedding model is not ready")
    return {
        "status": "ok",
        "model_id": _provider.model_id,
        "dimension": _provider.dimension,
        "scheduler": _scheduler.status(),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    settings = get_settings()
    if _provider is None:
        raise HTTPException(status_code=503, detail="Embedding model is not ready")
    if len(request.texts) > settings.embedding_service_max_batch_size:
        raise HTTPException(status_code=413, detail="Embedding batch is too large")
    if any(len(text) > settings.embedding_service_max_text_chars for text in request.texts):
        raise HTTPException(status_code=413, detail="Embedding text is too large")
    vectors = _scheduler.run(
        request.priority, lambda: _provider.embed_documents(request.texts)
    )
    return EmbedResponse(vectors=vectors)
