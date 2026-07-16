from __future__ import annotations

import json
import pickle
import sqlite3
from collections.abc import Callable
from threading import RLock
from typing import TypeVar
from uuid import UUID
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.errors import NotFoundError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from paper_rag.config import Settings
from paper_rag.models.chunk import Chunk
from paper_rag.services.chunking import ChunkDraft, make_vector_id
from paper_rag.services.embeddings import EmbeddingProvider


INDEX_VERSION = "cosine_v5"
HNSW_BATCH_SIZE = 1
HNSW_SYNC_THRESHOLD = 1
_T = TypeVar("_T")
_client = None
_collection = None
_cache_key: tuple[str, str] | None = None
_chroma_lock = RLock()
_last_read_error: str | None = None


class VectorIndexUnavailableError(RuntimeError):
    pass


def check_chroma_readable(settings: Settings) -> bool:
    global _last_read_error
    collection_label = "unknown"
    try:
        from paper_rag.services.embeddings import get_embedding_provider

        provider = get_embedding_provider(settings)
        collection_label = collection_name(provider.model_id)
        with _chroma_lock:
            collection = get_chroma_collection(settings, provider)
            collection.peek(limit=1)
        _last_read_error = None
        return True
    except Exception as exc:
        _last_read_error = (
            f"collection={collection_label}; path={settings.chroma_dir.resolve()}; "
            f"{type(exc).__name__}: {exc}"
        )[:1000]
        return False


def get_last_chroma_read_error() -> str | None:
    return _last_read_error


def finalize_chroma_persistence(settings: Settings, collection) -> int:
    database_path = settings.chroma_dir / "chroma.sqlite3"
    with sqlite3.connect(database_path, timeout=30.0) as connection:
        vector_segment = connection.execute(
            "select id from segments where collection=? and scope='VECTOR'",
            (str(collection.id),),
        ).fetchone()
        if vector_segment is None:
            return 0
        metadata_path = settings.chroma_dir / vector_segment[0] / "index_metadata.pickle"
        if not metadata_path.exists():
            return 0
        metadata = pickle.loads(metadata_path.read_bytes())
        persisted_ids = set(metadata.get("id_to_label", {}))
        topic = f"persistent://default/default/{collection.id}"
        queue_rows = connection.execute(
            "select seq_id, operation, id from embeddings_queue where topic=?",
            (topic,),
        ).fetchall()
        verified_seq_ids = [
            seq_id
            for seq_id, operation, vector_id in queue_rows
            if (operation in {0, 1, 2} and vector_id in persisted_ids)
            or (operation == 3 and vector_id not in persisted_ids)
        ]
        if len(verified_seq_ids) != len(queue_rows):
            return 0
        processed_high_watermark = connection.execute(
            "select max(seq_id) from max_seq_id"
        ).fetchone()[0]
        connection.executemany(
            "delete from embeddings_queue where seq_id=?",
            [(seq_id,) for seq_id in verified_seq_ids],
        )
        queue_high_watermark = connection.execute(
            "select max(seq_id) from embeddings_queue"
        ).fetchone()[0]
        if (
            processed_high_watermark is not None
            and (queue_high_watermark is None or queue_high_watermark < processed_high_watermark)
        ):
            connection.execute(
                """
                insert into embeddings_queue(seq_id, operation, topic, id)
                values (?, 3, 'maintenance://sequence-watermark', ?)
                """,
                (
                    processed_high_watermark,
                    f"sequence-watermark-{processed_high_watermark}",
                ),
            )
        connection.commit()
        return len(verified_seq_ids)


def legacy_collection_name(model_id: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in model_id).lower()
    return f"paper_chunks_{safe}"[:128]


def collection_name(model_id: str) -> str:
    return f"{legacy_collection_name(model_id)}_{INDEX_VERSION}"[:128]


def collection_configuration() -> CreateCollectionConfiguration:
    return {
        "hnsw": {
            "space": "cosine",
            "batch_size": HNSW_BATCH_SIZE,
            "sync_threshold": HNSW_SYNC_THRESHOLD,
        }
    }


def chroma_client_path(settings: Settings) -> str:
    resolved = settings.chroma_dir.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(settings.chroma_dir)


def get_chroma_collection(settings: Settings, provider: EmbeddingProvider):
    global _cache_key, _client, _collection
    key = (str(settings.chroma_dir.resolve()), collection_name(provider.model_id))
    with _chroma_lock:
        if _collection is not None and _cache_key == key:
            return _collection
        _close_cached_client()
        _client = chromadb.PersistentClient(path=chroma_client_path(settings))
        try:
            _collection = _client.get_collection(name=key[1])
        except (NotFoundError, ValueError):
            _collection = _client.create_collection(
                name=key[1],
                configuration=collection_configuration(),
                metadata={"model_id": provider.model_id, "dimension": provider.dimension, "index_version": INDEX_VERSION, "hnsw:space": "cosine"},
            )
        _cache_key = key
        return _collection


def refresh_chroma_collection(settings: Settings, provider: EmbeddingProvider):
    with _chroma_lock:
        _close_cached_client()
        SharedSystemClient.clear_system_cache()
        return get_chroma_collection(settings, provider)


def _close_cached_client() -> None:
    global _cache_key, _client, _collection
    if _client is not None:
        close = getattr(_client, "close", None)
        if callable(close):
            close()
    _client = None
    _collection = None
    _cache_key = None


def _document_vector_counts(
    session: Session,
    collection,
    document_ids: list[UUID],
) -> tuple[int, int]:
    if not document_ids:
        return 0, 0
    expected = len(list(session.scalars(select(Chunk.id).where(Chunk.document_id.in_(document_ids)))))
    where = {"document_id": str(document_ids[0])} if len(document_ids) == 1 else {
        "document_id": {"$in": [str(document_id) for document_id in document_ids]}
    }
    actual = len(collection.get(where=where, include=[])["ids"])
    return expected, actual


def run_synced_chroma_query(
    session: Session,
    settings: Settings,
    provider: EmbeddingProvider,
    document_ids: list[UUID],
    operation: Callable[[object], _T],
) -> _T:
    with _chroma_lock:
        collection = get_chroma_collection(settings, provider)
        try:
            counts = _document_vector_counts(session, collection, document_ids)
        except Exception:
            counts = (-1, -2)
        if counts[0] <= 0 or counts[0] != counts[1]:
            collection = refresh_chroma_collection(settings, provider)
            try:
                counts = _document_vector_counts(session, collection, document_ids)
            except Exception as exc:
                raise VectorIndexUnavailableError("向量索引刷新后仍无法读取") from exc
        if counts[0] <= 0 or counts[0] != counts[1]:
            raise VectorIndexUnavailableError(
                f"向量索引与知识库不同步：postgres={counts[0]}, chroma={counts[1]}"
            )
        try:
            return operation(collection)
        except Exception:
            collection = refresh_chroma_collection(settings, provider)
            try:
                refreshed_counts = _document_vector_counts(session, collection, document_ids)
                if refreshed_counts[0] <= 0 or refreshed_counts[0] != refreshed_counts[1]:
                    raise VectorIndexUnavailableError(
                        f"向量索引与知识库不同步：postgres={refreshed_counts[0]}, chroma={refreshed_counts[1]}"
                    )
                return operation(collection)
            except VectorIndexUnavailableError:
                raise
            except Exception as exc:
                raise VectorIndexUnavailableError("向量检索刷新重试失败") from exc


def replace_document_chunks(session: Session, document_id: UUID, drafts: list[ChunkDraft], chunking_version: str) -> list[Chunk]:
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    chunks = [Chunk(document_id=draft.document_id, vector_id=make_vector_id(draft.document_id, chunking_version, draft.chunk_index), content=draft.content, page_start=draft.page_start, page_end=draft.page_end, section_path=draft.section_path, content_type=draft.content_type, formula_ids_json=json.dumps([str(item) for item in draft.formula_ids]), chunk_index=draft.chunk_index, quality_score=draft.quality_score, has_low_confidence_ocr=draft.has_low_confidence_ocr) for draft in drafts]
    session.add_all(chunks)
    session.commit()
    return chunks


def upsert_chunks(collection, chunks: list[Chunk], vectors: list[list[float]], settings: Settings | None = None) -> None:
    collection.upsert(ids=[chunk.vector_id for chunk in chunks], embeddings=vectors, documents=[chunk.content for chunk in chunks], metadatas=[{"document_id": str(chunk.document_id), "chunk_id": str(chunk.id), "page_start": chunk.page_start, "page_end": chunk.page_end, "section_path": chunk.section_path or "", "content_type": chunk.content_type, "formula_ids": chunk.formula_ids_json, "quality_score": chunk.quality_score, "has_low_confidence_ocr": chunk.has_low_confidence_ocr} for chunk in chunks])
    if settings is not None:
        finalize_chroma_persistence(settings, collection)


def delete_document_vectors(collection, chunks: list[Chunk], settings: Settings | None = None) -> None:
    if chunks:
        collection.delete(ids=[chunk.vector_id for chunk in chunks])
        if settings is not None:
            finalize_chroma_persistence(settings, collection)


def verify_index_counts(session: Session, collection, document_id: UUID) -> tuple[int, int]:
    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document_id)))
    vector_count = len(collection.get(ids=[chunk.vector_id for chunk in chunks])["ids"]) if chunks else 0
    return len(chunks), vector_count
