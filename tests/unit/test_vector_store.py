from dataclasses import dataclass
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from paper_rag.services import vector_store
from paper_rag.services.vector_store import VectorIndexUnavailableError, chroma_client_path, collection_configuration, collection_name, finalize_chroma_persistence


def test_collection_name_changes_with_embedding_model() -> None:
    assert collection_name("hash-embedding-v1") != collection_name("bge-m3-v1")


def test_collection_name_is_chroma_safe() -> None:
    assert "-" not in collection_name("model/name:1")
    assert collection_name("BAAI/bge-m3") == "paper_chunks_baai_bge_m3_cosine_v5"


def test_collection_uses_frequent_persistence_for_restart_safety() -> None:
    configuration = collection_configuration()

    assert configuration["hnsw"]["batch_size"] == 1
    assert configuration["hnsw"]["sync_threshold"] == 1


def test_chroma_client_prefers_workspace_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(tmp_path / "data" / "chroma")

    assert chroma_client_path(settings) == str(Path("data") / "chroma")


def test_finalizer_clears_only_hnsw_confirmed_queue_entries(tmp_path) -> None:
    import chromadb

    settings = Settings(tmp_path / "chroma")
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.create_collection(
        "finalizer-probe",
        configuration=collection_configuration(),
    )
    collection.upsert(ids=["one"], embeddings=[[1.0, 0.0]], documents=["one"])

    cleared = finalize_chroma_persistence(settings, collection)
    with sqlite3.connect(settings.chroma_dir / "chroma.sqlite3") as connection:
        queue_count = connection.execute(
            "select count(*) from embeddings_queue where topic like ?",
            (f"%{collection.id}",),
        ).fetchone()[0]

    assert cleared == 1
    assert queue_count == 0
    assert collection.count() == 1

    collection.upsert(ids=["two"], embeddings=[[0.0, 1.0]], documents=["two"])
    assert collection.get(ids=["two"], include=[])["ids"] == ["two"]
    assert collection.count() == 2


@dataclass
class Provider:
    model_id: str = "cross-process-test"
    dimension: int = 2


@dataclass
class Settings:
    chroma_dir: object


def test_query_refreshes_collection_after_external_process_write(tmp_path, monkeypatch) -> None:
    settings = Settings(tmp_path / "chroma")
    provider = Provider()
    document_id = uuid4()
    collection = vector_store.refresh_chroma_collection(settings, provider)
    collection.upsert(
        ids=["old"],
        embeddings=[[1.0, 0.0]],
        documents=["old"],
        metadatas=[{"document_id": str(document_id)}],
    )
    writer = (
        "import chromadb,sys; "
        "c=chromadb.PersistentClient(path=sys.argv[1]).get_collection(sys.argv[2]); "
        "c.upsert(ids=['new'],embeddings=[[0.0,1.0]],documents=['new'],"
        "metadatas=[{'document_id':sys.argv[3]}])"
    )
    subprocess.run(
        [sys.executable, "-c", writer, str(settings.chroma_dir), collection.name, str(document_id)],
        check=True,
        capture_output=True,
        text=True,
    )

    def counts(_session, current_collection, _document_ids):
        actual = len(current_collection.get(where={"document_id": str(document_id)}, include=[])["ids"])
        return 2, actual

    monkeypatch.setattr(vector_store, "_document_vector_counts", counts)
    result = vector_store.run_synced_chroma_query(
        None,
        settings,
        provider,
        [document_id],
        lambda current_collection: current_collection.get(ids=["new"], include=[])["ids"],
    )

    assert result == ["new"]


def test_query_fails_explicitly_when_refresh_remains_inconsistent(tmp_path, monkeypatch) -> None:
    settings = Settings(tmp_path / "chroma")
    provider = Provider()
    document_id = uuid4()
    monkeypatch.setattr(vector_store, "_document_vector_counts", lambda *_args: (2, 1))

    with pytest.raises(VectorIndexUnavailableError, match="不同步"):
        vector_store.run_synced_chroma_query(
            None, settings, provider, [document_id], lambda _collection: None
        )
