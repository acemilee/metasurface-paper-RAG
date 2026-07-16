from __future__ import annotations

import argparse
from datetime import datetime
from uuid import uuid4

import chromadb
from chromadb.errors import NotFoundError
from sqlalchemy import select

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.models.chunk import Chunk
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.vector_store import (
    INDEX_VERSION,
    collection_configuration,
    collection_name,
    legacy_collection_name,
    upsert_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--drop-legacy", action="store_true")
    parser.add_argument("--resume-collection")
    args = parser.parse_args()
    settings = get_settings()
    provider = get_embedding_provider(settings)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    target_name = collection_name(provider.model_id)
    build_name = args.resume_collection or f"{target_name}_rebuild_{uuid4().hex[:8]}"[:128]
    if args.resume_collection:
        collection = client.get_collection(build_name)
    else:
        collection = client.create_collection(
            build_name,
            configuration=collection_configuration(),
            metadata={
                "model_id": provider.model_id,
                "dimension": provider.dimension,
                "index_version": INDEX_VERSION,
                "hnsw:space": "cosine",
            },
        )
    with SessionLocal() as session:
        chunks = list(session.scalars(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)))
        existing_ids = set(collection.get(include=[])["ids"]) if args.resume_collection else set()
        missing_chunks = [chunk for chunk in chunks if chunk.vector_id not in existing_ids]
        for start in range(0, len(missing_chunks), args.batch_size):
            batch = missing_chunks[start:start + args.batch_size]
            vectors = provider.embed_documents([chunk.content for chunk in batch])
            upsert_chunks(collection, batch, vectors, settings)
    if collection.count() != len(chunks):
        raise RuntimeError(f"Index rebuild mismatch: postgres={len(chunks)}, chroma={collection.count()}")
    existing_names = {item.name for item in client.list_collections()}
    if target_name in existing_names:
        existing = client.get_collection(target_name)
        backup_name = f"{target_name}_broken_{datetime.now().strftime('%Y%m%d%H%M%S')}"[:128]
        existing.modify(name=backup_name)
    collection.modify(name=target_name)
    if args.drop_legacy:
        legacy_name = legacy_collection_name(provider.model_id)
        if legacy_name in {item.name for item in client.list_collections()}:
            client.delete_collection(legacy_name)
    print({"collection": target_name, "metric": "cosine", "chunks": len(chunks), "resumed_from": len(existing_ids), "legacy_dropped": args.drop_legacy, "build_collection": build_name})


if __name__ == "__main__":
    main()
