from __future__ import annotations

from collections.abc import Generator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base, get_db_session


def _client() -> TestClient:
    from paper_rag.api.conversations import router

    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_session() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = override_session
    return TestClient(app)


def test_conversation_crud_and_restore_transcript() -> None:
    client = _client()

    created = client.post(
        "/api/conversations",
        json={"title": "Graphene notes", "scope": "selected", "document_ids": []},
    )
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]

    renamed = client.patch(
        f"/api/conversations/{conversation_id}",
        json={"title": "Graphene analysis"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Graphene analysis"

    listing = client.get("/api/conversations")
    assert listing.status_code == 200
    assert [item["conversation_id"] for item in listing.json()["items"]] == [conversation_id]

    restored = client.get(f"/api/conversations/{conversation_id}")
    assert restored.status_code == 200
    assert restored.json()["messages"] == []
    assert restored.json()["scope"] == "selected"

    reset = client.post(f"/api/conversations/{conversation_id}/reset")
    assert reset.status_code == 200
    assert reset.json()["messages"] == []

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_conversations_are_isolated() -> None:
    client = _client()
    first = client.post("/api/conversations", json={"title": "First"}).json()
    second = client.post("/api/conversations", json={"title": "Second"}).json()

    assert first["conversation_id"] != second["conversation_id"]
    assert client.get(f"/api/conversations/{first['conversation_id']}").json()["title"] == "First"
    assert client.get(f"/api/conversations/{second['conversation_id']}").json()["title"] == "Second"


def test_conversation_title_is_bounded_and_not_empty() -> None:
    client = _client()

    assert client.post("/api/conversations", json={"title": " "}).status_code == 422
    assert client.post("/api/conversations", json={"title": "x" * 201}).status_code == 422

