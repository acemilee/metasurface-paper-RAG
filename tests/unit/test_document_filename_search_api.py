from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag import models as _models  # noqa: F401
from paper_rag.db import Base, get_db_session
from paper_rag.main import create_app
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document, DocumentStatus


@pytest.fixture
def api() -> Iterator[tuple[TestClient, Session]]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        yield session

    app = create_app()
    app.dependency_overrides[get_db_session] = override_session
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.clear()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def add_document(session: Session, filename: str) -> Document:
    document = Document(
        original_filename=filename,
        stored_path=f"data/{uuid4()}.pdf",
        file_sha256=uuid4().hex + uuid4().hex,
        status=DocumentStatus.COMPLETED,
    )
    session.add(document)
    session.flush()
    return document


def test_list_documents_searches_normalized_original_filename(api) -> None:
    client, session = api
    wanted = add_document(session, "Ｇraphene－石墨烯－１２.PDF")
    add_document(session, "unrelated.pdf")
    session.commit()

    response = client.get(
        "/api/documents", params={"filename": "graphene-石墨烯-12.pdf"}
    )

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["items"]] == [str(wanted.id)]


def test_filename_search_does_not_search_body_chunks(api) -> None:
    client, session = api
    document = add_document(session, "ordinary.pdf")
    session.add(
        Chunk(
            document_id=document.id,
            vector_id=f"chunk-{uuid4()}",
            content="secret-doi-10.1234 graphene title and author",
            page_start=1,
            page_end=1,
            chunk_index=0,
        )
    )
    session.commit()

    response = client.get(
        "/api/documents", params={"filename": "secret-doi-10.1234"}
    )

    assert response.status_code == 200
    assert response.json()["items"] == []


@pytest.mark.parametrize("query", [None, "", "   "])
def test_empty_filename_query_returns_unfiltered_list(api, query) -> None:
    client, session = api
    add_document(session, "one.pdf")
    add_document(session, "two.pdf")
    session.commit()
    params = {} if query is None else {"filename": query}

    response = client.get("/api/documents", params=params)

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2


@pytest.mark.parametrize(
    ("query", "expected"),
    [("100%", "gain_100%.pdf"), ("gain_", "gain_100%.pdf")],
)
def test_filename_search_treats_sql_wildcards_as_literals(api, query, expected) -> None:
    client, session = api
    add_document(session, "gain_100%.pdf")
    add_document(session, "gainX100Y.pdf")
    session.commit()

    response = client.get("/api/documents", params={"filename": query})

    assert [item["original_filename"] for item in response.json()["items"]] == [expected]


def test_filename_search_preserves_cursor_pagination(api) -> None:
    client, session = api
    for index in range(3):
        document = add_document(session, f"target-{index}.pdf")
        document.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=index
        )
    add_document(session, "unrelated.pdf")
    session.commit()

    first = client.get(
        "/api/documents", params={"filename": "target", "limit": 2}
    ).json()
    second = client.get(
        "/api/documents",
        params={
            "filename": "target",
            "limit": 2,
            "cursor": first["next_cursor"],
        },
    ).json()

    ids = [item["document_id"] for item in first["items"] + second["items"]]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert second["next_cursor"] is None


def test_filename_search_rejects_cursor_outside_filter(api) -> None:
    client, session = api
    add_document(session, "target.pdf")
    outside = add_document(session, "outside.pdf")
    session.commit()

    response = client.get(
        "/api/documents",
        params={"filename": "target", "cursor": str(outside.id)},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document cursor does not match filename filter"
