from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag import models as _models  # noqa: F401 -- register model metadata
from paper_rag.db import Base
from paper_rag.models import audit as _audit  # noqa: F401
from paper_rag.models import chunk as _chunk  # noqa: F401
from paper_rag.models import formula as _formula  # noqa: F401
from paper_rag.models.document import (
    Document,
    DocumentStatus,
    DomainStatus,
    FormulaIndexStatus,
)


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
        value.rollback()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def documents(session: Session) -> list[Document]:
    result = [
        Document(
            id=uuid4(),
            original_filename=f"reference-paper-{index}.pdf",
            stored_path=f"data/reference-{uuid4()}.pdf",
            file_sha256=uuid4().hex + uuid4().hex,
            page_count=12,
            status=DocumentStatus.COMPLETED,
            domain_status=DomainStatus.ACCEPTED,
            formula_index_status=FormulaIndexStatus.READY,
        )
        for index in range(3)
    ]
    session.add_all(result)
    session.flush()
    return result


@pytest.fixture
def document(documents: list[Document]) -> Document:
    return documents[0]
