from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from paper_rag.api.formulas import router
from paper_rag.db import Base, get_db_session
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"
requires_sample_pdf = pytest.mark.skipif(
    not SAMPLE_PDF.is_file(),
    reason="private regression PDF is not distributed",
)


def _build_formula_client(
    *,
    bbox_json: str = "[37.5, 182.0, 289.0, 274.0]",
    page_number: int = 4,
    stored_path: str | None = None,
) -> tuple[TestClient, object]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    formula_id = uuid4()
    with testing_session() as session:
        document = Document(
            original_filename=SAMPLE_PDF.name,
            stored_path=stored_path or str(SAMPLE_PDF),
            file_sha256="b" * 64,
            page_count=9,
            status=DocumentStatus.COMPLETED,
        )
        session.add(document)
        session.flush()
        session.add(
            Formula(
                id=formula_id,
                document_id=document.id,
                page_number=page_number,
                placeholder=f"公式_placeholder_{formula_id}",
                bbox_json=bbox_json,
                raw_text="Kubo source glyphs",
                formula_number="1a",
                group_key="equation-1",
                normalized_text="Kubo source glyphs",
                fidelity_status="needs_review",
            )
        )
        session.commit()

    def override_session() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = override_session
    client = TestClient(app)
    return client, formula_id


@requires_sample_pdf
def test_formula_image_endpoint_renders_the_original_pdf_bbox() -> None:
    client, formula_id = _build_formula_client()

    response = client.get(f"/api/formulas/{formula_id}/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(response.content) > 1000
    assert client.get(f"/api/formulas/{uuid4()}/image").status_code == 404


@pytest.mark.parametrize(
    "bbox_json",
    ["null", "[1, 2, 3]", "[0, 0, 0, 10]", "[0, 0, 1e309, 10]"],
)
@requires_sample_pdf
def test_formula_image_rejects_invalid_bbox(bbox_json: str) -> None:
    client, formula_id = _build_formula_client(bbox_json=bbox_json)

    response = client.get(f"/api/formulas/{formula_id}/image")

    assert response.status_code == 422
    assert response.json()["detail"] == "Formula source region is invalid"


@requires_sample_pdf
def test_formula_image_rejects_page_outside_pdf() -> None:
    client, formula_id = _build_formula_client(page_number=999)

    assert client.get(f"/api/formulas/{formula_id}/image").status_code == 422


def test_formula_image_reports_missing_source_pdf(tmp_path: Path) -> None:
    client, formula_id = _build_formula_client(stored_path=str(tmp_path / "missing.pdf"))

    assert client.get(f"/api/formulas/{formula_id}/image").status_code == 404
