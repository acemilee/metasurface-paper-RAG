from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_rag.db import Base
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.formula import Formula
from paper_rag.services.formula_assets import refresh_formula_source_crop_hashes


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"


def test_source_crop_hash_is_stable_and_invalid_regions_fail_closed() -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    document = Document(
        original_filename=SAMPLE_PDF.name,
        stored_path=str(SAMPLE_PDF),
        file_sha256="e" * 64,
        page_count=9,
        status=DocumentStatus.COMPLETED,
    )
    session.add(document)
    session.flush()
    valid = Formula(
        id=uuid4(),
        document_id=document.id,
        page_number=4,
        placeholder="valid",
        bbox_json="[37.5, 182.0, 289.0, 274.0]",
        raw_text="valid formula",
        fidelity_status="needs_review",
    )
    invalid = Formula(
        id=uuid4(),
        document_id=document.id,
        page_number=4,
        placeholder="invalid",
        bbox_json="[1000, 1000, 1100, 1100]",
        raw_text="invalid formula",
        fidelity_status="needs_review",
    )
    session.add_all([valid, invalid])
    session.commit()

    first = refresh_formula_source_crop_hashes(session, document.id)
    first_hash = valid.source_crop_sha256
    second = refresh_formula_source_crop_hashes(session, document.id)

    assert first.hashed_formula_ids == (valid.id,)
    assert first.invalid_formula_ids == (invalid.id,)
    assert len(first_hash) == 64
    assert valid.source_crop_sha256 == first_hash
    assert second.hashed_formula_ids == (valid.id,)
    assert invalid.source_crop_sha256 is None
    assert invalid.fidelity_status == "unusable"
