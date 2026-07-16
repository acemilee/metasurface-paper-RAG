import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from paper_rag.config import Settings
from paper_rag.services.storage import save_uploaded_pdf


def _settings(tmp_path):
    return Settings(upload_dir=tmp_path / "uploads", parsed_dir=tmp_path / "parsed", chroma_dir=tmp_path / "chroma")


def test_save_uploaded_pdf_streams_and_hashes_valid_pdf(tmp_path) -> None:
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"%PDF-1.7\nexample"))

    stored = asyncio.run(save_uploaded_pdf(upload, _settings(tmp_path)))

    assert stored.path.exists()
    assert stored.size_bytes == len(b"%PDF-1.7\nexample")
    assert len(stored.sha256) == 64


def test_save_uploaded_pdf_rejects_non_pdf_and_removes_temp_file(tmp_path) -> None:
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"not a pdf"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(save_uploaded_pdf(upload, _settings(tmp_path)))

    assert error.value.status_code == 415
    assert not list((tmp_path / "uploads").glob("*"))


def test_save_uploaded_pdf_rejects_unexpected_mime_type(tmp_path) -> None:
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"%PDF-1.7\nexample"), headers={"content-type": "text/plain"})

    with pytest.raises(HTTPException) as error:
        asyncio.run(save_uploaded_pdf(upload, _settings(tmp_path)))

    assert error.value.status_code == 415
