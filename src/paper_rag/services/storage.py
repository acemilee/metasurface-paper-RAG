from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from paper_rag.config import Settings


@dataclass(frozen=True)
class StoredUpload:
    original_filename: str
    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf_header(path: Path) -> None:
    with path.open("rb") as file_handle:
        header = file_handle.read(5)
    if header != b"%PDF-":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File is not a PDF")


async def save_uploaded_pdf(upload: UploadFile, settings: Settings) -> StoredUpload:
    settings.ensure_directories()
    original_filename = upload.filename or "uploaded.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF uploads are allowed")
    if upload.content_type and upload.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload content type must be application/pdf")

    destination = settings.upload_dir / f"{uuid.uuid4()}.pdf"
    size_bytes = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_bytes:
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF exceeds size limit")
                output.write(chunk)
        validate_pdf_header(destination)
        return StoredUpload(original_filename=original_filename, path=destination, sha256=sha256_file(destination), size_bytes=size_bytes)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
