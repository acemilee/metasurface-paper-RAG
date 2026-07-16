from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import chromadb
import fitz
import httpx
from sqlalchemy import func, select

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.models.audit import AnswerAudit
from paper_rag.models.chunk import Chunk
from paper_rag.models.document import Document
from paper_rag.models.formula import Formula
from paper_rag.models.page import Page, TextBlock
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.vector_store import get_chroma_collection, verify_index_counts


SAMPLE_NAME = "Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"


def create_unique_pdf_copy(source: Path, destination: Path) -> None:
    document = fitz.open(source)
    with document:
        metadata = dict(document.metadata)
        metadata["keywords"] = f"phase35-acceptance-{uuid.uuid4()}"
        document.set_metadata(metadata)
        document.save(destination)


def wait_for_job(client: httpx.Client, job_id: str, timeout_seconds: int) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    states: list[str] = []
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        state = payload["state"]
        if not states or states[-1] != state:
            states.append(state)
        if state == "completed":
            return states
        if state == "failed":
            raise RuntimeError(
                f"Ingestion failed: {payload.get('error_code')}: {payload.get('error_message')}"
            )
        time.sleep(0.5)
    raise TimeoutError(f"Job did not complete within {timeout_seconds}s; observed={states}")


def database_snapshot(document_id: uuid.UUID) -> dict[str, int | bool]:
    settings = get_settings()
    provider = get_embedding_provider(settings)
    collection = get_chroma_collection(settings, provider)
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        if document is None:
            raise RuntimeError("Uploaded document is missing from PostgreSQL")
        pages = session.scalar(select(func.count()).select_from(Page).where(Page.document_id == document_id)) or 0
        blocks = session.scalar(
            select(func.count()).select_from(TextBlock).join(Page).where(Page.document_id == document_id)
        ) or 0
        formulas = session.scalar(
            select(func.count()).select_from(Formula).where(Formula.document_id == document_id)
        ) or 0
        chunks, vectors = verify_index_counts(session, collection, document_id)
    jsonl_path = settings.parsed_dir / f"{document_id}.jsonl"
    jsonl_lines = len(jsonl_path.read_text(encoding="utf-8").splitlines()) if jsonl_path.exists() else 0
    return {
        "pages": pages,
        "text_blocks": blocks,
        "formulas": formulas,
        "postgres_chunks": chunks,
        "chroma_vectors": vectors,
        "jsonl_exists": jsonl_path.exists(),
        "jsonl_lines": jsonl_lines,
    }


def ask(client: httpx.Client, session_id: str, question: str, document_id: str) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "question": question,
            "document_id": document_id,
            "top_n": 8,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run(base_url: str, timeout_seconds: int) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set in the current process environment")

    source = Path.cwd() / SAMPLE_NAME
    if not source.is_file():
        raise FileNotFoundError(f"Acceptance PDF is missing: {source}")

    with tempfile.TemporaryDirectory(prefix="paper-rag-phase35-") as temp_dir:
        upload_source = Path(temp_dir) / "phase35-acceptance.pdf"
        create_unique_pdf_copy(source, upload_source)

        with httpx.Client(base_url=base_url, timeout=30) as client:
            health = client.get("/health")
            health.raise_for_status()
            with upload_source.open("rb") as file_handle:
                upload = client.post(
                    "/api/documents",
                    files={"file": (upload_source.name, file_handle, "application/pdf")},
                )
            upload.raise_for_status()
            accepted = upload.json()
            document_id = uuid.UUID(accepted["document_id"])
            states = wait_for_job(client, accepted["job_id"], timeout_seconds)
            snapshot = database_snapshot(document_id)

            if snapshot["pages"] != 9 or snapshot["jsonl_lines"] != 9:
                raise AssertionError(f"Page/JSONL mismatch: {snapshot}")
            if snapshot["formulas"] <= 0:
                raise AssertionError(f"Formula stage did not produce records: {snapshot}")
            if snapshot["postgres_chunks"] <= 0 or snapshot["postgres_chunks"] != snapshot["chroma_vectors"]:
                raise AssertionError(f"PostgreSQL/Chroma mismatch: {snapshot}")

            session_id = f"phase35-{uuid.uuid4()}"
            key_response = client.post(
                "/api/session/deepseek-key",
                json={"session_id": session_id, "api_key": api_key},
            )
            key_response.raise_for_status()
            try:
                english = ask(client, session_id, "What is the absorption behavior around 17.6 GHz?", str(document_id))
                chinese = ask(client, session_id, "石墨烯柔性超表面的吸收调控机制是什么？", str(document_id))
                refusal = ask(client, session_id, "论文在100 THz光学共振方面报告了什么？", str(document_id))
            finally:
                client.delete(f"/api/session/deepseek-key/{session_id}")

    if english["refused"] or not english["citations"]:
        raise AssertionError(f"English grounded answer failed: {english}")
    if chinese["refused"] or not chinese["citations"]:
        raise AssertionError(f"Chinese cross-language answer failed: {chinese}")
    if not refusal["refused"] or refusal["citations"]:
        raise AssertionError(f"Unsupported question was not refused: {refusal}")

    with SessionLocal() as session:
        audit_rows = session.scalar(
            select(func.count()).select_from(AnswerAudit).where(AnswerAudit.document_id == document_id)
        ) or 0

    return {
        "document_id": str(document_id),
        "job_id": accepted["job_id"],
        "observed_states": states,
        "storage": snapshot,
        "english": {
            "refused": english["refused"],
            "audit_result": english["audit_result"],
            "citation_pages": [[item["page_start"], item["page_end"]] for item in english["citations"]],
        },
        "chinese": {
            "refused": chinese["refused"],
            "audit_result": chinese["audit_result"],
            "citation_pages": [[item["page_start"], item["page_end"]] for item in chinese["citations"]],
        },
        "unsupported": {
            "refused": refusal["refused"],
            "reason": refusal["refusal_reason"],
        },
        "answer_audit_rows": audit_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.timeout_seconds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
