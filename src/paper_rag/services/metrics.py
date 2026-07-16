from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import psutil
import httpx
from sqlalchemy import text

from paper_rag.config import Settings
from paper_rag.models.job import WorkerHeartbeat
from paper_rag.services.vector_store import check_chroma_readable, get_last_chroma_read_error


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    postgres: bool
    chroma_writable: bool
    chroma_readable: bool
    chroma_error: str | None
    worker_alive: bool
    embedding_service: bool
    disk_free_gb: float
    memory_percent: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_readiness_report(session, settings: Settings) -> ReadinessReport:
    postgres = True
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        postgres = False
    chroma_writable = settings.chroma_dir.exists() and settings.chroma_dir.is_dir()
    chroma_readable = chroma_writable and check_chroma_readable(settings)
    cutoff = datetime.now().astimezone() - timedelta(
        seconds=settings.readiness_worker_stale_seconds
    )
    worker_alive = session.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.last_seen_at >= cutoff
    ).first() is not None if postgres else False
    embedding_service = False
    try:
        response = httpx.get(
            f"{settings.embedding_service_url.rstrip('/')}/health", timeout=2.0
        )
        embedding_service = (
            response.status_code == 200
            and response.json().get("model_id") == settings.embedding_model_id
        )
    except (httpx.HTTPError, ValueError):
        pass
    disk_free_gb = psutil.disk_usage(str(settings.chroma_dir.resolve())).free / 1024**3
    memory_percent = psutil.virtual_memory().percent
    ready = (
        postgres
        and chroma_writable
        and chroma_readable
        and worker_alive
        and embedding_service
        and disk_free_gb >= settings.readiness_min_disk_free_gb
        and memory_percent < settings.worker_max_memory_percent
    )
    return ReadinessReport(
        ready=ready,
        postgres=postgres,
        chroma_writable=chroma_writable,
        chroma_readable=chroma_readable,
        chroma_error=get_last_chroma_read_error(),
        worker_alive=worker_alive,
        embedding_service=embedding_service,
        disk_free_gb=disk_free_gb,
        memory_percent=memory_percent,
    )
