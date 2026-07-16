from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from paper_rag.services import metrics


def test_readiness_is_false_when_main_vector_collection_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = object()
    settings = SimpleNamespace(
        chroma_dir=tmp_path,
        readiness_worker_stale_seconds=30,
        embedding_service_url="http://embedding",
        embedding_model_id="BAAI/bge-m3",
        readiness_min_disk_free_gb=0.0,
        worker_max_memory_percent=100.0,
    )
    monkeypatch.setattr(metrics, "check_chroma_readable", lambda *_args: False)
    monkeypatch.setattr(metrics, "get_last_chroma_read_error", lambda: "InternalError: test")
    monkeypatch.setattr(
        metrics.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"model_id": "BAAI/bge-m3"},
        ),
    )
    monkeypatch.setattr(metrics.psutil, "virtual_memory", lambda: SimpleNamespace(percent=10.0))

    report = metrics.build_readiness_report(session, settings)

    assert report.chroma_writable is True
    assert report.chroma_readable is False
    assert report.chroma_error == "InternalError: test"
    assert report.ready is False
