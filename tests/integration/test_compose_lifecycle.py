from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN_COMPOSE_TESTS = os.getenv("PAPER_RAG_RUN_COMPOSE_TESTS") == "1"
SKIP_BUILD = os.getenv("PAPER_RAG_COMPOSE_SKIP_BUILD") == "1"


def compose(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def build_image(timeout: int = 1900) -> subprocess.CompletedProcess[str]:
    command = ["docker", "build", "--tag", "paper-rag:local"]
    proxy = os.getenv("PAPER_RAG_BUILD_PROXY")
    if proxy:
        command.extend(
            [
                "--build-arg",
                f"HTTP_PROXY={proxy}",
                "--build-arg",
                f"HTTPS_PROXY={proxy}",
            ]
        )
    command.append(".")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def wait_for_ready(timeout: int = 300) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8010/ready", timeout=10
            ) as response:
                payload = json.load(response)
            if payload.get("ready") is True:
                return payload
        except Exception as exc:  # readiness legitimately changes during restart
            last_error = exc
        time.sleep(2)
    raise AssertionError(f"readiness did not recover: {last_error}")


def document_count() -> int:
    result = compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "paper_rag",
        "-d",
        "paper_rag",
        "-Atc",
        "select count(*) from documents;",
    )
    return int(result.stdout.strip())


@pytest.mark.skipif(
    not RUN_COMPOSE_TESTS,
    reason="set PAPER_RAG_RUN_COMPOSE_TESTS=1 to run Docker lifecycle acceptance",
)
def test_compose_start_stop_restart_preserves_library_and_recovers_readiness() -> None:
    compose("config", "--quiet")
    sentinel = ROOT / "data" / ".compose-lifecycle-sentinel"
    sentinel_value = str(uuid4())
    sentinel.write_text(sentinel_value, encoding="utf-8")
    try:
        if not SKIP_BUILD:
            build_image()
        compose(
            "up",
            "--no-build",
            "--detach",
            "--wait",
            "--wait-timeout",
            "1800",
            timeout=1900,
        )
        with urllib.request.urlopen("http://127.0.0.1:8010/", timeout=10) as response:
            assert response.status == 200
        first_ready = wait_for_ready()
        assert all(
            first_ready[key]
            for key in (
                "postgres",
                "chroma_writable",
                "chroma_readable",
                "worker_alive",
                "embedding_service",
            )
        )

        count_before = document_count()
        assert (ROOT / "models" / "BAAI-bge-m3" / "modules.json").is_file()

        compose("down")
        compose("up", "--detach", "--wait", "--wait-timeout", "600", timeout=700)
        assert wait_for_ready()["ready"] is True
        assert document_count() == count_before
        assert sentinel.read_text(encoding="utf-8") == sentinel_value

        compose("restart", "embedding", "worker", "api", timeout=180)
        assert wait_for_ready(timeout=600)["ready"] is True
    except Exception:
        diagnostics = compose(
            "logs",
            "--tail",
            "200",
            "model-init",
            "migrate",
            "embedding",
            "worker",
            "api",
            timeout=60,
        )
        pytest.fail(f"Compose lifecycle failed.\n{diagnostics.stdout}\n{diagnostics.stderr}")
    finally:
        sentinel.unlink(missing_ok=True)
