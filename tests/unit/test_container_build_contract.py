from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_application_image_uses_stable_python_and_excludes_local_state() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert 'requires-python = ">=3.12,<3.14"' in pyproject
    assert "PYTHONPATH=/app/src" in dockerfile
    assert "torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" in dockerfile
    assert "f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338" in dockerfile
    assert "ARG HTTPS_PROXY" in dockerfile
    assert "hashlib.sha256" in dockerfile
    assert "for attempt in 1 2 3" in dockerfile
    assert 'if [ "$attempt" -eq 3 ]' in dockerfile
    assert '"openai>=2.0,<3.0"' in pyproject
    assert '"alembic>=1.15,<2.0"' in pyproject
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert dockerfile.index("python -m pip install .") < dockerfile.index("COPY src ./src")
    assert "data/" in dockerignore
    assert "models/" in dockerignore
    assert "*.pdf" in dockerignore


def test_image_declares_agpl_source_version_and_revision() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG PAPER_RAG_VERSION=0.1.0" in dockerfile
    assert "ARG PAPER_RAG_REVISION=unknown" in dockerfile
    assert 'org.opencontainers.image.source="https://github.com/acemilee/metasurface-paper-RAG"' in dockerfile
    assert 'org.opencontainers.image.version="${PAPER_RAG_VERSION}"' in dockerfile
    assert 'org.opencontainers.image.revision="${PAPER_RAG_REVISION}"' in dockerfile
    assert 'org.opencontainers.image.licenses="AGPL-3.0-only"' in dockerfile
    assert "COPY LICENSE THIRD_PARTY_NOTICES.md ./" in dockerfile
