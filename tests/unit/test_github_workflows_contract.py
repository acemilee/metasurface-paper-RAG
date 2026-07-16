from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return yaml.safe_load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
    )


def test_quality_workflow_is_reusable_and_tests_312_313() -> None:
    workflow = load("quality.yml")
    triggers = workflow[True]
    matrix = workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"]

    assert "workflow_call" in triggers
    assert "push" in triggers
    assert "pull_request" in triggers
    assert set(map(str, matrix)) == {"3.12", "3.13"}


def test_release_waits_for_quality_and_pushes_agpl_amd64_image() -> None:
    workflow = load("release.yml")
    assert workflow[True]["push"]["tags"] == ["v*"]
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == "quality"
    assert workflow["permissions"] == {
        "contents": "write",
        "packages": "write",
        "id-token": "write",
        "attestations": "write",
    }
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "linux/amd64" in text
    assert "ghcr.io/acemilee/metasurface-paper-rag" in text
    assert "AGPL-3.0-only" in text
    assert "PAPER_RAG_REVISION" in text
    assert "actions/attest-build-provenance" in text


def test_image_audit_runs_the_published_digest_from_a_clean_clone() -> None:
    workflow = load("image-audit.yml")
    text = (ROOT / ".github/workflows/image-audit.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch" in workflow[True]
    assert workflow["permissions"] == {"contents": "read"}
    assert "sha256:a78ed0769aaf6b5a0d2a09eb7a4a86904d9f0dbf08b7df2ade5d59b07523210f" in text
    assert "git clone --branch v0.1.0 --depth 1" in text
    assert "python scripts/release_audit.py check-tracked" in text
    assert "import fitz; import paper_rag" in text
    assert "docker export" in text
    assert "^app/.*\\.log$" in text
    assert "scripts/start_services.sh --no-browser" in text
    assert "if: always()" in text
