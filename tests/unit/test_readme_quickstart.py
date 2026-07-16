from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readme_documents_the_complete_public_lifecycle() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "Docker Desktop",
        "Docker Compose v2",
        "start.cmd",
        "scripts\\start_services.ps1",
        "scripts/start_services.sh",
        "http://127.0.0.1:8010",
        "docker compose ps",
        "docker compose logs",
        "scripts\\stop_services.ps1",
        "首次启动",
        "约 2.3 GB",
        "8010",
        "model-init",
        "migrate",
        "/ready",
        "不会删除",
    ):
        assert required in content


def test_readme_is_the_v010_chinese_release_entrypoint() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        "# metasurface-paper-RAG",
        "v0.1.0",
        "ghcr.io/acemilee/metasurface-paper-rag:0.1.0",
        "领域正向准入",
        "review_required",
        "DeepSeek API 密钥",
        "linux/amd64",
        "AGPL-3.0-only",
        "10.1109/TAP.2020.3011115",
        "10.12000/JR23230",
    ):
        assert required in content

    assert "$env:PAPER_RAG_IMAGE='paper-rag:local'" in content


def test_readme_does_not_publish_private_eval_material() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    for forbidden in (
        "data/domain_gate_eval",
        "private_negative_pdfs",
        "POS-41",
        "NEG-13",
    ):
        assert forbidden not in content
