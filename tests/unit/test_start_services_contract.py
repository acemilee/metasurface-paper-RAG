from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
START_PS1 = ROOT / "scripts" / "start_services.ps1"
STOP_PS1 = ROOT / "scripts" / "stop_services.ps1"
START_SH = ROOT / "scripts" / "start_services.sh"
STOP_SH = ROOT / "scripts" / "stop_services.sh"
START_CMD = ROOT / "start.cmd"


def test_windows_launcher_uses_compose_and_reports_ready_only_after_success() -> None:
    content = START_PS1.read_text(encoding="utf-8")

    assert "[switch]$NoBrowser" in content
    assert "[switch]$BuildLocal" in content
    assert "[string]$BuildProxy" in content
    assert "PAPER_RAG_BUILD_PROXY" in content
    assert "host.docker.internal" in content
    assert "$WaitTimeoutSeconds = 1800" in content
    assert "Get-Command docker" in content
    assert "docker compose version" in content
    assert "docker build --tag paper-rag:local ." in content
    assert "ghcr.io/acemilee/metasurface-paper-rag:0.1.0" in content
    assert "docker pull" in content
    assert "PAPER_RAG_IMAGE" in content
    assert '"--build-arg", "HTTP_PROXY=$BuildProxy"' in content
    assert '"--build-arg", "HTTPS_PROXY=$BuildProxy"' in content
    assert "docker compose up --detach --no-build --wait --wait-timeout" in content
    assert "docker compose ps" in content
    assert "docker compose logs --tail" in content
    assert "Invoke-RestMethod \"http://127.0.0.1:8010/ready\"" in content
    assert "GUI ready: http://127.0.0.1:8010" in content
    assert "F:\\python313" not in content
    assert "paper_rag.main:app" not in content
    assert content.index("Invoke-RestMethod") < content.index("GUI ready:")


def test_stop_launchers_are_non_destructive_compose_wrappers() -> None:
    for path in (STOP_PS1, STOP_SH):
        content = path.read_text(encoding="utf-8")
        assert "docker compose down" in content
        assert "down -v" not in content
        assert "volume rm" not in content
        assert "Stop-Process" not in content


def test_posix_and_cmd_launchers_delegate_to_the_same_compose_contract() -> None:
    shell = START_SH.read_text(encoding="utf-8")
    cmd = START_CMD.read_text(encoding="utf-8")

    assert "set -eu" in shell
    assert "--no-browser" in shell
    assert "--build-local" in shell
    assert "--wait-timeout" in shell
    assert "docker compose version" in shell
    assert "docker build --tag paper-rag:local ." in shell
    assert "ghcr.io/acemilee/metasurface-paper-rag:0.1.0" in shell
    assert "docker pull" in shell
    assert "PAPER_RAG_IMAGE" in shell
    assert "PAPER_RAG_BUILD_PROXY" in shell
    assert "--build-arg HTTP_PROXY" in shell
    assert "docker compose up --detach --no-build --wait --wait-timeout" in shell
    assert "docker compose ps" in shell
    assert "docker compose logs --tail" in shell
    assert "/ready" in shell
    assert "GUI ready: http://127.0.0.1:8010" in shell

    assert "scripts\\start_services.ps1" in cmd
    assert "if errorlevel 1" in cmd.lower()
