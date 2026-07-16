from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_defines_the_complete_application_service_set() -> None:
    config = load_compose()
    assert set(config["services"]) == {
        "postgres",
        "model-init",
        "migrate",
        "embedding",
        "worker",
        "api",
    }


def test_compose_blocks_application_until_model_migration_and_embedding_ready() -> None:
    services = load_compose()["services"]

    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert (
        services["embedding"]["depends_on"]["model-init"]["condition"]
        == "service_completed_successfully"
    )
    for service_name in ("worker", "api"):
        dependencies = services[service_name]["depends_on"]
        assert dependencies["migrate"]["condition"] == "service_completed_successfully"
        assert dependencies["embedding"]["condition"] == "service_healthy"


def test_compose_exposes_only_loopback_gui_and_keeps_embedding_internal() -> None:
    services = load_compose()["services"]

    assert services["api"]["ports"] == ["127.0.0.1:8010:8010"]
    assert "ports" not in services["embedding"]
    assert services["postgres"]["ports"] == ["127.0.0.1:5433:5432"]


def test_compose_persists_state_and_restarts_only_long_running_services() -> None:
    services = load_compose()["services"]

    for service_name in ("postgres", "embedding", "worker", "api"):
        assert services[service_name]["restart"] == "unless-stopped"
    for service_name in ("model-init", "migrate"):
        assert "restart" not in services[service_name]

    assert "paper_rag_postgres:/var/lib/postgresql/data" in services["postgres"]["volumes"]
    assert "./models:/app/models" in services["model-init"]["volumes"]
    assert "./data:/app/data" in services["worker"]["volumes"]
    assert "./data:/app/data" in services["api"]["volumes"]


def test_compose_uses_container_addresses_and_readiness_healthchecks() -> None:
    services = load_compose()["services"]
    api_environment = services["api"]["environment"]

    assert "@postgres:5432/" in api_environment["PAPER_RAG_POSTGRES_DSN"]
    assert api_environment["PAPER_RAG_EMBEDDING_SERVICE_URL"] == "http://embedding:8011"
    assert "/ready" in " ".join(services["api"]["healthcheck"]["test"])
    assert "/health" in " ".join(services["embedding"]["healthcheck"]["test"])


def test_compose_defaults_to_fixed_ghcr_image_and_allows_override() -> None:
    content = COMPOSE.read_text(encoding="utf-8")
    config = load_compose()

    assert "${PAPER_RAG_IMAGE:-ghcr.io/acemilee/metasurface-paper-rag:0.1.0}" in content
    assert config["x-app"]["build"] == {"context": "."}
