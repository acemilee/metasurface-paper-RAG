from pathlib import Path

import pytest

from scripts.bootstrap_model import model_is_complete


def test_model_is_complete_requires_sentence_transformer_metadata_and_weights(
    tmp_path: Path,
) -> None:
    target = tmp_path / "BAAI-bge-m3"
    target.mkdir()
    (target / "modules.json").write_text("[]", encoding="utf-8")
    (target / "config.json").write_text("{}", encoding="utf-8")
    assert model_is_complete(target) is False

    (target / "model.safetensors").write_bytes(b"weights")
    assert model_is_complete(target) is True


def test_ensure_model_downloads_to_staging_then_replaces_target(
    tmp_path: Path,
) -> None:
    from scripts.bootstrap_model import ensure_model

    target = tmp_path / "BAAI-bge-m3"

    def fake_download(*, repo_id: str, local_dir: str) -> None:
        assert repo_id == "BAAI/bge-m3"
        staging = Path(local_dir)
        (staging / "modules.json").write_text("[]", encoding="utf-8")
        (staging / "config.json").write_text("{}", encoding="utf-8")
        (staging / "model.safetensors").write_bytes(b"weights")

    assert ensure_model(target, "BAAI/bge-m3", fake_download) == "downloaded"
    assert (target / "model.safetensors").read_bytes() == b"weights"
    assert not (tmp_path / "BAAI-bge-m3.partial").exists()


def test_ensure_model_reuses_complete_model_without_downloading(
    tmp_path: Path,
) -> None:
    from scripts.bootstrap_model import ensure_model

    target = tmp_path / "BAAI-bge-m3"
    target.mkdir()
    (target / "modules.json").write_text("[]", encoding="utf-8")
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"weights")

    def unexpected_download(**_kwargs: str) -> None:
        raise AssertionError("complete models must not be downloaded again")

    assert ensure_model(target, "BAAI/bge-m3", unexpected_download) == "ready"


def test_ensure_model_removes_incomplete_staging_on_failure(tmp_path: Path) -> None:
    from scripts.bootstrap_model import ensure_model

    target = tmp_path / "BAAI-bge-m3"

    def incomplete_download(*, repo_id: str, local_dir: str) -> None:
        assert repo_id == "BAAI/bge-m3"
        (Path(local_dir) / "modules.json").write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        ensure_model(target, "BAAI/bge-m3", incomplete_download)

    assert not target.exists()
    assert not (tmp_path / "BAAI-bge-m3.partial").exists()
