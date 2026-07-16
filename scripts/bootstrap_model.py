from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path


REQUIRED_FILES = ("modules.json", "config.json")


def model_is_complete(path: Path) -> bool:
    has_metadata = path.is_dir() and all(
        (path / name).is_file() for name in REQUIRED_FILES
    )
    has_weights = (
        any(path.glob("*.safetensors"))
        or (path / "pytorch_model.bin").is_file()
    )
    return has_metadata and has_weights


def ensure_model(
    target: Path,
    repo_id: str,
    download: Callable[..., object],
) -> str:
    if model_is_complete(target):
        return "ready"

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        download(repo_id=repo_id, local_dir=str(staging))
        if not model_is_complete(staging):
            raise RuntimeError("Downloaded embedding model is incomplete")
        shutil.rmtree(target, ignore_errors=True)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return "downloaded"


def main() -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(
        description="Prepare the local sentence-transformer model atomically."
    )
    parser.add_argument("--repo-id", default="BAAI/bge-m3")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    result = ensure_model(args.target, args.repo_id, snapshot_download)
    print(f"Embedding model {result}: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
