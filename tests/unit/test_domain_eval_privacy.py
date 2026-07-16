from pathlib import Path


PRIVATE_EVAL_ROOT = "data/domain_gate_eval/"


def test_private_domain_eval_pdfs_are_excluded_from_release_inputs() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert PRIVATE_EVAL_ROOT in gitignore.splitlines()
    assert PRIVATE_EVAL_ROOT in dockerignore.splitlines()
    assert "data/" in dockerignore.splitlines()
    assert "*.pdf" in dockerignore.splitlines()
