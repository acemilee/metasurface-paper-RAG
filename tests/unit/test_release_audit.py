from pathlib import Path

from scripts.release_audit import audit_paths, collect_public_paths


def test_public_inventory_excludes_private_and_generated_state(tmp_path: Path) -> None:
    for relative in (
        "src/paper_rag/main.py",
        "README.md",
        "data/uploads/private.pdf",
        "papers/private.pdf",
        "models/model.bin",
        "docs/superpowers/plans/internal.md",
        "HANDOFF.md",
        "tests/__pycache__/test_x.pyc",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"public")

    paths = collect_public_paths(tmp_path)

    assert Path("src/paper_rag/main.py") in paths
    assert Path("README.md") in paths
    assert Path("data/uploads/private.pdf") not in paths
    assert Path("papers/private.pdf") not in paths
    assert Path("models/model.bin") not in paths
    assert Path("docs/superpowers/plans/internal.md") not in paths
    assert Path("HANDOFF.md") not in paths
    assert Path("tests/__pycache__/test_x.pyc") not in paths


def test_public_inventory_excludes_tests_that_require_private_evals(
    tmp_path: Path,
) -> None:
    private_tests = (
        Path("tests/unit/test_domain_admission_acceptance.py"),
        Path("tests/unit/test_formula_phase_f_acceptance.py"),
    )
    for relative in private_tests:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from evals import private_runner\n", encoding="utf-8")

    paths = collect_public_paths(tmp_path)

    assert not set(private_tests) & set(paths)


def test_audit_rejects_pdf_env_and_realistic_secrets(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    env = tmp_path / ".env"
    secret = tmp_path / "config.txt"
    pdf.write_bytes(b"%PDF")
    env.write_text("X=1", encoding="utf-8")
    secret.write_text("DEEPSEEK=" + "s" + "k-" + "A" * 32, encoding="utf-8")

    violations = audit_paths(
        tmp_path, [Path("paper.pdf"), Path(".env"), Path("config.txt")]
    )

    assert {item.code for item in violations} == {"forbidden_path", "credential"}
