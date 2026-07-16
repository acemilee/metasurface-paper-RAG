from pathlib import Path
import re


def test_alembic_revision_ids_fit_version_table() -> None:
    root = Path(__file__).resolve().parents[2] / "alembic/versions"
    revisions = []
    for path in root.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        match = re.search(r'^revision\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert match, f"Missing revision in {path.name}"
        revisions.append(match.group(1))

    assert all(len(revision) <= 32 for revision in revisions)
    assert len(revisions) == len(set(revisions))


def test_filename_search_migration_backfills_before_not_null() -> None:
    path = Path(__file__).resolve().parents[2] / "alembic/versions/0020_filename_search_key.py"
    content = path.read_text(encoding="utf-8")

    assert 'revision = "0020_filename_search_key"' in content
    assert 'down_revision = "0019_verified_formula_latex"' in content
    assert 'op.add_column("documents"' in content
    assert "normalize_filename_search_key" in content
    assert 'op.alter_column("documents", "filename_search_key", nullable=False)' in content
    assert content.index("normalize_filename_search_key") < content.index("nullable=False")
