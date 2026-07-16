from subprocess import CompletedProcess

import pytest

from scripts.migrate_and_verify import parse_revision, run_migrations


def test_parse_revision_accepts_alembic_head_annotation() -> None:
    assert (
        parse_revision("0019_verified_formula_latex (head)\n")
        == "0019_verified_formula_latex"
    )


def test_run_migrations_upgrades_before_comparing_current_to_head() -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        commands.append(tuple(command))
        stdout = ""
        if command[-1] == "heads":
            stdout = "0019_verified_formula_latex (head)\n"
        elif command[-1] == "current":
            stdout = "0019_verified_formula_latex (head)\n"
        return CompletedProcess(command, 0, stdout=stdout, stderr="")

    assert run_migrations(runner) == "0019_verified_formula_latex"
    assert commands == [
        ("alembic", "upgrade", "head"),
        ("alembic", "heads"),
        ("alembic", "current"),
    ]


def test_run_migrations_rejects_database_behind_head() -> None:
    def runner(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        stdout = ""
        if command[-1] == "heads":
            stdout = "0019_verified_formula_latex (head)\n"
        elif command[-1] == "current":
            stdout = "0018_formula_dependencies\n"
        return CompletedProcess(command, 0, stdout=stdout, stderr="")

    with pytest.raises(
        RuntimeError,
        match=(
            "Database migration check failed: "
            "current=0018_formula_dependencies head=0019_verified_formula_latex"
        ),
    ):
        run_migrations(runner)


def test_run_migrations_rejects_multiple_heads() -> None:
    def runner(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        stdout = ""
        if command[-1] == "heads":
            stdout = "0019_alpha (head)\n0019_beta (head)\n"
        elif command[-1] == "current":
            stdout = "0019_alpha (head)\n"
        return CompletedProcess(command, 0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="Expected exactly one Alembic head"):
        run_migrations(runner)
