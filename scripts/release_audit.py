from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PUBLIC_ROOT_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "alembic.ini",
    "CONTRIBUTING.md",
    "docker-compose.yml",
    "Dockerfile",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "SECURITY.md",
    "start.cmd",
    "THIRD_PARTY_NOTICES.md",
}
PUBLIC_DIRS = {".github", "alembic", "scripts", "src", "tests"}
PRIVATE_TEST_FILES = {
    Path("tests/unit/test_domain_admission_acceptance.py"),
    Path("tests/unit/test_formula_phase_f_acceptance.py"),
}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".whl",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".db",
    ".sqlite",
    ".log",
}
FORBIDDEN_NAMES = {".env", "handoff.md", "mvp_tasks.md", "概述.txt"}
FORBIDDEN_ROOTS = {"data", "docs", "evals", "models", "papers", "release-artifacts"}


@dataclass(frozen=True)
class Violation:
    path: Path
    code: str
    detail: str


def collect_public_paths(root: Path) -> list[Path]:
    paths = [Path(name) for name in PUBLIC_ROOT_FILES if (root / name).is_file()]
    for directory in sorted(PUBLIC_DIRS):
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not path.is_symlink()
                and relative not in PRIVATE_TEST_FILES
                and not (set(relative.parts) & SKIP_PARTS)
                and path.suffix != ".pyc"
            ):
                paths.append(relative)
    return sorted(set(paths), key=lambda item: item.as_posix())


def _path_is_forbidden(path: Path) -> bool:
    return (
        path.name.lower() in FORBIDDEN_NAMES
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
        or bool(path.parts and path.parts[0].lower() in FORBIDDEN_ROOTS)
        or (
            path.name.lower().startswith(".env.")
            and path.name.lower() != ".env.example"
        )
    )


def audit_paths(root: Path, paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    secret_prefix = ("s" + "k-").encode()
    private_key_marker = ("BEGIN " + "PRIVATE KEY").encode()
    for relative in sorted(set(paths), key=lambda item: item.as_posix()):
        if relative.is_absolute() or ".." in relative.parts or _path_is_forbidden(relative):
            violations.append(
                Violation(
                    relative,
                    "forbidden_path",
                    "path is outside the public release contract",
                )
            )
            continue
        path = root / relative
        if path.is_symlink():
            violations.append(
                Violation(relative, "forbidden_path", "symbolic links are not released")
            )
            continue
        if not path.is_file():
            violations.append(
                Violation(relative, "missing", "inventory path is not a file")
            )
            continue
        data = path.read_bytes()
        if (
            re.search(
                rb"\b" + re.escape(secret_prefix) + rb"[A-Za-z0-9_-]{16,}\b",
                data,
            )
            or private_key_marker in data
        ):
            violations.append(
                Violation(relative, "credential", "credential-like content detected")
            )
    return violations


def _read_path_file(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_paths(root: Path, *arguments: str) -> list[Path]:
    result = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        Path(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--paths-file", type=Path, required=True)
    subparsers.add_parser("check-staged")
    subparsers.add_parser("check-tracked")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    if args.command == "inventory":
        paths = collect_public_paths(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(f"{path.as_posix()}\n" for path in paths),
            encoding="utf-8",
        )
    elif args.command == "check":
        paths = _read_path_file(args.paths_file)
    elif args.command == "check-staged":
        paths = _git_paths(root, "diff", "--cached", "--name-only")
    else:
        paths = _git_paths(root, "ls-files")

    violations = audit_paths(root, paths)
    for item in violations:
        print(
            f"{item.path.as_posix()}: {item.code}: {item.detail}",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
