from __future__ import annotations

import re
import subprocess
from collections.abc import Callable


def parse_revision(output: str) -> str:
    revisions = re.findall(r"(?m)^([0-9A-Za-z_]+)(?:\s|$)", output.strip())
    if len(revisions) != 1:
        raise RuntimeError(
            f"Unable to parse Alembic revision from: {output!r}"
        )
    return revisions[0]


def run_migrations(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    runner(["alembic", "upgrade", "head"], check=True)
    heads = runner(
        ["alembic", "heads"], check=True, capture_output=True, text=True
    )
    current = runner(
        ["alembic", "current"], check=True, capture_output=True, text=True
    )
    head_revisions = re.findall(
        r"(?m)^([0-9A-Za-z_]+)(?:\s|$)", heads.stdout.strip()
    )
    if len(head_revisions) != 1:
        raise RuntimeError(
            f"Expected exactly one Alembic head, found {len(head_revisions)}"
        )
    current_revisions = re.findall(
        r"(?m)^([0-9A-Za-z_]+)(?:\s|$)", current.stdout.strip()
    )
    if len(current_revisions) != 1:
        raise RuntimeError(
            "Expected exactly one current Alembic revision, "
            f"found {len(current_revisions)}"
        )
    head_revision = head_revisions[0]
    current_revision = current_revisions[0]
    if current_revision != head_revision:
        raise RuntimeError(
            "Database migration check failed: "
            f"current={current_revision} head={head_revision}"
        )
    print(f"Database migration ready: {current_revision}")
    return current_revision


def main() -> int:
    run_migrations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
