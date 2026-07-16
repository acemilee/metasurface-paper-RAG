from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_rag.db import SessionLocal
from paper_rag.services.formula_governance import scan_formula_inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only full-library formula inventory")
    parser.add_argument("--document-id", action="append", type=UUID, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/results/formula-library-inventory.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with SessionLocal() as session:
        report = scan_formula_inventory(session, document_ids=args.document_id)
    payload = report.as_dict()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
