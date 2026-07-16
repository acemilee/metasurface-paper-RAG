from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class TableCandidate:
    page_number: int
    rows: list[list[str]]
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


def extract_table_candidates(
    pdf_path: Path, page_number: int
) -> list[TableCandidate]:
    candidates = []
    with pdfplumber.open(pdf_path) as document:
        for found_table in document.pages[page_number - 1].find_tables():
            table = found_table.extract()
            rows = [[(cell or "").strip() for cell in row] for row in table if row]
            populated = sum(bool(cell) for row in rows for cell in row)
            total = sum(len(row) for row in rows)
            if len(rows) < 2 or populated < 4:
                continue
            confidence = min(0.95, 0.55 + 0.4 * populated / max(total, 1))
            candidates.append(TableCandidate(page_number, rows, confidence, tuple(found_table.bbox)))
    return candidates


def table_to_markdown(table: TableCandidate) -> str:
    width = max(len(row) for row in table.rows)
    normalized = [row + [""] * (width - len(row)) for row in table.rows]
    lines = ["| " + " | ".join(row) + " |" for row in normalized]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)
