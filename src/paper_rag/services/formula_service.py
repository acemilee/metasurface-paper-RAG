from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from uuid import UUID, uuid5

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from paper_rag.models.formula import Formula
from paper_rag.models.page import Page
from paper_rag.services.pdf_parser import ParsedPage, ParsedTextBlock


MATH_MARKERS = re.compile(r"[=¼≈≅≤≥∑∫√∂±×÷^_]|[\x00-\x08\x0b\x0c\x0e-\x1f]")
FORMULA_NUMBER = re.compile(r"\((?P<number>\d{1,3})(?P<suffix>[a-z]?)\)\s*$", re.IGNORECASE)
FORMULA_PARSER_VERSION = "formula-layout-v3"


def _normalize_ligatures(text: str) -> str:
    return text.replace("ﬁ", "fi").replace("ﬂ", "fl")


@dataclass(frozen=True)
class FormulaCandidate:
    page_number: int
    bbox: tuple[float, float, float, float]
    raw_text: str
    context_before: str
    context_after: str
    formula_number: str | None = None
    group_key: str | None = None
    part_index: int = 0
    fidelity_status: str = "needs_review"


def _is_prose_block(block: ParsedTextBlock) -> bool:
    compact = block.text.replace("\n", " ").strip()
    return len(compact) > 220 or len(re.findall(r"[A-Za-z]{3,}", compact)) > 24


def _is_formula_fragment(block: ParsedTextBlock) -> bool:
    compact = block.text.strip()
    if not compact or _is_prose_block(block):
        return False
    return bool(
        FORMULA_NUMBER.search(compact)
        or MATH_MARKERS.search(compact)
        or len(compact) <= 10
    )


def _is_context_block(block: ParsedTextBlock) -> bool:
    compact = block.text.replace("\n", " ").strip()
    return not _is_formula_fragment(block) and len(re.findall(r"[A-Za-z]{3,}", compact)) >= 4


def _merge_spatial_text(blocks: list[ParsedTextBlock]) -> str:
    rows: list[list[ParsedTextBlock]] = []
    for block in sorted(blocks, key=lambda item: (item.y0, item.x0)):
        matching_row = next(
            (
                row
                for row in rows
                if block.y0 <= max(item.y1 for item in row)
                and block.y1 >= min(item.y0 for item in row)
            ),
            None,
        )
        if matching_row is None:
            rows.append([block])
        else:
            matching_row.append(block)
    return "\n".join(
        " ".join(item.text.strip() for item in sorted(row, key=lambda item: item.x0))
        for row in rows
    ).strip()


def _vertically_connected(
    anchor: ParsedTextBlock,
    blocks: list[ParsedTextBlock],
    *,
    max_gap: float = 12.0,
) -> list[ParsedTextBlock]:
    connected = [anchor]
    remaining = [block for block in blocks if block is not anchor]
    changed = True
    while changed:
        changed = False
        for block in remaining.copy():
            if any(
                max(0.0, max(block.y0, item.y0) - min(block.y1, item.y1)) <= max_gap
                for item in connected
            ):
                connected.append(block)
                remaining.remove(block)
                changed = True
    return connected


def _nearest_context(
    page: ParsedPage,
    formula_blocks: list[ParsedTextBlock],
    *,
    region_top: float = float("-inf"),
    region_bottom: float = float("inf"),
    formula_number: str | None = None,
) -> tuple[str, str]:
    top = min(block.y0 for block in formula_blocks)
    bottom = max(block.y1 for block in formula_blocks)
    page_width = max((block.x1 for block in page.blocks), default=1.0)
    same_column_prose = [
        block
        for block in page.blocks
        if _is_context_block(block)
        and _same_column(formula_blocks, block, page_width)
    ]
    prose = [
        block
        for block in same_column_prose
        if region_top < (block.y0 + block.y1) / 2 <= region_bottom
    ]
    if formula_number:
        explicit = re.compile(
            rf"(?:Equation|Eq\.)\s*\(\s*{re.escape(formula_number)}\s*\)",
            re.IGNORECASE,
        )
        referenced = [block for block in same_column_prose if explicit.search(block.text)]
        prose.extend(block for block in referenced if block not in prose)
    before = max(
        (block for block in prose if block.y1 <= top),
        key=lambda block: block.y1,
        default=None,
    )
    after = min(
        (block for block in prose if block.y0 >= bottom),
        key=lambda block: block.y0,
        default=None,
    )
    return before.text if before else "", after.text if after else ""


def _same_column(
    formula_blocks: list[ParsedTextBlock],
    context: ParsedTextBlock,
    page_width: float,
) -> bool:
    left = min(block.x0 for block in formula_blocks)
    right = max(block.x1 for block in formula_blocks)
    overlap = max(0.0, min(right, context.x1) - max(left, context.x0))
    formula_width = max(1.0, right - left)
    if overlap / formula_width >= 0.25:
        return True
    formula_center = (left + right) / 2
    context_center = (context.x0 + context.x1) / 2
    return (formula_center < page_width / 2) == (context_center < page_width / 2)


def _valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    return all(math.isfinite(value) for value in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _bbox_overlap_ratio(left: FormulaCandidate, right: FormulaCandidate) -> float:
    x0 = max(left.bbox[0], right.bbox[0])
    y0 = max(left.bbox[1], right.bbox[1])
    x1 = min(left.bbox[2], right.bbox[2])
    y1 = min(left.bbox[3], right.bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(
        (left.bbox[2] - left.bbox[0]) * (left.bbox[3] - left.bbox[1]),
        (right.bbox[2] - right.bbox[0]) * (right.bbox[3] - right.bbox[1]),
    )
    return intersection / max(1.0, smaller)


def _validate_candidates(candidates: list[FormulaCandidate]) -> list[FormulaCandidate]:
    accepted: list[FormulaCandidate] = []
    seen_numbers: set[tuple[int, str]] = set()
    seen_parts: set[tuple[int, str, int]] = set()
    for candidate in candidates:
        if not candidate.raw_text.strip() or not _valid_bbox(candidate.bbox):
            continue
        number_key = (candidate.page_number, candidate.formula_number or "")
        part_key = (candidate.page_number, candidate.group_key or "", candidate.part_index)
        if candidate.formula_number and number_key in seen_numbers:
            continue
        if candidate.group_key and part_key in seen_parts:
            continue
        if any(_bbox_overlap_ratio(candidate, previous) >= 0.85 for previous in accepted):
            continue
        accepted.append(candidate)
        if candidate.formula_number:
            seen_numbers.add(number_key)
        if candidate.group_key:
            seen_parts.add(part_key)
    return accepted


def detect_formula_regions(page: ParsedPage) -> list[FormulaCandidate]:
    candidates: list[FormulaCandidate] = []
    anchors = [
        (index, block, FORMULA_NUMBER.search(block.text.strip()))
        for index, block in enumerate(page.blocks)
        if FORMULA_NUMBER.search(block.text.strip())
    ]
    max_x = max((block.x1 for block in page.blocks), default=1.0)
    used_orders: set[int] = set()
    for anchor_index, (_, anchor, match) in enumerate(anchors):
        center = (anchor.y0 + anchor.y1) / 2
        previous_center = (
            (anchors[anchor_index - 1][1].y0 + anchors[anchor_index - 1][1].y1) / 2
            if anchor_index
            else float("-inf")
        )
        next_center = (
            (anchors[anchor_index + 1][1].y0 + anchors[anchor_index + 1][1].y1) / 2
            if anchor_index + 1 < len(anchors)
            else float("inf")
        )
        top = (previous_center + center) / 2 if anchor_index else float("-inf")
        bottom = (center + next_center) / 2 if anchor_index + 1 < len(anchors) else float("inf")
        left_column = anchor.x1 <= max_x * 0.6
        formula_blocks = [
            block
            for block in page.blocks
            if top < (block.y0 + block.y1) / 2 <= bottom
            and ((left_column and block.x1 <= max_x * 0.62) or (not left_column and block.x0 >= max_x * 0.38))
            and _is_formula_fragment(block)
        ]
        formula_blocks = _vertically_connected(anchor, formula_blocks)
        used_orders.update(block.reading_order for block in formula_blocks)
        raw_text = _merge_spatial_text(formula_blocks)
        number = f"{match.group('number')}{match.group('suffix').lower()}"
        before, after = _nearest_context(
            page,
            formula_blocks,
            region_top=top,
            region_bottom=bottom,
            formula_number=number,
        )
        suffix = match.group("suffix").lower()
        candidates.append(
            FormulaCandidate(
                page.page_number,
                (
                    min(block.x0 for block in formula_blocks),
                    min(block.y0 for block in formula_blocks),
                    max(block.x1 for block in formula_blocks),
                    max(block.y1 for block in formula_blocks),
                ),
                raw_text,
                before,
                after,
                number,
                f"equation-{match.group('number')}",
                ord(suffix) - ord("a") if suffix else 0,
                "needs_review" if MATH_MARKERS.search(raw_text) else "source_exact",
            )
        )

    for index, block in enumerate(page.blocks):
        if block.reading_order in used_orders:
            continue
        compact = block.text.replace("\n", " ").strip()
        is_short_display_line = len(compact) <= 220 and (compact.count(" ") <= 18 or bool(FORMULA_NUMBER.search(compact)))
        is_math_like = (len(MATH_MARKERS.findall(compact)) >= 1 and is_short_display_line) or bool(FORMULA_NUMBER.search(compact))
        if not is_math_like:
            continue
        before = page.blocks[index - 1].text if index else ""
        after = page.blocks[index + 1].text if index + 1 < len(page.blocks) else ""
        candidates.append(FormulaCandidate(page.page_number, (block.x0, block.y0, block.x1, block.y1), compact, before, after))
    return _validate_candidates(candidates)


def formula_placeholder(formula_id: UUID) -> str:
    return f"公式_placeholder_{formula_id}"


def normalize_formula_text(raw_text: str) -> str:
    translations = str.maketrans(
        {
            "¼": "=",
            "\x01": "/",
            "\x03": "-",
            "\x04": "[",
            "\x05": "]",
            "\x06": "",
            "\x07": "|",
            "\x08": "",
        }
    )
    normalized = _normalize_ligatures(raw_text).translate(translations)
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", normalized)
    return "\n".join(" ".join(line.split()) for line in normalized.splitlines() if line.strip())


def _candidate_identity(candidate: FormulaCandidate) -> str:
    stable = [
        FORMULA_PARSER_VERSION,
        str(candidate.page_number),
        candidate.group_key or "ungrouped",
    ]
    if candidate.formula_number:
        stable.append(candidate.formula_number.lower())
    else:
        stable.extend(
            [
                normalize_formula_text(candidate.raw_text)[:160],
                ",".join(f"{round(value / 2) * 2:.0f}" for value in candidate.bbox),
            ]
        )
    return "|".join(stable)


def build_formula_semantic_card(candidate: FormulaCandidate) -> tuple[str | None, str]:
    for evidence in (candidate.context_after, candidate.context_before):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", evidence)
            if sentence.strip()
        ]
        grounded = [
            sentence
            for sentence in sentences
            if any(
                token in sentence.lower()
                for token in (
                    "represent",
                    "denote",
                    "where",
                    "coefficient",
                    "impedance",
                    "capacitance",
                    "resistance",
                    "wavelength",
                )
            )
        ]
        if grounded:
            return _normalize_ligatures(" ".join(grounded[:2])), "grounded"
    return None, "insufficient_context"


def create_formula_records(document_id: UUID, page: ParsedPage) -> list[Formula]:
    records: list[Formula] = []
    for candidate in detect_formula_regions(page):
        formula_id = uuid5(document_id, _candidate_identity(candidate))
        meaning, semantic_status = build_formula_semantic_card(candidate)
        records.append(
            Formula(
                id=formula_id,
                document_id=document_id,
                page_number=candidate.page_number,
                placeholder=formula_placeholder(formula_id),
                bbox_json=json.dumps(candidate.bbox),
                raw_text=candidate.raw_text,
                context_before=candidate.context_before,
                context_after=candidate.context_after,
                physical_meaning=meaning,
                semantic_status=semantic_status,
                formula_number=candidate.formula_number,
                group_key=candidate.group_key,
                part_index=candidate.part_index,
                parser_version=FORMULA_PARSER_VERSION,
                normalized_text=normalize_formula_text(candidate.raw_text),
                fidelity_status=candidate.fidelity_status,
            )
        )
    return records


def replace_formula_regions_with_placeholders(page: ParsedPage, formulas: list[Formula]) -> str:
    replacements = {formula.raw_text: formula.placeholder for formula in formulas if formula.raw_text}
    text = page.text
    for raw_text, placeholder in replacements.items():
        text = text.replace(raw_text, placeholder, 1)
    return text


def _load_requested_pages(
    session: Session,
    document_id: UUID,
    page_numbers: list[int],
) -> list[Page]:
    requested_pages = sorted(set(page_numbers))
    if not requested_pages:
        return []
    pages = list(
        session.scalars(
            select(Page)
            .options(selectinload(Page.blocks))
            .where(
                Page.document_id == document_id,
                Page.page_number.in_(requested_pages),
            )
            .order_by(Page.page_number)
        )
    )
    found_pages = {page.page_number for page in pages}
    missing_pages = [number for number in requested_pages if number not in found_pages]
    if missing_pages:
        raise ValueError(f"Parsed pages not found: {missing_pages}")
    return pages


def _to_parsed_page(page: Page) -> ParsedPage:
    return ParsedPage(
        page_number=page.page_number,
        text=page.text,
        blocks=[
            ParsedTextBlock(
                page.page_number,
                block.reading_order,
                block.text,
                block.x0,
                block.y0,
                block.x1,
                block.y1,
                block.source,
                block.confidence,
            )
            for block in sorted(page.blocks, key=lambda item: item.reading_order)
        ],
        extraction_method=page.extraction_method,
        quality_score=page.quality_score,
        ocr_confidence=page.ocr_confidence,
    )


def build_formula_records_for_pages(
    session: Session,
    document_id: UUID,
    page_numbers: list[int],
) -> list[Formula]:
    return [
        record
        for page in _load_requested_pages(session, document_id, page_numbers)
        for record in create_formula_records(document_id, _to_parsed_page(page))
    ]


def rebuild_formula_pages(
    session: Session,
    document_id: UUID,
    page_numbers: list[int],
) -> list[Formula]:
    requested_pages = sorted(set(page_numbers))
    if not requested_pages:
        return []
    records = build_formula_records_for_pages(session, document_id, requested_pages)

    session.execute(
        delete(Formula).where(
            Formula.document_id == document_id,
            Formula.page_number.in_(requested_pages),
        )
    )
    session.add_all(records)
    session.commit()
    return records
