from __future__ import annotations

import re
from dataclasses import dataclass


CAPTION_PATTERN = re.compile(
    r"^(?:fig(?:ure)?\.?|table)\s*\d+[.:]?\s+", re.IGNORECASE
)


@dataclass(frozen=True)
class Caption:
    page_number: int
    text: str
    bbox: tuple[float, float, float, float]


def extract_figure_captions(blocks) -> list[Caption]:
    return [
        Caption(block.page_number, block.text, (block.x0, block.y0, block.x1, block.y1))
        for block in blocks
        if CAPTION_PATTERN.match(block.text.strip())
    ]
