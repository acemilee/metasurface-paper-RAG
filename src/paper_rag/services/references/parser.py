from __future__ import annotations

import re
import unicodedata

from paper_rag.services.references.types import (
    ReferenceKind,
    ReferenceSource,
    TypedReference,
)


FORMULA_REFERENCE = re.compile(
    r"(?:"
    r"(?:公式|方程|(?<![模形样格])式)[^\S\r\n]*[（(]?[^\S\r\n]*"
    r"(?P<cn>\d+[a-z]?)(?![a-z0-9])[^\S\r\n]*[）)]?"
    r"|"
    r"\beq(?:uation)?\.?[^\S\r\n]*[（(]?[^\S\r\n]*"
    r"(?P<en>\d+[a-z]?)(?![a-z0-9])[^\S\r\n]*[）)]?"
    r")",
    re.IGNORECASE,
)
FIGURE_REFERENCE = re.compile(
    r"(?:图|\bfig(?:ure)?\.?)[^\S\r\n]*(?P<number>\d+)(?![a-z0-9])"
    r"(?:[^\S\r\n]*[（(](?P<part>[a-z])[^\S\r\n]*[）)])?",
    re.IGNORECASE,
)
TABLE_REFERENCE = re.compile(
    r"(?:(?<!代)表|\btable)[^\S\r\n]*"
    r"(?P<number>\d+|[ivxlc]{1,6})(?![a-z0-9])",
    re.IGNORECASE,
)
SECTION_REFERENCE = re.compile(
    r"(?:第[^\S\r\n]*(?P<cn>\d+(?:\.\d+)*)(?![a-z0-9])[^\S\r\n]*(?:节|章)|"
    r"\bsection[^\S\r\n]*(?P<en>\d+(?:\.\d+)*)(?![a-z0-9]))",
    re.IGNORECASE,
)
PAGE_REFERENCE = re.compile(
    r"(?:第[^\S\r\n]*(?P<cn>\d+)(?![a-z0-9])[^\S\r\n]*页|"
    r"\bpage[^\S\r\n]*(?P<en>\d+)(?![a-z0-9]))",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE = re.compile(
    r"(?:"
    r"(?:论文|文档)[^\S\r\n]*[《\"“](?P<cn>[^》\"”]{1,512})[》\"”]"
    r"|"
    r"\b(?:document|paper)[^\S\r\n]*[\"“](?P<en>[^\"”]{1,512})[\"”]"
    r")",
    re.IGNORECASE,
)

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _matched_number(match: re.Match[str]) -> str:
    return (match.group("cn") or match.group("en")).lower()


def roman_to_int(value: str) -> int:
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = ROMAN_VALUES[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def _parse_all(text: str, source: ReferenceSource) -> list[TypedReference]:
    normalized = unicodedata.normalize("NFKC", text)
    found: list[tuple[int, TypedReference]] = []

    for match in FORMULA_REFERENCE.finditer(normalized):
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.FORMULA,
                    surface=match.group(0),
                    normalized_key=_matched_number(match),
                    qualifier=None,
                    source=source,
                ),
            )
        )
    for match in FIGURE_REFERENCE.finditer(normalized):
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.FIGURE,
                    surface=match.group(0),
                    normalized_key=match.group("number"),
                    qualifier=(match.group("part") or "").lower() or None,
                    source=source,
                ),
            )
        )
    for match in TABLE_REFERENCE.finditer(normalized):
        raw_number = match.group("number")
        key = raw_number if raw_number.isdigit() else str(roman_to_int(raw_number))
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.TABLE,
                    surface=match.group(0),
                    normalized_key=key,
                    qualifier=None,
                    source=source,
                ),
            )
        )
    for match in SECTION_REFERENCE.finditer(normalized):
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.SECTION,
                    surface=match.group(0),
                    normalized_key=match.group("cn") or match.group("en"),
                    qualifier=None,
                    source=source,
                ),
            )
        )
    for match in PAGE_REFERENCE.finditer(normalized):
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.PAGE,
                    surface=match.group(0),
                    normalized_key=match.group("cn") or match.group("en"),
                    qualifier=None,
                    source=source,
                ),
            )
        )
    for match in DOCUMENT_REFERENCE.finditer(normalized):
        title = (match.group("cn") or match.group("en")).strip().casefold()
        found.append(
            (
                match.start(),
                TypedReference(
                    kind=ReferenceKind.DOCUMENT,
                    surface=match.group(0),
                    normalized_key=title,
                    qualifier=None,
                    source=source,
                ),
            )
        )

    seen: set[tuple[ReferenceKind, str, str | None]] = set()
    result: list[TypedReference] = []
    for _, reference in sorted(found, key=lambda item: item[0]):
        identity = (reference.kind, reference.normalized_key, reference.qualifier)
        if identity not in seen:
            seen.add(identity)
            result.append(reference)
    return result


def parse_typed_references(
    original_question: str,
    standalone_question: str = "",
) -> tuple[TypedReference, ...]:
    original = _parse_all(
        original_question,
        ReferenceSource.ORIGINAL_QUESTION,
    )
    if original:
        return tuple(original)
    return tuple(
        _parse_all(
            standalone_question,
            ReferenceSource.STANDALONE_QUESTION,
        )
    )
