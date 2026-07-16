from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID


_SENSITIVE_DIAGNOSTIC_KEYS = ("reasoning", "api_key", "secret", "token")


def _safe_diagnostic_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _safe_diagnostic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not any(
                marker in str(key).lower()
                for marker in _SENSITIVE_DIAGNOSTIC_KEYS
            )
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_diagnostic_value(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str) and "sk-" in value.lower():
        return "[redacted]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ReferenceKind(StrEnum):
    FORMULA = "formula"
    FIGURE = "figure"
    TABLE = "table"
    SECTION = "section"
    PAGE = "page"
    DOCUMENT = "document"


class ReferenceSource(StrEnum):
    ORIGINAL_QUESTION = "original_question"
    STANDALONE_QUESTION = "standalone_question"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"
    INVALID = "invalid"
    INDEX_INCONSISTENT = "index_inconsistent"


@dataclass(frozen=True)
class TypedReference:
    kind: ReferenceKind
    surface: str
    normalized_key: str
    qualifier: str | None
    source: ReferenceSource


@dataclass(frozen=True)
class ReferenceResolution:
    reference: TypedReference
    status: ResolutionStatus
    document_ids: tuple[UUID, ...] = ()
    target_ids: tuple[UUID, ...] = ()
    evidence_chunk_ids: tuple[UUID, ...] = ()
    resolution_source: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "surface": self.reference.surface,
            "canonical": self.reference.normalized_key,
            "entity_type": self.reference.kind.value,
            "must_link": True,
            "linked": self.status == ResolutionStatus.RESOLVED,
            "resolution_status": self.status.value,
            "resolution_source": self.resolution_source,
            "matched_document_ids": [str(item) for item in self.document_ids],
            "target_ids": [str(item) for item in self.target_ids],
            "evidence_chunk_ids": [str(item) for item in self.evidence_chunk_ids],
            "diagnostics": _safe_diagnostic_value(self.diagnostics),
        }


def serialize_reference_resolutions(
    resolutions: list[ReferenceResolution] | tuple[ReferenceResolution, ...],
) -> list[dict[str, object]]:
    return [item.as_dict() for item in resolutions]
