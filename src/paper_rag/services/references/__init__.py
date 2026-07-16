from paper_rag.services.references.control import (
    decide_reference_control,
    enqueue_reference_repairs,
    prepare_reference_control,
)
from paper_rag.services.references.evidence import merge_resolved_reference_evidence
from paper_rag.services.references.formula import resolve_formula_reference
from paper_rag.services.references.parser import parse_typed_references
from paper_rag.services.references.registry import resolve_typed_references
from paper_rag.services.references.structure import resolve_structure_reference
from paper_rag.services.references.types import (
    ReferenceResolution,
    TypedReference,
    serialize_reference_resolutions,
)

__all__ = [
    "ReferenceResolution",
    "TypedReference",
    "parse_typed_references",
    "resolve_formula_reference",
    "resolve_structure_reference",
    "resolve_typed_references",
    "merge_resolved_reference_evidence",
    "decide_reference_control",
    "prepare_reference_control",
    "enqueue_reference_repairs",
    "serialize_reference_resolutions",
]
