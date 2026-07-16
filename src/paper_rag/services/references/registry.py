from __future__ import annotations

from collections.abc import Collection, Iterable
from uuid import UUID

from sqlalchemy.orm import Session

from paper_rag.services.references.formula import resolve_formula_reference
from paper_rag.services.references.structure import resolve_structure_reference
from paper_rag.services.references.types import (
    ReferenceKind,
    ReferenceResolution,
    TypedReference,
)


RESOLVERS = {
    ReferenceKind.FORMULA: resolve_formula_reference,
    ReferenceKind.FIGURE: resolve_structure_reference,
    ReferenceKind.TABLE: resolve_structure_reference,
    ReferenceKind.SECTION: resolve_structure_reference,
    ReferenceKind.PAGE: resolve_structure_reference,
    ReferenceKind.DOCUMENT: resolve_structure_reference,
}


def resolve_typed_references(
    session: Session,
    references: Iterable[TypedReference],
    document_ids: Collection[UUID],
) -> tuple[ReferenceResolution, ...]:
    scope = tuple(sorted(set(document_ids), key=str))
    return tuple(
        RESOLVERS[item.kind](session, item, scope)
        for item in references
    )
