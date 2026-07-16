from __future__ import annotations

import pytest
from sqlalchemy import select

from paper_rag.models.document import Document
from paper_rag.services.references import parse_typed_references
from paper_rag.services.references.types import ReferenceKind, ReferenceSource


def test_reference_fixtures_create_isolated_ready_documents(session, documents) -> None:
    assert len(documents) == 3
    assert len({item.id for item in documents}) == 3
    assert len(session.scalars(select(Document)).all()) == 3
    assert all(item.formula_index_status == "ready" for item in documents)


def test_formula_number_is_parsed_from_original_question_without_llm() -> None:
    references = parse_typed_references("公式5讲了什么", "请解释第五个公式")

    assert len(references) == 1
    assert references[0].kind == ReferenceKind.FORMULA
    assert references[0].normalized_key == "5"
    assert references[0].surface == "公式5"
    assert references[0].source == ReferenceSource.ORIGINAL_QUESTION


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("公式5讲了什么", "5"),
        ("式（5）是什么意思", "5"),
        ("Eq. (5) means what?", "5"),
        ("Equation 1A", "1a"),
    ],
)
def test_formula_reference_variants_share_one_key(
    question: str,
    expected: str,
) -> None:
    assert parse_typed_references(question)[0].normalized_key == expected


def test_ordinal_formula_phrase_is_not_silently_treated_as_equation_number() -> None:
    assert parse_typed_references("第五个公式讲了什么") == ()


@pytest.mark.parametrize(
    ("question", "kind", "key", "qualifier"),
    [
        ("解释图3", ReferenceKind.FIGURE, "3", None),
        ("Figure 3(b)", ReferenceKind.FIGURE, "3", "b"),
        ("表Ⅱ列出了什么", ReferenceKind.TABLE, "2", None),
        ("Table IV", ReferenceKind.TABLE, "4", None),
    ],
)
def test_figure_and_table_references_are_normalized(
    question: str,
    kind: ReferenceKind,
    key: str,
    qualifier: str | None,
) -> None:
    reference = parse_typed_references(question)[0]
    assert (reference.kind, reference.normalized_key, reference.qualifier) == (
        kind,
        key,
        qualifier,
    )


@pytest.mark.parametrize(
    ("question", "kind", "key"),
    [
        ("第4.2节讲了什么", ReferenceKind.SECTION, "4.2"),
        ("Section 4.2", ReferenceKind.SECTION, "4.2"),
        ("第8页", ReferenceKind.PAGE, "8"),
        ("page 8", ReferenceKind.PAGE, "8"),
    ],
)
def test_section_and_page_references_are_normalized(
    question: str,
    kind: ReferenceKind,
    key: str,
) -> None:
    reference = parse_typed_references(question)[0]
    assert (reference.kind, reference.normalized_key) == (kind, key)


def test_original_reference_wins_when_rewrite_changes_the_number() -> None:
    references = parse_typed_references("公式5讲了什么", "解释公式6")
    assert [(item.normalized_key, item.source) for item in references] == [
        ("5", ReferenceSource.ORIGINAL_QUESTION)
    ]


@pytest.mark.parametrize(
    "question",
    [
        "论文《alpha metasurface.pdf》讲了什么",
        'document "alpha metasurface.pdf"',
    ],
)
def test_explicit_document_reference_is_parsed_without_llm(question: str) -> None:
    reference = parse_typed_references(question)[0]
    assert reference.kind == ReferenceKind.DOCUMENT
    assert reference.normalized_key == "alpha metasurface.pdf"


@pytest.mark.parametrize(
    "text",
    [
        "模式5用于控制状态",
        "configure 3 parameters",
        "分别代表0和π两种状态",
        "a timetable 4 weeks long",
        "table values for the proposed absorber",
    ],
)
def test_reference_parser_rejects_embedded_word_false_positives(text: str) -> None:
    assert parse_typed_references(text) == ()


def test_figure_label_does_not_consume_number_from_next_line() -> None:
    references = parse_typed_references("图4 单元设计流程图\n2684\n后续正文")
    figures = [item for item in references if item.kind == ReferenceKind.FIGURE]
    assert [item.normalized_key for item in figures] == ["4"]
