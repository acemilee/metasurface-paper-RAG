from paper_rag.models.document import DocumentGenre
from paper_rag.services.document_genre import build_genre_segments, classify_document_genre


class AmbiguousGenreProvider:
    model_id = "ambiguous-genre"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_ambiguous_short_document_is_not_classified_by_page_count() -> None:
    result = classify_document_genre("paper.pdf", ["content"] * 9, AmbiguousGenreProvider())

    assert result.genre == DocumentGenre.UNCLASSIFIED


def test_ambiguous_long_document_is_not_classified_by_page_count() -> None:
    result = classify_document_genre("document.pdf", ["content"] * 100, AmbiguousGenreProvider())

    assert result.genre == DocumentGenre.UNCLASSIFIED


def test_explicit_research_article_overrides_ambiguous_embeddings() -> None:
    pages = [
        "Research Article\nAbstract: We propose a device and measure its performance.\n1. Introduction",
        "2. Structure design and experiment",
    ]

    result = classify_document_genre("paper.pdf", pages, AmbiguousGenreProvider())

    assert result.genre == DocumentGenre.RESEARCH_PAPER
    assert result.decision_source == "explicit_publication_type"
    assert result.evidence[0]["text"] == "Research Article"


def test_explicit_review_article_is_preserved() -> None:
    result = classify_document_genre(
        "review.pdf", ["Review Article\nA systematic comparison of prior studies."], AmbiguousGenreProvider()
    )

    assert result.genre == DocumentGenre.REVIEW_PAPER


def test_segment_budgets_preserve_tail_when_front_is_long() -> None:
    pages = ["Research Article\n" + "front " * 2000, "content", "Conclusion\nfinal finding"]

    segments = build_genre_segments("paper.pdf", pages)

    assert len(segments["front_matter"]) == 2500
    assert "final finding" in segments["tail_matter"]


def test_reference_body_is_excluded_from_tail_segment() -> None:
    pages = ["Abstract", "Conclusion\nresult\nReferences\nReview Article cited title"]

    segments = build_genre_segments("paper.pdf", pages)

    assert "Review Article cited title" not in segments["tail_matter"]
