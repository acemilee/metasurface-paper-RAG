from paper_rag.services.query_intent import QueryIntent, classify_query_intent


class SemanticIntentProvider:
    model_id = "semantic-intent-test"
    dimension = 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "创新点" in text else [0.0, 1.0]
            for text in texts
        ]


class AmbiguousIntentProvider(SemanticIntentProvider):
    model_id = "ambiguous-intent-test"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_semantic_router_recognizes_novelty_without_regex() -> None:
    result = classify_query_intent("这项工作的亮点在哪里", SemanticIntentProvider())

    assert result.intent == QueryIntent.NOVELTY
    assert result.source == "bge_semantic"


def test_semantic_router_falls_back_when_intents_are_tied() -> None:
    result = classify_query_intent("含义不明确的问题", AmbiguousIntentProvider())

    assert result.intent == QueryIntent.GENERAL
    assert result.source == "fallback"
