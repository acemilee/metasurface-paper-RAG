from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from paper_rag.db import SessionLocal
from paper_rag.models.document import Document
from paper_rag.models.page import Page
from paper_rag.services.document_genre import classify_document_genre
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.config import get_settings


def main() -> None:
    provider = get_embedding_provider(get_settings())
    with SessionLocal() as session:
        documents = list(session.scalars(select(Document).order_by(Document.created_at)))
        for document in documents:
            page_texts = list(session.scalars(select(Page.text).where(Page.document_id == document.id).order_by(Page.page_number)))
            result = classify_document_genre(document.original_filename, page_texts, provider)
            document.document_genre = result.genre
            document.genre_score = result.score
            document.genre_classifier_version = result.classifier_version
            document.genre_checked_at = datetime.now().astimezone()
            print({"document": document.original_filename, "genre": result.genre.value, "score": round(result.score, 3), "margin": round(result.margin, 3)})
        session.commit()


if __name__ == "__main__":
    main()
