from __future__ import annotations

import argparse
import json
from datetime import datetime

from sqlalchemy import select

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.models.document import Document, DocumentStatus
from paper_rag.models.page import Page
from paper_rag.services.document_genre import classify_document_genre
from paper_rag.services.embeddings import get_embedding_provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    provider = get_embedding_provider(get_settings())
    with SessionLocal() as session:
        documents = list(session.scalars(
            select(Document).where(Document.status == DocumentStatus.COMPLETED).order_by(Document.created_at)
        ))
        for document in documents:
            if document.genre_manually_overridden:
                print(f"SKIP manual {document.original_filename}")
                continue
            pages = list(session.scalars(
                select(Page).where(Page.document_id == document.id).order_by(Page.page_number)
            ))
            result = classify_document_genre(document.original_filename, [page.text for page in pages], provider)
            print(
                f"{document.original_filename}: {document.document_genre} -> {result.genre.value} "
                f"confidence={result.score:.3f} source={result.decision_source}"
            )
            if args.dry_run:
                continue
            if document.genre_original_prediction is None:
                document.genre_original_prediction = document.document_genre
            document.document_genre = result.genre
            document.genre_score = result.score
            document.genre_decision_source = result.decision_source
            document.genre_scores_json = json.dumps(result.scores, ensure_ascii=False)
            document.genre_evidence_json = json.dumps(result.evidence, ensure_ascii=False)
            document.genre_conflicts_json = json.dumps(result.conflicts, ensure_ascii=False)
            document.genre_classifier_version = result.classifier_version
            document.genre_checked_at = datetime.now().astimezone()
        if not args.dry_run:
            session.commit()


if __name__ == "__main__":
    main()
