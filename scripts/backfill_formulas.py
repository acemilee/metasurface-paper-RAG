from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_rag.config import get_settings
from paper_rag.db import SessionLocal
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.services.formula_backfill import apply_formula_backfill, plan_formula_backfill
from paper_rag.services.formula_jobs import (
    enqueue_stale_formula_jobs,
    execute_persisted_formula_job,
    requeue_resumable_formula_jobs,
    run_formula_job_batch,
)
from paper_rag.services.vector_store import get_chroma_collection


def parse_pages(value: str) -> list[int]:
    pages: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError("Page list contains an empty item")
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise ValueError(f"Invalid page range: {token}")
            start, end = (int(piece) for piece in pieces)
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {token}")
            pages.update(range(start, end + 1))
        else:
            if not token.isdigit() or int(token) < 1:
                raise ValueError(f"Invalid page number: {token}")
            pages.add(int(token))
    return sorted(pages)


def _formula_summary(formula) -> dict:
    return {
        "formula_id": str(formula.id),
        "page_number": formula.page_number,
        "formula_number": formula.formula_number,
        "group_key": formula.group_key,
        "part_index": formula.part_index,
        "bbox": json.loads(formula.bbox_json),
        "parser_version": formula.parser_version,
        "semantic_status": formula.semantic_status,
        "fidelity_status": formula.fidelity_status,
    }


def _plan_report(plan, *, mode: str) -> dict:
    changed = [
        {
            "chunk_index": item.chunk_index,
            "action": item.action,
            "vector_id": item.vector_id,
            "old_content_sha256": item.old_content_sha256,
            "new_content_sha256": item.new_content_sha256,
            "old_formula_ids": list(item.old_formula_ids),
            "new_formula_ids": list(item.new_formula_ids),
        }
        for item in plan.changed_chunks
    ]
    return {
        "mode": mode,
        "document_id": str(plan.document_id),
        "pages": list(plan.page_numbers),
        "formula_before": [_formula_summary(item) for item in plan.old_formulas],
        "formula_after": [_formula_summary(item) for item in plan.new_formulas],
        "changed_chunks": changed,
        "changed_vector_ids": [item.vector_id for item in plan.changed_chunks],
        "status": "planned" if mode == "dry-run" else "applying",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned formula-layout backfill")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--document-id", type=UUID)
    target.add_argument("--all-stale", action="store_true")
    parser.add_argument("--pages")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--apply-safe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/results/formula-backfill-report.json"),
    )
    args = parser.parse_args()
    if args.document_id is not None and not args.pages:
        parser.error("--pages is required with --document-id")
    if args.all_stale and args.pages:
        parser.error("--pages cannot be used with --all-stale")
    if args.all_stale and args.apply:
        parser.error("Use --apply-safe for full-library backfill")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    settings = get_settings()
    with SessionLocal() as session:
        if args.all_stale:
            provider = get_embedding_provider(settings) if args.apply_safe else None
            collection = get_chroma_collection(settings, provider) if provider is not None else None
            resumed = requeue_resumable_formula_jobs(session) if args.resume else 0
            jobs = enqueue_stale_formula_jobs(
                session,
                batch_size=args.batch_size,
                apply_safe=args.apply_safe,
            )
            batch = run_formula_job_batch(
                session,
                batch_size=args.batch_size,
                worker_id="formula-backfill-cli",
                execute_job=lambda job: execute_persisted_formula_job(
                    session,
                    job,
                    settings,
                    provider,
                    collection,
                ),
            )
            report = {
                "mode": "all-stale-apply-safe" if args.apply_safe else "all-stale-dry-run",
                "status": "completed",
                "enqueued_job_ids": [str(job.id) for job in jobs],
                "resumed": resumed,
                "claimed": batch.claimed,
                "completed": batch.completed,
                "needs_review": batch.needs_review,
                "failed": batch.failed,
                "job_ids": [str(job_id) for job_id in batch.job_ids],
            }
        else:
            pages = parse_pages(args.pages)
            plan = plan_formula_backfill(session, args.document_id, pages, settings)
            should_apply = args.apply or args.apply_safe
            report = _plan_report(plan, mode="apply" if should_apply else "dry-run")
            if should_apply:
                provider = get_embedding_provider(settings)
                collection = get_chroma_collection(settings, provider)
                report.update(
                    apply_formula_backfill(
                        session,
                        plan,
                        settings,
                        provider,
                        collection,
                    )
                )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
