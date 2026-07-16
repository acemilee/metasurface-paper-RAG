from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from paper_rag.services.evaluation import (
    EvalCase,
    evaluate_citation_precision,
    evaluate_recall_at_k,
    evaluate_refusal_accuracy,
)


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        cases.append(EvalCase(record["id"], record["question"], set(record["expected_pages"]), record["should_refuse"]))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/metasurface_questions.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--document-id")
    parser.add_argument("--session-id")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    search_results = []
    answers = []
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        for case in cases:
            payload = {"question": case.question, "top_n": args.top_k}
            if args.document_id:
                payload["document_id"] = args.document_id
            response = client.post("/api/search", json=payload)
            response.raise_for_status()
            search_results.append(response.json())
            if args.session_id:
                chat_payload = {**payload, "session_id": args.session_id}
                answer = client.post("/api/chat", json=chat_payload)
                answer.raise_for_status()
                answers.append(answer.json())
    refusal_input = answers or search_results
    refusal = evaluate_refusal_accuracy(cases, refusal_input)
    report = {
        "cases": len(cases),
        f"recall_at_{args.top_k}": evaluate_recall_at_k(cases, search_results, args.top_k),
        "refusal_accuracy": refusal.accuracy,
        "false_answers": refusal.false_answers,
        "false_refusals": refusal.false_refusals,
    }
    if answers:
        report["citation_precision"] = evaluate_citation_precision(cases, answers)
        report["generated_answers"] = sum(not answer["refused"] for answer in answers)
        report["audit_passed"] = sum(answer["audit_result"] == "passed" for answer in answers)
    report["per_case"] = [
        {
            "id": case.case_id,
            "should_refuse": case.should_refuse,
            "search_sufficient": search["sufficient"],
            "refused": answer["refused"] if answers else not search["sufficient"],
            "audit_result": answer["audit_result"] if answers else None,
            "citation_pages": sorted({
                page
                for citation in (answer["citations"] if answers else [])
                for page in range(citation["page_start"], citation["page_end"] + 1)
            }),
            "answer": answer["answer"] if answers else None,
            "refusal_reason": answer["refusal_reason"] if answers else search["reason"],
        }
        for case, search, answer in zip(cases, search_results, answers or [{}] * len(cases))
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
