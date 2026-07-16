from paper_rag.services.evaluation import EvalCase, evaluate_recall_at_k, evaluate_refusal_accuracy


def test_recall_and_refusal_metrics() -> None:
    cases = [
        EvalCase("found", "question", {2}, False),
        EvalCase("missing", "outside", set(), True),
    ]
    results = [
        {"sufficient": True, "evidence": [{"page_start": 2, "page_end": 2}]},
        {"sufficient": False, "evidence": []},
    ]

    assert evaluate_recall_at_k(cases, results, 1) == 1.0
    assert evaluate_refusal_accuracy(cases, results).accuracy == 1.0
