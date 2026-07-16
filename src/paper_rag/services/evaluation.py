from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_pages: set[int]
    should_refuse: bool


@dataclass(frozen=True)
class RefusalMetrics:
    accuracy: float
    true_refusals: int
    false_answers: int
    false_refusals: int


class BadCaseCategory(StrEnum):
    PARSING = "parsing"
    RETRIEVAL = "retrieval"
    CITATION = "citation"
    REFUSAL = "refusal"
    GENERATION = "generation"
    NONE = "none"


def evaluate_recall_at_k(cases: list[EvalCase], results: list[dict], k: int) -> float:
    answerable = [(case, result) for case, result in zip(cases, results) if not case.should_refuse]
    if not answerable:
        return 0.0
    hits = 0
    for case, result in answerable:
        pages = set()
        for item in result.get("evidence", [])[:k]:
            pages.update(range(item["page_start"], item["page_end"] + 1))
        hits += bool(pages & case.expected_pages)
    return hits / len(answerable)


def evaluate_citation_precision(cases: list[EvalCase], answers: list[dict]) -> float:
    supported = 0
    total = 0
    for case, answer in zip(cases, answers):
        for citation in answer.get("citations", []):
            total += 1
            pages = set(range(citation["page_start"], citation["page_end"] + 1))
            supported += bool(pages & case.expected_pages)
    return supported / total if total else 0.0


def evaluate_refusal_accuracy(cases: list[EvalCase], answers: list[dict]) -> RefusalMetrics:
    true_refusals = false_answers = false_refusals = correct_answers = 0
    for case, answer in zip(cases, answers):
        refused = bool(answer.get("refused", not answer.get("sufficient", False)))
        if case.should_refuse and refused:
            true_refusals += 1
        elif case.should_refuse:
            false_answers += 1
        elif refused:
            false_refusals += 1
        else:
            correct_answers += 1
    accuracy = (true_refusals + correct_answers) / len(cases) if cases else 0.0
    return RefusalMetrics(accuracy, true_refusals, false_answers, false_refusals)


def classify_bad_case(
    *, parsed: bool, retrieved: bool, citation_valid: bool, refusal_correct: bool
) -> BadCaseCategory:
    if not parsed:
        return BadCaseCategory.PARSING
    if not retrieved:
        return BadCaseCategory.RETRIEVAL
    if not citation_valid:
        return BadCaseCategory.CITATION
    if not refusal_correct:
        return BadCaseCategory.REFUSAL
    return BadCaseCategory.NONE
