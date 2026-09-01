from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ABSTENTION_PHRASE = "I don't have enough information in the enterprise knowledge base to answer that."


@dataclass(frozen=True)
class AnswerEvaluationCase:
    """One deterministic answer-quality evaluation example."""

    case_id: str
    question: str
    expected_answer_terms: tuple[str, ...]
    expected_source: str
    should_abstain: bool


def load_answer_cases(path: str | Path) -> list[AnswerEvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Answer evaluation dataset must contain a non-empty JSON list.")

    cases: list[AnswerEvaluationCase] = []
    required = {"id", "question", "expected_answer_terms", "expected_source", "should_abstain"}
    for item in data:
        if not isinstance(item, dict) or required.difference(item):
            raise ValueError("Each answer evaluation case must contain all required fields.")
        terms = item["expected_answer_terms"]
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise ValueError("expected_answer_terms must be a list of strings.")
        cases.append(
            AnswerEvaluationCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                expected_answer_terms=tuple(term.strip().lower() for term in terms),
                expected_source=str(item["expected_source"]),
                should_abstain=bool(item["should_abstain"]),
            )
        )
    return cases


def _normalized_source(value: str) -> str:
    return Path(value.replace("\\", "/")).name.lower()


def evaluate_answer(
    case: AnswerEvaluationCase,
    answer: str,
    sources: Sequence[dict],
) -> dict:
    """Score an answer with transparent, deterministic checks."""
    normalized_answer = answer.strip().lower()
    expected_source = _normalized_source(case.expected_source) if case.expected_source else ""
    source_names = [_normalized_source(str(source.get("source", ""))) for source in sources]

    source_correct = bool(expected_source) and expected_source in source_names
    matched_terms = [term for term in case.expected_answer_terms if term in normalized_answer]
    answer_coverage = (
        len(matched_terms) / len(case.expected_answer_terms)
        if case.expected_answer_terms
        else 1.0
    )

    abstained = ABSTENTION_PHRASE.lower() in normalized_answer
    abstention_correct = abstained == case.should_abstain

    groundedness = (
        all(term in "\n".join(str(source.get("content", "")).lower() for source in sources) for term in matched_terms)
        if matched_terms
        else (abstained or not case.expected_answer_terms)
    )

    return {
        "id": case.case_id,
        "question": case.question,
        "answer_coverage": round(answer_coverage, 4),
        "matched_terms": matched_terms,
        "source_correct": source_correct,
        "abstention_correct": abstention_correct,
        "abstained": abstained,
        "groundedness": bool(groundedness),
        "passed": bool(
            source_correct
            and abstention_correct
            and answer_coverage == 1.0
            and groundedness
        ) if not case.should_abstain else bool(abstention_correct and abstained),
    }


def evaluate_generator(
    generate_fn: Callable[[str], dict],
    cases: Sequence[AnswerEvaluationCase],
) -> dict:
    """Evaluate generated answers without using an LLM as a judge."""
    if not cases:
        raise ValueError("At least one answer evaluation case is required.")

    results = []
    for case in cases:
        result = generate_fn(case.question)
        results.append(evaluate_answer(case, result.get("answer", ""), result.get("sources", [])))

    total = len(results)
    return {
        "cases": total,
        "pass_rate": round(sum(r["passed"] for r in results) / total, 4),
        "answer_coverage": round(sum(r["answer_coverage"] for r in results) / total, 4),
        "source_accuracy": round(sum(r["source_correct"] for r in results) / total, 4),
        "groundedness_rate": round(sum(r["groundedness"] for r in results) / total, 4),
        "abstention_accuracy": round(sum(r["abstention_correct"] for r in results) / total, 4),
        "results": results,
    }


def run_evaluation(dataset_path: str | Path) -> dict:
    from src.omniassist.generator import generate_answer

    cases = load_answer_cases(dataset_path)
    return evaluate_generator(generate_answer, cases)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run_evaluation(root / "evaluation" / "answer_dataset.json"), indent=2))


if __name__ == "__main__":
    main()
