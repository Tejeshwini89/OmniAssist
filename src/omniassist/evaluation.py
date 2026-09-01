from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class EvaluationCase:
    """One retrieval evaluation example with expected evidence."""

    case_id: str
    question: str
    expected_source: str
    expected_terms: tuple[str, ...]


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load retrieval evaluation cases from a JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a JSON list.")

    cases: list[EvaluationCase] = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each evaluation case must be a JSON object.")

        required = {"id", "question", "expected_source", "expected_terms"}
        missing = required.difference(item)
        if missing:
            raise ValueError(
                "Evaluation case is missing fields: "
                + ", ".join(sorted(missing))
            )

        terms = item["expected_terms"]
        if not isinstance(terms, list) or not all(
            isinstance(term, str) and term.strip() for term in terms
        ):
            raise ValueError("expected_terms must be a non-empty list of strings.")

        cases.append(
            EvaluationCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                expected_source=str(item["expected_source"]),
                expected_terms=tuple(term.strip().lower() for term in terms),
            )
        )

    if not cases:
        raise ValueError("Evaluation dataset cannot be empty.")

    return cases


def _source_name(value: str) -> str:
    """Normalize a source path so absolute/local prefixes do not matter."""

    return Path(value.replace("\\", "/")).name.lower()


def _document_source(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("source", "unknown"))


def _document_text(document) -> str:
    return str(getattr(document, "page_content", ""))


def evaluate_retrieval(
    retrieve_fn: Callable[[str, int], Sequence],
    cases: Sequence[EvaluationCase],
    k: int = 3,
) -> dict:
    """Evaluate retrieval using source hit rate, MRR and evidence recall."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    source_hits = 0
    reciprocal_rank_total = 0.0
    evidence_recall_total = 0.0
    case_results = []

    for case in cases:
        documents = list(retrieve_fn(case.question, k))
        expected_source = _source_name(case.expected_source)
        normalized_sources = [_source_name(_document_source(doc)) for doc in documents]

        rank = None
        for index, source in enumerate(normalized_sources, start=1):
            if source == expected_source:
                rank = index
                break

        source_hit = rank is not None
        if source_hit:
            source_hits += 1
            reciprocal_rank_total += 1.0 / rank

        retrieved_text = "\n\n".join(
            _document_text(document).lower() for document in documents
        )
        matched_terms = [
            term for term in case.expected_terms if term in retrieved_text
        ]
        evidence_recall = len(matched_terms) / len(case.expected_terms)
        evidence_recall_total += evidence_recall

        case_results.append(
            {
                "id": case.case_id,
                "question": case.question,
                "source_hit": source_hit,
                "source_rank": rank,
                "matched_terms": matched_terms,
                "evidence_recall": round(evidence_recall, 4),
            }
        )

    total = len(cases)

    return {
        "k": k,
        "cases": total,
        "source_hit_rate": round(source_hits / total, 4),
        "mrr": round(reciprocal_rank_total / total, 4),
        "evidence_recall": round(evidence_recall_total / total, 4),
        "results": case_results,
    }


def run_evaluation(
    dataset_path: str | Path,
    k: int = 3,
) -> dict:
    """Run the evaluation against OmniAssist's real retriever."""

    from src.omniassist.retriever import retrieve_documents

    cases = load_evaluation_cases(dataset_path)
    return evaluate_retrieval(retrieve_documents, cases, k=k)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset_path = root / "evaluation" / "retrieval_dataset.json"
    result = run_evaluation(dataset_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
