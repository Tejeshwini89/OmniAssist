from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.omniassist.evaluation import (
    EvaluationCase,
    evaluate_retrieval,
    load_evaluation_cases,
)


DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "retrieval_dataset.json"


def test_load_evaluation_cases():
    cases = load_evaluation_cases(DATASET)

    assert len(cases) == 12
    assert cases[0].case_id == "password-reset"
    assert cases[0].expected_source == "company_it_knowledge.txt"
    assert "company identity portal" in cases[0].expected_terms
    assert cases[1].expected_source == "security_policy.txt"
    assert cases[6].expected_source == "hr_leave_policy.txt"


def test_evaluate_retrieval_calculates_metrics():
    cases = [
        EvaluationCase(
            case_id="one",
            question="password",
            expected_source="company_it_knowledge.txt",
            expected_terms=("identity portal", "password"),
        ),
        EvaluationCase(
            case_id="two",
            question="vpn",
            expected_source="vpn.txt",
            expected_terms=("corporate vpn", "authentication"),
        ),
    ]

    def fake_retriever(question: str, k: int):
        assert k == 2
        if question == "password":
            return [
                Document(
                    page_content="Use the Identity Portal for password reset.",
                    metadata={"source": "C:/docs/company_it_knowledge.txt"},
                )
            ]
        return [
            Document(
                page_content="VPN client information.",
                metadata={"source": "other.txt"},
            ),
            Document(
                page_content="Connect to the corporate VPN.",
                metadata={"source": "vpn.txt"},
            ),
        ]

    result = evaluate_retrieval(fake_retriever, cases, k=2)

    assert result["cases"] == 2
    assert result["source_hit_rate"] == 1.0
    assert result["mrr"] == 0.75
    assert result["evidence_recall"] == 0.75
    assert result["results"][0]["source_rank"] == 1
    assert result["results"][1]["source_rank"] == 2


def test_load_evaluation_cases_rejects_invalid_dataset(tmp_path):
    dataset = tmp_path / "invalid.json"
    dataset.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_evaluation_cases(dataset)


def test_evaluate_retrieval_rejects_invalid_k():
    case = EvaluationCase(
        case_id="one",
        question="password",
        expected_source="company_it_knowledge.txt",
        expected_terms=("password",),
    )

    with pytest.raises(ValueError, match="k must be at least 1"):
        evaluate_retrieval(lambda question, k: [], [case], k=0)
