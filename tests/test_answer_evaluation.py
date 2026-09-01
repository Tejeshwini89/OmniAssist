from src.omniassist.answer_evaluation import (
    ABSTENTION_PHRASE,
    AnswerEvaluationCase,
    evaluate_answer,
    evaluate_generator,
)


def make_case(**overrides):
    values = {
        "case_id": "case-1",
        "question": "How do I reset my password?",
        "expected_answer_terms": ("identity portal", "password reset"),
        "expected_source": "company_it_knowledge.txt",
        "should_abstain": False,
    }
    values.update(overrides)
    return AnswerEvaluationCase(**values)


def test_supported_answer_passes_when_grounded_and_complete():
    case = make_case()
    result = evaluate_answer(
        case,
        "Use the Identity Portal for the password reset.",
        [
            {
                "source": "C:/docs/company_it_knowledge.txt",
                "content": "Employees can use the Identity Portal for password reset.",
            }
        ],
    )

    assert result["passed"] is True
    assert result["answer_coverage"] == 1.0
    assert result["source_correct"] is True
    assert result["groundedness"] is True


def test_incomplete_answer_fails():
    case = make_case()
    result = evaluate_answer(
        case,
        "Use the Identity Portal.",
        [
            {
                "source": "company_it_knowledge.txt",
                "content": "Use the Identity Portal for password reset.",
            }
        ],
    )

    assert result["passed"] is False
    assert result["answer_coverage"] == 0.5


def test_correct_abstention_passes():
    case = make_case(
        expected_answer_terms=(),
        expected_source="",
        should_abstain=True,
    )
    result = evaluate_answer(case, ABSTENTION_PHRASE, [])

    assert result["passed"] is True
    assert result["abstained"] is True
    assert result["abstention_correct"] is True


def test_supported_case_with_wrong_source_fails():
    case = make_case()
    result = evaluate_answer(
        case,
        "Use the Identity Portal for the password reset.",
        [
            {
                "source": "security_policy.txt",
                "content": "Use the Identity Portal for password reset.",
            }
        ],
    )

    assert result["passed"] is False
    assert result["source_correct"] is False


def test_generator_aggregates_metrics():
    supported_question = "How do I reset my password?"
    unsupported_question = "What is the international business-class reimbursement policy?"

    cases = [
        make_case(case_id="supported", question=supported_question),
        make_case(
            case_id="unsupported",
            question=unsupported_question,
            expected_answer_terms=(),
            expected_source="",
            should_abstain=True,
        ),
    ]

    def fake_generate(question):
        if question == supported_question:
            return {
                "answer": "Use the Identity Portal for the password reset.",
                "sources": [
                    {
                        "source": "company_it_knowledge.txt",
                        "content": "Use the Identity Portal for password reset.",
                    }
                ],
            }
        return {"answer": ABSTENTION_PHRASE, "sources": []}

    result = evaluate_generator(fake_generate, cases)

    assert result["cases"] == 2
    assert result["pass_rate"] == 1.0
    assert result["source_accuracy"] == 0.5
    assert result["groundedness_rate"] == 1.0
    assert result["abstention_accuracy"] == 1.0
