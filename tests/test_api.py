from unittest.mock import patch

from fastapi.testclient import TestClient

from src.omniassist.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "OmniAssist",
    }


def test_ask_returns_generated_answer():
    mocked_result = {
        "question": "How do I reset my password?",
        "answer": "Use the Company Identity Portal.",
        "sources": [
            {
                "source": "company_it_knowledge.txt",
                "content": "Password reset information.",
            }
        ],
    }

    with patch(
        "src.omniassist.api.generate_answer",
        return_value=mocked_result,
    ):
        response = client.post(
            "/ask",
            json={"question": "How do I reset my password?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "How do I reset my password?"
    assert data["answer"] == "Use the Company Identity Portal."
    assert len(data["sources"]) == 1


def test_ask_strips_question_whitespace():
    mocked_result = {
        "question": "How do I reset my password?",
        "answer": "Use the Company Identity Portal.",
        "sources": [],
    }

    with patch("src.omniassist.api.generate_answer", return_value=mocked_result) as mock:
        response = client.post(
            "/ask",
            json={"question": "  How do I reset my password?  "},
        )

    assert response.status_code == 200
    mock.assert_called_once_with("How do I reset my password?")


def test_ask_rejects_blank_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_rejects_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_question_over_2000_characters():
    response = client.post("/ask", json={"question": "x" * 2001})
    assert response.status_code == 422


def test_ask_returns_400_for_domain_validation_error():
    with patch(
        "src.omniassist.api.generate_answer",
        side_effect=ValueError("Query cannot be empty."),
    ):
        response = client.post("/ask", json={"question": "valid question"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


def test_ask_returns_503_for_missing_vector_index():
    with patch(
        "src.omniassist.api.generate_answer",
        side_effect=RuntimeError("Vector index not found"),
    ):
        response = client.post("/ask", json={"question": "valid question"})

    assert response.status_code == 503
    assert "Vector index not found" in response.json()["detail"]


def test_ask_returns_502_without_exposing_internal_error():
    with patch(
        "src.omniassist.api.generate_answer",
        side_effect=ConnectionError("secret internal connection details"),
    ):
        response = client.post("/ask", json={"question": "valid question"})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The answer generation service is temporarily unavailable."
    )
