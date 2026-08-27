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
            json={
                "question": "How do I reset my password?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "How do I reset my password?"
    )

    assert data["answer"] == (
        "Use the Company Identity Portal."
    )

    assert len(data["sources"]) == 1


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question():
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422