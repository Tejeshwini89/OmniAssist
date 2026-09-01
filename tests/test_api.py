from unittest.mock import patch

from fastapi.testclient import TestClient

from src.omniassist.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "OmniAssist"}


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["service"] == "OmniAssist"


def test_request_id_is_returned():
    response = client.get("/health", headers={"X-Request-ID": "demo-request-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "demo-request-123"
    assert "X-Process-Time-ms" in response.headers


def test_ask_returns_generated_answer():
    mocked_result = {
        "question": "How do I reset my password?",
        "answer": "Use the Company Identity Portal.",
        "sources": [{"source": "company_it_knowledge.txt", "content": "Password reset information."}],
    }
    with patch("src.omniassist.api.generate_answer", return_value=mocked_result):
        response = client.post("/ask", json={"question": "How do I reset my password?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Use the Company Identity Portal."


def test_ask_passes_identity_to_generator():
    result = {"question": "x", "answer": "x", "sources": []}
    with patch("src.omniassist.api.generate_answer", return_value=result) as mock:
        response = client.post(
            "/ask",
            json={"question": "x"},
            headers={
                "X-User-ID": "alice",
                "X-User-Roles": "employee,manager",
                "X-User-Groups": "finance",
            },
        )
    assert response.status_code == 200
    user = mock.call_args.kwargs["user"]
    assert user.user_id == "alice"
    assert user.roles == frozenset({"employee", "manager"})
    assert user.groups == frozenset({"finance"})


def test_ask_strips_question_whitespace():
    with patch("src.omniassist.api.generate_answer", return_value={"question": "x", "answer": "x", "sources": []}) as mock:
        response = client.post("/ask", json={"question": "  How do I reset my password?  "})
    assert response.status_code == 200
    mock.assert_called_once()
    assert mock.call_args.args[0] == "How do I reset my password?"


def test_ask_rejects_blank_question():
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_rejects_missing_question():
    assert client.post("/ask", json={}).status_code == 422


def test_ask_rejects_question_over_2000_characters():
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_ask_returns_400_for_domain_validation_error():
    with patch("src.omniassist.api.generate_answer", side_effect=ValueError("Query cannot be empty.")):
        response = client.post("/ask", json={"question": "valid question"})
    assert response.status_code == 400


def test_ask_returns_503_for_missing_vector_index():
    with patch("src.omniassist.api.generate_answer", side_effect=RuntimeError("Vector index not found")):
        response = client.post("/ask", json={"question": "valid question"})
    assert response.status_code == 503
    assert "Vector index not found" in response.json()["detail"]


def test_ask_returns_502_without_exposing_internal_error():
    with patch("src.omniassist.api.generate_answer", side_effect=ConnectionError("secret internal connection details")):
        response = client.post("/ask", json={"question": "valid question"})
    assert response.status_code == 502
    assert response.json()["detail"] == "The answer generation service is temporarily unavailable."
    assert "secret internal connection details" not in response.text
