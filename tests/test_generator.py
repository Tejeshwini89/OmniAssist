from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.omniassist import generator


def test_build_context_combines_documents():
    documents = [
        Document(page_content="Password reset information."),
        Document(page_content="VPN information."),
    ]

    context = generator.build_context(documents)

    assert "Password reset information." in context
    assert "VPN information." in context
    assert "\n\n" in context


def test_build_prompt_contains_question_and_context():
    prompt = generator.build_prompt(
        "How do I reset my password?",
        "Use the Identity Portal.",
    )

    assert "How do I reset my password?" in prompt
    assert "Use the Identity Portal." in prompt
    assert "ONLY" in prompt
    assert "enterprise" in prompt.lower()
    assert "knowledge" in prompt.lower()
    assert "context" in prompt.lower()
    assert "Do not invent policies" in prompt


def test_generate_answer_returns_answer_and_sources():
    documents = [
        Document(
            page_content="Reset your password through the Identity Portal.",
            metadata={
                "source": "company_it_knowledge.txt"
            },
        )
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "Reset your password through the Identity Portal."
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "src.omniassist.generator.retrieve_documents",
        return_value=documents,
    ), patch(
        "src.omniassist.generator.requests.post",
        return_value=mock_response,
    ) as mock_post:

        result = generator.generate_answer(
            "How do I reset my password?"
        )

    assert result["question"] == (
        "How do I reset my password?"
    )

    assert result["answer"] == (
        "Reset your password through the Identity Portal."
    )

    assert len(result["sources"]) == 1
    assert (
        result["sources"][0]["source"]
        == "company_it_knowledge.txt"
    )

    mock_post.assert_called_once()