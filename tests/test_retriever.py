from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.omniassist import retriever


def test_retrieve_documents_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve_documents("   ")


def test_retrieve_documents_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.retrieve_documents("password", k=0)


def test_retrieve_documents_returns_relevant_results():
    mock_store = MagicMock()

    expected = [
        Document(
            page_content="Employees can reset their password through the Identity Portal."
        )
    ]

    mock_store.similarity_search.return_value = expected

    with patch(
        "src.omniassist.retriever.load_vector_store",
        return_value=mock_store,
    ):
        results = retriever.retrieve_documents(
            "How do I reset my password?"
        )

    assert len(results) == 1
    assert results[0].page_content.startswith(
        "Employees can reset"
    )

    mock_store.similarity_search.assert_called_once_with(
        "How do I reset my password?",
        k=3,
    )


def test_load_vector_store_fails_when_index_missing():
    retriever.load_vector_store.cache_clear()

    with patch(
        "pathlib.Path.exists",
        return_value=False,
    ):
        with pytest.raises(
            RuntimeError,
            match="Vector index not found",
        ):
            retriever.load_vector_store()


def test_refresh_vector_store_clears_cache():
    with patch.object(
        retriever.load_vector_store,
        "cache_clear",
    ) as cache_clear:
        retriever.refresh_vector_store()

    cache_clear.assert_called_once_with()
