from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = ROOT / "artifacts" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_vector_store():
    """Load and cache the persisted FAISS vector store.

    Loading the embedding model and FAISS index for every request is
    unnecessarily expensive. The cache keeps one initialized store in
    memory for the lifetime of the application process.
    """

    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Vector index not found: {INDEX_DIR}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def refresh_vector_store() -> None:
    """Clear the cached vector store after rebuilding the index."""

    load_vector_store.cache_clear()


def retrieve_documents(
    query: str,
    k: int = 3,
):
    """Retrieve the most relevant document chunks."""

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    vector_store = load_vector_store()

    return vector_store.similarity_search(
        query,
        k=k,
    )


def main():
    query = input("Question: ").strip()

    results = retrieve_documents(query)

    print("\n=== Retrieved Context ===\n")

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(
            f"--- Result {index} ---"
        )

        print(
            document.page_content
        )

        print()


if __name__ == "__main__":
    main()
