from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = ROOT / "artifacts" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_store():
    """Load the persisted FAISS vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Vector index not found: {INDEX_DIR}"
        )

    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def retrieve_documents(
    query: str,
    k: int = 3,
):
    """Retrieve the most relevant document chunks."""

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


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