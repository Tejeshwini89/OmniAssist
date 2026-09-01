from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.omniassist.security import DEFAULT_USER, User, policy


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "artifacts" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_vector_store():
    if not INDEX_DIR.exists():
        raise RuntimeError(f"Vector index not found: {INDEX_DIR}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def refresh_vector_store() -> None:
    load_vector_store.cache_clear()


def retrieve_documents(
    query: str,
    k: int = 3,
    user: User = DEFAULT_USER,
):
    """Retrieve candidates and enforce document authorization before return."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    # Over-fetch so ACL filtering does not unnecessarily reduce useful context.
    candidates = load_vector_store().similarity_search(query, k=max(k * 3, k))
    authorized = policy.filter_documents(user, candidates)
    return authorized[:k]


def main():
    query = input("Question: ").strip()
    results = retrieve_documents(query)
    print("\n=== Retrieved Context ===\n")
    for index, document in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()
