from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS = ROOT / "data" / "documents"
ARTIFACTS = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS / "faiss_index"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load supported documents from the documents directory."""

    documents = []

    for path in DOCUMENTS.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            documents.extend(loader.load())

        elif suffix == ".txt":
            loader = TextLoader(
                str(path),
                encoding="utf-8",
            )
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    """Create and persist a FAISS vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(INDEX_DIR))

    return vector_store


def main():
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No PDF or TXT documents found in "
            f"{DOCUMENTS}"
        )

    chunks = split_documents(documents)

    create_vector_store(chunks)

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": 800,
        "chunk_overlap": 120,
    }

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ARTIFACTS / "ingestion_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Loaded documents : {len(documents)}"
    )

    print(
        f"Created chunks   : {len(chunks)}"
    )

    print(
        f"Embedding model  : {EMBEDDING_MODEL}"
    )

    print(
        f"Vector index     : {INDEX_DIR}"
    )


if __name__ == "__main__":
    main()