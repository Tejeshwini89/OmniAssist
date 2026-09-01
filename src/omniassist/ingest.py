from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ROOT / "data" / "documents"
ARTIFACTS = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default document access policy. Production deployments should replace this
# mapping with metadata from the enterprise document-management system.
DOCUMENT_ACCESS = {
    "security_policy.txt": {"allowed_roles": ["security", "admin"]},
    "hr_leave_policy.txt": {"allowed_roles": ["hr", "admin"]},
    "remote_work_policy.txt": {"allowed_roles": ["employee", "admin"]},
    "it_service_desk_policy.txt": {"allowed_roles": ["employee", "it", "admin"]},
    "company_it_knowledge.txt": {"allowed_roles": ["employee", "it", "admin"]},
}


def apply_access_metadata(documents):
    """Attach explicit document ACL metadata before indexing."""
    for document in documents:
        source = Path(str(document.metadata.get("source", ""))).name
        access = DOCUMENT_ACCESS.get(source)
        if access:
            document.metadata.update(access)
    return documents


def load_documents():
    documents = []
    for path in DOCUMENTS.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif suffix == ".txt":
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return apply_access_metadata(documents)


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


def main():
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No PDF or TXT documents found in {DOCUMENTS}")

    chunks = split_documents(documents)
    create_vector_store(chunks)

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": 800,
        "chunk_overlap": 120,
        "access_control": "role-based metadata",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "ingestion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Loaded documents : {len(documents)}")
    print(f"Created chunks   : {len(chunks)}")
    print(f"Embedding model  : {EMBEDDING_MODEL}")
    print(f"Vector index     : {INDEX_DIR}")


if __name__ == "__main__":
    main()
