from pathlib import Path

from langchain_core.documents import Document

from src.omniassist import ingest


def test_load_documents_reads_text_files(tmp_path, monkeypatch):
    document = tmp_path / "test.txt"
    document.write_text(
        "OmniAssist test document.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ingest,
        "DOCUMENTS",
        tmp_path,
    )

    documents = ingest.load_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "OmniAssist test document."


def test_load_documents_ignores_unsupported_files(tmp_path, monkeypatch):
    document = tmp_path / "test.xyz"
    document.write_text(
        "Should be ignored.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ingest,
        "DOCUMENTS",
        tmp_path,
    )

    documents = ingest.load_documents()

    assert documents == []


def test_split_documents_creates_chunks():
    text = "OmniAssist " * 200

    documents = [
        Document(page_content=text)
    ]

    chunks = ingest.split_documents(documents)

    assert len(chunks) > 1