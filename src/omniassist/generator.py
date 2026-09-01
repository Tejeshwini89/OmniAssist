from __future__ import annotations

import os

import requests

from src.omniassist.retriever import retrieve_documents


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    os.getenv(
        "OMNIASSIST_OLLAMA_URL",
        "http://localhost:11434/api/generate",
    ),
)
MODEL_NAME = os.getenv(
    "OMNIASSIST_MODEL",
    "qwen3:4b",
)


def build_context(documents) -> str:
    """Combine retrieved document chunks into one context string."""

    return "\n\n".join(
        document.page_content
        for document in documents
    )


def build_prompt(
    question: str,
    context: str,
) -> str:
    """Build a grounded RAG prompt."""

    return f"""
You are OmniAssist, an enterprise knowledge assistant.

Answer the user's question using ONLY the provided enterprise
knowledge context.

If the answer is not present in the context, say:
"I don't have enough information in the enterprise knowledge base
to answer that."

Do not invent policies, procedures, numbers, or facts.

Enterprise Knowledge Context:
------------------------------
{context}
------------------------------

User Question:
{question}

Provide a concise and clear answer.
""".strip()


def generate_answer(
    question: str,
    k: int = 3,
) -> dict:
    """Retrieve relevant context and generate an answer with Qwen."""

    documents = retrieve_documents(
        question,
        k=k,
    )

    context = build_context(documents)

    prompt = build_prompt(
        question,
        context,
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return {
        "question": question,
        "answer": result["response"].strip(),
        "sources": [
            {
                "source": document.metadata.get(
                    "source",
                    "unknown",
                ),
                "content": document.page_content,
            }
            for document in documents
        ],
    }


def main():
    question = input("Question: ").strip()

    result = generate_answer(question)

    print("\n=== OmniAssist Answer ===\n")
    print(result["answer"])

    print("\n=== Sources ===\n")

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"--- Source {index} ---"
        )
        print(source["source"])
        print()


if __name__ == "__main__":
    main()
