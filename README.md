# OmniAssist

OmniAssist is an enterprise AI knowledge assistant built using Retrieval-Augmented Generation (RAG).

The project is intentionally being developed beyond a basic chatbot prototype: the goal is to build a measurable, testable, secure, and deployable enterprise-style AI system.

## Features

- Enterprise document ingestion
- PDF and TXT document support
- Document chunking
- Hugging Face embeddings
- FAISS vector search
- Retrieval-Augmented Generation
- Local LLM inference using Ollama
- FastAPI backend
- Web-based chat interface
- Source attribution
- Grounded responses to reduce hallucination
- Automated tests with pytest
- Quantitative retrieval evaluation
- GitHub Actions CI test pipeline

## Architecture

```text
User
  |
Web UI
  |
FastAPI
  |
Retriever
  |
FAISS Vector Index
  |
Relevant Context
  |
Ollama
  |
Qwen3:4B
  |
Answer
  |
Web UI
```

## Project Structure

```text
OmniAssist/
├── artifacts/
│   └── faiss_index/
├── data/
│   └── documents/
├── evaluation/
│   └── retrieval_dataset.json
├── src/
│   └── omniassist/
│       ├── api.py
│       ├── evaluation.py
│       ├── generator.py
│       ├── ingest.py
│       ├── retriever.py
│       └── __init__.py
├── tests/
│   ├── test_api.py
│   ├── test_evaluation.py
│   ├── test_generator.py
│   ├── test_ingest.py
│   └── test_retriever.py
├── web/
│   └── index.html
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── requirements.txt
└── README.md
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| RAG Framework | LangChain |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| LLM Runtime | Ollama |
| LLM | Qwen3:4B |
| Frontend | HTML/CSS/JavaScript |
| Testing | Pytest |
| CI | GitHub Actions |

## RAG Pipeline

1. Enterprise documents are loaded from the `data/documents` directory.
2. Documents are split into smaller overlapping chunks.
3. Each chunk is converted into an embedding.
4. Embeddings are stored in a FAISS vector index.
5. A user's question is converted into an embedding.
6. FAISS retrieves the most relevant chunks.
7. Retrieved context is passed to Qwen3:4B through Ollama.
8. The model generates a grounded answer.
9. FastAPI returns the answer and source information to the web interface.

## Grounded Answering

OmniAssist instructs the language model to use only the retrieved enterprise knowledge context.

If the required information is not available, the assistant responds:

> I don't have enough information in the enterprise knowledge base to answer that.

This helps reduce hallucinations and prevents the model from inventing enterprise policies or procedures.

## Retrieval Evaluation

OmniAssist now includes a quantitative retrieval evaluation layer in `src/omniassist/evaluation.py`.

The initial evaluation dataset contains eight representative questions covering password reset, VPN access, laptop replacement, leave policy, and IT Service Desk workflows. Each case defines the expected source and evidence terms.

The evaluator reports:

- **Source Hit Rate** — percentage of questions where the expected source appears in the top-K results.
- **MRR (Mean Reciprocal Rank)** — rewards retrieving the expected source near the top of the ranking.
- **Evidence Recall** — percentage of expected evidence terms found in the retrieved context.

Run the evaluation locally with:

```powershell
python -m src.omniassist.evaluation
```

The current dataset is a baseline evaluation set, not a claim of production-level retrieval quality. As OmniAssist's knowledge base grows, the evaluation set should grow with it and include difficult, ambiguous, and adversarial queries.

## API

### Health Check

```text
GET /health
```

### Ask a Question

```text
POST /ask
```

Example request:

```json
{
  "question": "How do I reset my corporate password?"
}
```

## Testing

Run the complete test suite:

```powershell
python -m pytest -q
```

The repository includes tests for API validation, generation/prompt construction, document ingestion, retrieval behavior, and retrieval evaluation metrics.

GitHub Actions runs the test suite automatically on pushes to `main` and on pull requests targeting `main`.

## Running the Application

### Start Ollama

```powershell
ollama run qwen3:4b
```

### Build the Vector Index

```powershell
python -m src.omniassist.ingest
```

### Start FastAPI

```powershell
uvicorn src.omniassist.api:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### Start the Frontend

```powershell
python -m http.server 5500 --directory web
```

Frontend:

```text
http://127.0.0.1:5500
```

## Engineering Roadmap

The project is being strengthened in stages rather than by adding superficial features.

### Completed foundation

- End-to-end RAG pipeline
- Local LLM inference
- Source attribution
- Grounded-answer/refusal behavior
- API and component tests
- Quantitative retrieval evaluation framework
- CI test workflow

### Next priorities

- Expand and diversify the retrieval evaluation dataset
- Add retrieval score inspection and latency measurements
- Improve retrieval with metadata filtering and reranking where justified by evaluation
- Add answer-level groundedness and citation evaluation
- Add structured logging and request tracing
- Add robust LLM/retrieval error handling
- Add authentication and document-level authorization
- Add rate limiting and security hardening
- Containerize the full application workflow
- Add deployment, monitoring, and observability

The goal is a production-style enterprise AI system whose quality can be demonstrated with evidence, not just a working demo.
