# OmniAssist

OmniAssist is an enterprise AI knowledge assistant built using Retrieval-Augmented Generation (RAG).

The project is intentionally developed beyond a basic chatbot prototype: the goal is a measurable, testable, secure-by-design, observable, and reproducibly deployable enterprise-style AI system.

## Features

- Enterprise document ingestion
- PDF and TXT document support
- Document chunking
- Hugging Face embeddings
- FAISS vector search with in-process caching
- Retrieval-Augmented Generation
- Local LLM inference using Ollama and Qwen3:4B
- FastAPI backend
- Web-based chat interface
- Source attribution
- Grounded responses and explicit abstention behavior
- Quantitative retrieval evaluation
- Deterministic answer-quality evaluation
- Automated tests with pytest
- GitHub Actions CI
- API validation and controlled error responses
- Request IDs, latency headers, and privacy-conscious logs
- Process-local request rate limiting
- Explicit document access-policy boundary
- Docker deployment

## Architecture

```text
User
  |
Web UI
  |
FastAPI API
  |-- validation / rate limiting / request tracing
  |
Access Policy
  |
Retriever
  |
FAISS Vector Index
  |
Relevant Enterprise Context
  |
Grounded Prompt
  |
Ollama / Qwen3:4B
  |
Answer + Sources
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
│   ├── retrieval_dataset.json
│   └── answer_dataset.json
├── src/
│   └── omniassist/
│       ├── api.py
│       ├── answer_evaluation.py
│       ├── evaluation.py
│       ├── generator.py
│       ├── ingest.py
│       ├── observability.py
│       ├── retriever.py
│       └── security.py
├── tests/
│   ├── test_api.py
│   ├── test_answer_evaluation.py
│   ├── test_evaluation.py
│   ├── test_generator.py
│   ├── test_ingest.py
│   └── test_retriever.py
├── web/
│   └── index.html
├── .github/
│   └── workflows/
│       └── tests.yml
├── .dockerignore
├── Dockerfile
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
| Packaging | Docker |

## RAG Pipeline

1. Enterprise documents are loaded from `data/documents`.
2. Documents are split into overlapping chunks.
3. Each chunk is converted into an embedding.
4. Embeddings are stored in a FAISS index.
5. A user question is embedded and used for similarity retrieval.
6. Relevant chunks are supplied as enterprise context to the generator.
7. Qwen3:4B generates an answer through Ollama.
8. FastAPI returns the answer and source information.
9. The UI displays the grounded answer and sources.

The vector store is cached inside the application process so the embedding model and FAISS index are not reloaded for every request.

## Grounded Answering

OmniAssist instructs the language model to use only the retrieved enterprise knowledge context.

If the required information is not available, the assistant is instructed to respond:

> I don't have enough information in the enterprise knowledge base to answer that.

This is a deliberate abstention behavior, not an application failure. It reduces the risk of the model inventing enterprise policies or procedures.

## Retrieval Evaluation

`src/omniassist/evaluation.py` provides a deterministic retrieval benchmark using `evaluation/retrieval_dataset.json`.

It reports:

- **Source Hit Rate** — percentage of questions where the expected source appears in the top-K results.
- **MRR (Mean Reciprocal Rank)** — rewards retrieving the expected source near the top of the ranking.
- **Evidence Recall** — percentage of expected evidence terms found in retrieved context.

Run it locally with:

```powershell
python -m src.omniassist.evaluation
```

The benchmark is a baseline, not a claim of production-level retrieval quality. It should grow with the knowledge base and include ambiguous and adversarial questions.

## Answer Evaluation

`src/omniassist/answer_evaluation.py` separately evaluates generated answers using `evaluation/answer_dataset.json`.

It measures:

- **Answer coverage** — whether expected answer evidence appears in the generated answer.
- **Source correctness** — whether the expected enterprise source was retrieved.
- **Groundedness** — whether matched answer evidence is supported by retrieved context.
- **Abstention accuracy** — whether unsupported questions are refused correctly.
- **Pass rate** — whether a case satisfies the defined deterministic checks.

The evaluator deliberately does not rely on another LLM as the sole judge. This makes the baseline transparent and reproducible.

## API

### Health Check

```text
GET /health
```

### Operational Metrics

```text
GET /metrics
```

Returns non-sensitive process-local operational counters.

### Ask a Question

```text
POST /ask
```

Example:

```json
{
  "question": "How do I reset my corporate password?"
}
```

The API returns the question, grounded answer, and source metadata.

### API Hardening

The `/ask` endpoint:

- trims and validates questions
- limits question length to 2,000 characters
- applies process-local rate limiting
- returns controlled 400/429/503/502 responses
- avoids exposing unexpected internal exception details
- returns an `X-Request-ID` and processing-time response header
- logs operational metadata without logging question content

The included `security.py` module defines an explicit document-access policy boundary. It is ready to be connected to a real identity provider and document ACL metadata; the local demo does not pretend that a client-supplied identity is enterprise authentication.

## Docker

Build the image:

```powershell
docker build -t omniassist:latest .
```

Run it:

```powershell
docker run --rm -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434/api/generate omniassist:latest
```

Ollama remains an external inference service. The container connects to an Ollama instance running on the host through `host.docker.internal`.

The Docker image excludes `.env` and other local-secret files through `.dockerignore`.

## Running Locally Without Docker

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

## Testing

Run the complete suite:

```powershell
python -m pytest -q
```

Tests cover API validation and failure handling, prompt/generator behavior, ingestion, retrieval, and both evaluation layers.

GitHub Actions runs the test suite automatically on pushes to `main` and on pull requests targeting `main`.

## Engineering Status

### Completed engineering foundation

- End-to-end RAG pipeline
- Multi-document enterprise knowledge base
- Local Qwen inference
- Grounded generation and abstention behavior
- Source attribution
- Cached retrieval infrastructure
- Quantitative retrieval evaluation
- Answer-quality evaluation
- Automated tests
- CI
- Reproducible Docker packaging
- API validation and controlled failures
- Request tracing and latency visibility
- Basic process-local rate limiting
- Explicit document authorization boundary
- Secret exclusion from Docker build context

### Remaining production integrations

These require deployment-specific infrastructure rather than more demo code:

- Connect `security.py` to a real identity provider and document ACL system.
- Replace process-local rate limiting with a shared distributed limiter when horizontally scaling.
- Add persistent metrics/tracing backend for production observability.
- Add a production vector database if FAISS no longer meets scale/availability requirements.
- Add deployment infrastructure, secrets management, and monitoring for the target cloud/platform.
- Expand evaluation data continuously with real anonymized enterprise questions.

The goal is not to claim that a local student project is already a production SaaS. The goal is to make every important engineering boundary explicit, measurable, testable, and ready for real infrastructure.
