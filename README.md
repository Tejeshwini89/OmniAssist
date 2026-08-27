# OmniAssist

OmniAssist is an enterprise AI knowledge assistant built using Retrieval-Augmented Generation (RAG).

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
├── src/
│   └── omniassist/
│       ├── api.py
│       ├── generator.py
│       ├── ingest.py
│       ├── retriever.py
│       └── __init__.py
├── tests/
│   ├── test_api.py
│   ├── test_generator.py
│   ├── test_ingest.py
│   └── test_retriever.py
├── web/
│   └── index.html
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

This helps reduce hallucination and prevents the model from inventing enterprise policies or procedures.

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

Current test status:

```text
13 passed
```

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

## Future Improvements

- Authentication and authorization
- Conversation memory
- Streaming responses
- Improved retrieval evaluation
- Metadata filtering
- Structured logging
- Rate limiting
- Docker deployment
- Monitoring and observability
- Support for larger enterprise knowledge bases