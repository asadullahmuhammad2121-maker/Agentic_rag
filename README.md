# RAG Foundation

Production-oriented retrieval-augmented generation (RAG) system built incrementally.

## Project Overview

This project evolves through three major stages:

```text
Basic RAG
    ↓
Advanced RAG
    ↓
Agentic RAG
```

**Current phase: Phase 1D–1E — Retrieval + Answer Generation**

Basic RAG is complete:

```text
Query → Embedding → Qdrant search → Context → Prompt → Groq → Answer + Citations
```

API:

- `POST /documents/upload` — ingest PDFs
- `POST /query` — ask grounded questions

Still out of scope:

- Hybrid search / BM25 / reranking / query rewriting
- Context compression / evaluation framework
- Agentic RAG / web search / multi-agent workflows
- PostgreSQL / SQLAlchemy / Alembic

## Current Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12+ |
| API | FastAPI + Pydantic v2 |
| LLM | Groq |
| Embeddings | Hugging Face |
| Vector DB | Qdrant |
| Packaging | Docker / Docker Compose |
| Quality | pytest, Ruff, mypy |

## Architecture

```text
app/
├── main.py                 # FastAPI entrypoint
├── api/                    # HTTP routes + dependency injection
├── core/                   # config, logging, exceptions
├── schemas/                # API response models
├── services/
│   ├── llm/                # LLMService → GroqLLMService
│   ├── embeddings/         # EmbeddingService → HuggingFaceEmbeddingService
│   ├── chunking/           # ChunkingService
│   ├── ingestion/          # PDF extraction + DocumentIngestionService
│   ├── retrieval/          # RetrievalService
│   └── rag/                # PromptBuilder + RAGService
├── vector_store/           # VectorStore → QdrantVectorStore
├── repositories/           # reserved for future SQL phase
└── utils/                  # checksum + id helpers
```

The application depends on abstractions, not concrete provider SDKs, so Groq, Hugging Face, or Qdrant can be swapped later without rewriting RAG business logic.

## Local Setup

### 1. Clone / open the project

```bash
cd "untitled folder"   # or your clone path
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set provider keys when you need them. Phase 1A does **not** call Groq or Hugging Face APIs for generation/embeddings yet; Qdrant connectivity is the main runtime dependency.

### 5. Start Qdrant

```bash
docker compose up -d qdrant
```

### 6. Start FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 7. Run tests

```bash
pytest
```

## Development Commands

```bash
# Application
uvicorn app.main:app --reload --port 8000

# Tests
pytest
pytest -m integration          # integration suite (Qdrant tests auto-skip if down)
pytest tests/unit -q

# Lint
ruff check .
ruff format .

# Type check
mypy app

# Full stack (app + Qdrant)
docker compose up --build
```

## Roadmap (not implemented yet)

| Phase | Focus |
| --- | --- |
| 1A | Foundation (complete) |
| 1B | Document ingestion + PDF processing (complete) |
| 1C | Chunking + embeddings (complete) |
| 1D–1E | Retrieval + Groq generation + citations (current) |
| 2 | Advanced RAG |
| 3 | Agentic RAG |

## License

Proprietary — internal use unless otherwise specified.
