# RAG Foundation

Production-oriented retrieval-augmented generation (RAG) system with Advanced RAG and Agentic RAG layers.

## Project Overview

```text
Basic RAG → Advanced RAG → Agentic RAG
```

**Current phase: Phase 3F — Query decomposition / task planning**

Advanced RAG pipeline:

```text
Query → (optional rewrite / multi-query) → hybrid retrieval → context optimization → Groq → Answer + Citations
```

Agentic RAG pipeline:

```text
Query → routing / planning → ToolRegistry (RAG, Tavily) → generation → Answer + Citations
```

### API

| Endpoint | Description |
| --- | --- |
| `POST /documents/upload` | Ingest documents |
| `POST /query` | Advanced RAG (direct) |
| `POST /agent/query` | Agent orchestrator (RAG + optional Tavily) |
| `GET /health` | Overall health (always 200; reports degraded dependencies) |
| `GET /ready` | Readiness probe (503 until Qdrant + keyword index are available) |
| `GET /live` | Liveness probe |

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12+ |
| API | FastAPI + Pydantic v2 |
| LLM | Groq |
| Embeddings | Hugging Face |
| Vector DB | Qdrant |
| Web search | Tavily (optional) |
| Packaging | Docker / Docker Compose |
| Quality | pytest, Ruff, mypy |

## Architecture

```text
app/
├── main.py                 # FastAPI entrypoint + lifespan
├── api/                    # HTTP routes + dependency injection
├── core/                   # config, logging, exceptions, middleware
├── schemas/                # API response models
├── services/
│   ├── llm/                # GroqLLMService
│   ├── embeddings/         # HuggingFaceEmbeddingService
│   ├── ingestion/          # DocumentIngestionService
│   ├── retrieval/          # vector + BM25 hybrid retrieval
│   ├── rag/                # RAGService
│   └── agent/              # routing, planning, tools, AgentService
├── vector_store/           # QdrantVectorStore
└── utils/
```

## Production / Docker Architecture

```text
                    ┌─────────────────────────────┐
                    │  gateway (nginx) :8000      │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
        ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
        │  app #1   │        │  app #2   │        │  app #3   │
        │ (stateless│        │ (stateless│        │ (stateless│
        │  FastAPI) │        │  FastAPI) │        │  FastAPI) │
        └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
        ┌─────▼─────┐                          ┌────────▼────────┐
        │  Qdrant   │                          │ keyword_index   │
        │  volume   │                          │ volume (BM25)   │
        └───────────┘                          └─────────────────┘
```

Each FastAPI replica is **stateless**. Shared state lives in:

- **Qdrant** — vector storage (named volume `qdrant_storage`)
- **BM25 keyword index** — JSON file on shared volume `keyword_index_data`

The BM25 index uses file locking, atomic writes, and reload-on-read so multiple API replicas can safely share the same index volume.

External provider calls (Groq, Hugging Face, Tavily) are made per request; no session state is stored in memory beyond process-local caches (embeddings, clients).

## Local Development

### 1. Virtual environment

```bash
cd /path/to/agentic_rag
python3.12 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/python -m ...
pip install -e ".[dev]"
```

### 2. Environment

```bash
cp .env.example .env
```

Set at minimum:

- `GROQ_API_KEY`
- `HUGGINGFACE_API_KEY`
- `QDRANT_URL=http://localhost:6333`

Optional for agent web search:

- `TAVILY_ENABLED=true`
- `TAVILY_API_KEY=...`

### 3. Start Qdrant only

```bash
docker compose up -d qdrant
```

### 4. Start FastAPI locally

```bash
.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### 5. Health checks

```bash
curl http://127.0.0.1:8001/live
curl http://127.0.0.1:8001/ready
curl http://127.0.0.1:8001/health
```

### 6. API smoke test

```bash
# Upload
curl -F "files=@sample.pdf" http://127.0.0.1:8001/documents/upload

# RAG query
curl -X POST http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?"}'

# Agent query
curl -X POST http://127.0.0.1:8001/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?"}'
```

## Docker Compose

Secrets stay in `.env` on the host and are injected at runtime — they are **not** baked into the image.

### Build and start (single replica)

```bash
docker compose up --build
```

The API is exposed through the **gateway** (default [http://localhost:8080](http://localhost:8080); override with `GATEWAY_PORT=8000`).

### Horizontal scaling

Scale stateless API containers behind the nginx gateway:

```bash
docker compose up --build --scale app=3
```

All replicas share:

- Qdrant at `http://qdrant:6333`
- BM25 index volume at `/app/data/keyword_index/index.json`

Verify health through the gateway:

```bash
curl http://localhost:8080/ready
curl http://localhost:8080/health
```

### Scaling notes

| Topic | Behavior |
| --- | --- |
| `/query`, `/agent/query` | Safe to run on any replica (read-heavy) |
| `/documents/upload` | Safe across replicas; BM25 writes are serialized with file locks |
| Qdrant | Shared external service; handles concurrent reads/writes |
| BM25 index | Shared JSON file with lock + atomic write + reload-on-read |
| Process caches | Embedding/client caches are per-replica (acceptable) |
| Sticky sessions | Not required |

### Scaling limitations

- BM25 updates are serialized — heavy concurrent upload load may queue on the index lock.
- No distributed cache for embeddings — each replica maintains its own in-process cache.
- Provider rate limits (Groq, Hugging Face, Tavily) apply across all replicas collectively.
- Docker Compose scaling is suitable for development and small production deployments; larger deployments may need an external load balancer or orchestrator.

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key production settings:

| Variable | Purpose |
| --- | --- |
| `APP_ENV` | `development`, `staging`, `production`, `test` |
| `GROQ_API_KEY` | Required in production |
| `HUGGINGFACE_API_KEY` | Required in production |
| `QDRANT_URL` | Qdrant HTTP endpoint |
| `KEYWORD_INDEX_PATH` | BM25 index file path |
| `REQUEST_TIMEOUT_SECONDS` | HTTP request timeout middleware |
| `UVICORN_WORKERS` | Worker processes per container (default 1; prefer scaling containers) |
| `TAVILY_ENABLED` | Enable web search tool |
| `HYBRID_SEARCH_ENABLED` | Enable vector + BM25 hybrid retrieval |

## Development Commands

```bash
# Tests
pytest
pytest -m integration
pytest tests/unit -q

# Lint / type check
ruff check .
mypy app

# Docker
docker compose build
docker compose up --build
docker compose up --build --scale app=3
```

## License

Proprietary — internal use unless otherwise specified.
