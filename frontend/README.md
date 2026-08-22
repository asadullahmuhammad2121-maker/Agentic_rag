# Agentic RAG Frontend

Next.js dashboard and UI for the Agentic RAG FastAPI backend.

## Features

| Page | Backend integration |
| --- | --- |
| Dashboard | Health and system status |
| Agent Chat | `POST /agent/query` |
| Documents | Upload, list, detail, delete |
| Retrieval | `POST /retrieval/explore` |
| Agent Runs | `GET /agent/runs`, run detail |
| Settings | `GET /settings` (read-only) |

The app proxies API calls through `/backend/*` so the browser never needs direct CORS access to the backend.

## Setup

```bash
cd frontend
cp .env.example .env.local
npm install
```

## Environment

| Variable | Description |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Browser API base path (default `/backend`) |
| `API_URL` | Server-side rewrite target (default `http://localhost:8001`; Compose uses `http://gateway:80`) |

See [`next.config.ts`](next.config.ts) for the rewrite rule.

## Run (local dev)

```bash
# Terminal 1 — backend at repo root
.venv/bin/python -m uvicorn app.main:app --reload --port 8001

# Terminal 2 — frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Docker

Built and served by the root `docker-compose.yml` `frontend` service on port `3000`. API traffic is rewritten to the gateway service.

## Scripts

```bash
npm run dev
npm run build
npm run start
npm run lint
npm run typecheck
```

## Documentation

Project overview and deployment: [../README.md](../README.md).
