# Agentic RAG Frontend (Phase 1)

Next.js dashboard and agent chat UI for the FastAPI backend.

## Setup

```bash
cd frontend
cp .env.example .env.local
npm install
```

## Environment

| Variable | Description |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Browser-facing API base (default `/backend` via Next.js rewrite) |
| `API_URL` | Server-side proxy target (default `http://localhost:8001`) |

The frontend proxies `/backend/*` to the FastAPI backend so CORS changes are not required.

## Run

```bash
# Terminal 1 — backend
cd ..
.venv/bin/python -m uvicorn app.main:app --reload --port 8001

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Scripts

```bash
npm run dev
npm run build
npm run start
npm run lint
npm run typecheck
```

## Phase 1 scope

- Application shell with responsive sidebar
- Dashboard (real health data + clearly marked placeholders)
- Agent Chat (`POST /agent/query`)
- Placeholder nav pages: Documents, Retrieval, Agent Runs, Settings
