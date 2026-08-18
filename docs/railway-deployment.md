# Railway Deployment Guide

Deploy the Agentic RAG stack on Railway as **four services** while preserving the architecture:

```text
Browser → Frontend (Next.js) → Gateway (nginx) → Backend (FastAPI) → Qdrant
```

Only the **frontend** service receives a public domain. Backend, gateway, and Qdrant use private networking.

---

## Services overview

| Service | Root directory | Dockerfile | Public domain |
|---------|----------------|------------|---------------|
| **qdrant** | — (Docker image) | `qdrant/qdrant:v1.12.5` | No |
| **backend** | `/` (repo root) | `Dockerfile` | No |
| **gateway** | `/` (repo root) | `deploy/Dockerfile.gateway` | No |
| **frontend** | `/frontend` | `frontend/Dockerfile` | **Yes** |

---

## 1. Qdrant service

**Image:** `qdrant/qdrant:v1.12.5`

**Root directory:** N/A (use Railway Docker image deploy or empty with custom image reference)

**Persistent volume:** Mount at `/qdrant/storage` so vector data survives redeploys.

**Environment variables:** None required (uses image defaults).

**Networking:** Private only. Other services reach it at:

```text
http://<qdrant-service-name>.railway.internal:6333
```

**Health:** Qdrant listens on port `6333` inside the container.

---

## 2. Backend service (FastAPI)

**Root directory:** `.` (repository root)

**Dockerfile:** `Dockerfile`

**Public domain:** Do **not** generate one. Use private networking only.

**Persistent volume (recommended):** Mount `/app/data/keyword_index` for the BM25 index.

### Required environment variables

| Variable | Value | Notes |
|----------|-------|-------|
| `APP_ENV` | `production` | Required for production secret validation |
| `GROQ_API_KEY` | `<secret>` | Railway secret |
| `HUGGINGFACE_API_KEY` | `<secret>` | Railway secret |
| `TAVILY_API_KEY` | `<secret>` | Railway secret |
| `WEB_SEARCH_ENABLED` | `true` | Enables Tavily web search |
| `HYBRID_SEARCH_ENABLED` | `true` | Enables BM25 + vector hybrid search |
| `VECTOR_SEARCH_WEIGHT` | `0.5` | Hybrid fusion weight |
| `KEYWORD_SEARCH_WEIGHT` | `0.5` | Hybrid fusion weight |
| `HYBRID_TOP_K` | `10` | Hybrid result count |
| `UVICORN_WORKERS` | `1` | Keep at 1 per Railway instance |
| `QDRANT_URL` | `http://<qdrant>.railway.internal:6333` | Private Qdrant URL |
| `KEYWORD_INDEX_PATH` | `/app/data/keyword_index/index.json` | Use with mounted volume |

### Port behavior

Railway injects `PORT` at runtime. The backend entrypoint listens on:

```text
PORT (Railway) → else APP_PORT (8000) → else 8000
```

Do **not** set `PORT` manually. Local Docker Compose continues to use `APP_PORT=8000`.

### Optional tuning

`GROQ_MODEL`, `LLM_MAX_TOKENS`, `CHUNKING_STRATEGY`, `RETRIEVAL_TOP_K`, `LOG_LEVEL`, etc. — see `.env.example`.

---

## 3. Gateway service (nginx)

**Root directory:** `.` (repository root)

**Dockerfile:** `deploy/Dockerfile.gateway`

**Public domain:** Do **not** generate one. Frontend reaches gateway over private networking.

### Required environment variables

| Variable | Value | Notes |
|----------|-------|-------|
| `BACKEND_UPSTREAM` | `<backend-host>:<backend-port>` | nginx upstream target |

On Railway private networking, use the backend service private hostname and the port the backend listens on (Railway's injected `PORT` for that service). Example pattern:

```text
BACKEND_UPSTREAM=${{backend.RAILWAY_PRIVATE_DOMAIN}}:${{backend.PORT}}
```

(Use Railway variable references in the Railway dashboard — do not hardcode URLs in source code.)

### Port behavior

Railway injects `PORT`. The gateway entrypoint renders nginx config with:

```text
listen PORT (Railway) → else 80 (Docker Compose default)
```

Proxy routes are unchanged: all paths proxy to `BACKEND_UPSTREAM` with the same headers and timeouts as local Compose.

---

## 4. Frontend service (Next.js)

**Root directory:** `frontend`

**Dockerfile:** `frontend/Dockerfile`

**Public domain:** **Generate domain here.** This is the URL users open in the browser.

### Build-time variables (Docker build args)

Set these in Railway **before build** (Variables tab → available at build time):

| Variable | Value | Notes |
|----------|-------|-------|
| `NEXT_PUBLIC_API_URL` | `/backend` | Browser-visible API base path (not a secret) |
| `API_URL` | `http://<gateway-private-host>:<gateway-port>` | Next.js server rewrite target |

Example `API_URL` pattern (Railway variable reference):

```text
http://${{gateway.RAILWAY_PRIVATE_DOMAIN}}:${{gateway.PORT}}
```

The browser calls `/backend/*`; Next.js rewrites to the gateway over private networking. **No backend secrets** are exposed to the browser.

### Runtime variables

| Variable | Value | Notes |
|----------|-------|-------|
| `NODE_ENV` | `production` | Set automatically or explicitly |
| `HOSTNAME` | `0.0.0.0` | Baked in Dockerfile |
| `PORT` | *(injected by Railway)* | Do not set manually |

Do **not** set `API_URL` only at runtime — Next.js bakes rewrite destinations at **build time** in `next.config.ts`. Rebuild the frontend when the gateway private URL changes.

---

## Private networking summary

```text
┌─────────────┐     private      ┌─────────────┐     private      ┌─────────────┐
│  Frontend   │ ───────────────► │   Gateway   │ ───────────────► │   Backend   │
│  (public)   │   API_URL        │  (private)  │ BACKEND_UPSTREAM │  (private)  │
└─────────────┘                  └─────────────┘                  └──────┬──────┘
                                                                         │
                                                                         │ QDRANT_URL
                                                                         ▼
                                                                  ┌─────────────┐
                                                                  │   Qdrant    │
                                                                  │  (private)  │
                                                                  └─────────────┘
```

Enable **Private Networking** for all four services in the Railway project settings.

---

## API_URL and NEXT_PUBLIC_API_URL

| Variable | Where | Purpose |
|----------|-------|---------|
| `NEXT_PUBLIC_API_URL` | Build time | Browser `fetch()` base path → `/backend` |
| `API_URL` | Build time | Next.js rewrite: `/backend/:path*` → gateway |

**Never** set `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, `TAVILY_API_KEY`, or `QDRANT_URL` on the frontend service.

---

## Local Docker Compose (unchanged behavior)

```bash
docker compose up --build -d
```

| URL | Service |
|-----|---------|
| http://localhost:3000 | Frontend |
| http://localhost:8080 | Gateway → Backend |

Compose sets `APP_PORT=8000`, `BACKEND_UPSTREAM=app:8000`, `PORT=3000` (frontend), and gateway listens on `80` internally.

---

## Deployment checklist

1. Create Railway project with four services: qdrant, backend, gateway, frontend.
2. Attach persistent volume to **qdrant** (`/qdrant/storage`) and **backend** (`/app/data/keyword_index`).
3. Enable private networking on all services.
4. Set backend secrets and production config (see table above).
5. Set gateway `BACKEND_UPSTREAM` to backend private host:port.
6. Set frontend build args: `NEXT_PUBLIC_API_URL=/backend`, `API_URL=<gateway private URL>`.
7. Generate public domain **only** on the frontend service.
8. Verify: open frontend URL → Dashboard shows healthy backend via `/backend/health`.

---

## Limitations

- **Four separate Railway services** are required to preserve the nginx gateway layer; a single-service deploy would bypass gateway.
- **Frontend rebuild required** when gateway private URL changes (`API_URL` is build-time).
- **SQLite agent history** (`data/agent_runs.db`) is ephemeral unless a volume is mounted on the backend.
- **BM25 index** requires a persistent volume on the backend service.
- **No HTTPS/TLS** is configured inside the app; Railway terminates TLS on the public frontend domain.
- **Railway `PORT`** differs per service and deploy — always use Railway variable references, not hardcoded ports in source.
