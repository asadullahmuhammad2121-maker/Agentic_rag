#!/bin/sh
set -e

APP_PORT="${APP_PORT:-8000}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
UVICORN_TIMEOUT_KEEP_ALIVE="${UVICORN_TIMEOUT_KEEP_ALIVE:-5}"

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${APP_PORT}" \
  --workers "${UVICORN_WORKERS}" \
  --timeout-keep-alive "${UVICORN_TIMEOUT_KEEP_ALIVE}" \
  --proxy-headers \
  --forwarded-allow-ips="*"
