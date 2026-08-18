#!/bin/sh
set -e

# Railway injects PORT at runtime; Docker Compose defaults to 80 inside the container.
LISTEN_PORT="${PORT:-80}"
BACKEND_UPSTREAM="${BACKEND_UPSTREAM:-app:8000}"

sed \
  -e "s/__LISTEN_PORT__/${LISTEN_PORT}/g" \
  -e "s/__BACKEND_UPSTREAM__/${BACKEND_UPSTREAM}/g" \
  /etc/nginx/templates/default.conf.template \
  > /etc/nginx/conf.d/default.conf
