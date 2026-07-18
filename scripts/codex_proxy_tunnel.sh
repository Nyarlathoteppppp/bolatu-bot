#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_PROXY_PORT="${LOCAL_CODEX_PROXY_PORT:-7898}"

if ! /usr/sbin/lsof -nP -iTCP:"${LOCAL_PROXY_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  nohup /usr/bin/python3 \
    "${PROJECT_DIR}/scripts/direct_http_proxy.py" \
    --host 127.0.0.1 \
    --port "${LOCAL_PROXY_PORT}" \
    >> "${PROJECT_DIR}/logs/direct-http-proxy.log" \
    2>> "${PROJECT_DIR}/logs/direct-http-proxy.err.log" &
  sleep 1
fi

ssh qqbot-server \
  'pids=$(sudo lsof -tiTCP:7897 -sTCP:LISTEN 2>/dev/null || true); if [ -n "$pids" ]; then sudo kill $pids || true; sleep 1; fi' \
  >/dev/null 2>&1 || true

exec /usr/bin/ssh \
  -N \
  -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes \
  -R 127.0.0.1:7897:127.0.0.1:"${LOCAL_PROXY_PORT}" \
  qqbot-server
