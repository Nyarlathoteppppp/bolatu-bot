#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_PROXY_PORT="${LOCAL_CODEX_PROXY_PORT:-7897}"

if ! /usr/bin/nc -z 127.0.0.1 "${LOCAL_PROXY_PORT}" >/dev/null 2>&1; then
  LOCAL_PROXY_PORT="${LOCAL_CODEX_PROXY_FALLBACK_PORT:-7898}"
  nohup /usr/bin/python3 \
    "${PROJECT_DIR}/scripts/direct_http_proxy.py" \
    --host 127.0.0.1 \
    --port "${LOCAL_PROXY_PORT}" \
    >> "${PROJECT_DIR}/logs/direct-http-proxy.log" \
    2>> "${PROJECT_DIR}/logs/direct-http-proxy.err.log" &
fi

for _ in 1 2 3 4 5; do
  if /usr/bin/nc -z 127.0.0.1 "${LOCAL_PROXY_PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! /usr/bin/nc -z 127.0.0.1 "${LOCAL_PROXY_PORT}" >/dev/null 2>&1; then
  echo "local Codex proxy did not start on 127.0.0.1:${LOCAL_PROXY_PORT}" >&2
  exit 1
fi

ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=8 \
  qqbot-server \
  'timeout 5s sh -c '"'"'pids=$(lsof -tiTCP:7897 -sTCP:LISTEN 2>/dev/null || true); if [ -n "$pids" ]; then kill $pids || true; sleep 1; fi'"'"'' \
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
