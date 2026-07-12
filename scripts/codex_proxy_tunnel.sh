#!/usr/bin/env bash
set -euo pipefail

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
  -R 127.0.0.1:7897:127.0.0.1:7897 \
  qqbot-server
