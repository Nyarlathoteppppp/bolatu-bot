#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/ssh \
  -N \
  -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o TCPKeepAlive=yes \
  -R 127.0.0.1:7897:127.0.0.1:7897 \
  qqbot-server
