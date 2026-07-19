#!/usr/bin/env bash
set -euo pipefail

project_dir="/opt/qq-social-agent"
compose=(docker compose -p qq-social-agent -f docker-compose.server.yml)

cd "$project_dir"
"${compose[@]}" up -d --build --no-deps bot
"${compose[@]}" ps bot

if [[ "${BOT_RESTART_PRUNE_DOCKER:-1}" == "1" ]]; then
  prune_until="${BOT_RESTART_PRUNE_UNTIL:-24h}"
  echo "Pruning Docker build cache older than ${prune_until}..."
  docker builder prune -af --filter "until=${prune_until}" >/dev/null || true
  docker image prune -af --filter "until=${prune_until}" >/dev/null || true
fi
