#!/usr/bin/env bash
set -euo pipefail

project_dir="/opt/qq-social-agent"
compose=(docker compose -p qq-social-agent -f docker-compose.server.yml)

cd "$project_dir"
exec "${compose[@]}" up --no-deps bot
