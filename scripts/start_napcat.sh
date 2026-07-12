#!/usr/bin/env bash
set -euo pipefail

project_dir="/opt/qq-social-agent"
compose=(docker compose -p qq-social-agent -f docker-compose.server.yml)

cd "$project_dir"

echo "Warning: starting or recreating NapCat may require QQ login or trigger account verification."
echo "This script only starts NapCat when invoked explicitly."
"${compose[@]}" up -d --no-deps napcat
"${compose[@]}" ps napcat
echo "NapCat WebUI is bound to http://127.0.0.1:6099/webui"
