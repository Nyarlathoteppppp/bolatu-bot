#!/usr/bin/env bash
set -euo pipefail

project_dir="/opt/qq-social-agent"
compose=(docker compose -p qq-social-agent -f docker-compose.server.yml)

cd "$project_dir"

echo "Project: $project_dir"
echo
"${compose[@]}" ps

echo
echo "Configured LLM routes and safe runtime overrides:"
if "${compose[@]}" ps --status running --services | grep -qx bot; then
  "${compose[@]}" exec -T bot python - <<'PY'
import json
from pathlib import Path

from qq_social_agent.config import load_config
from qq_social_agent.memory import MemoryStore

config = load_config().deepseek
memory = MemoryStore(Path("/app/data/bot.sqlite3"))
raw_overrides = memory.app_kv_get("llm_model_route_overrides") or "{}"
try:
    overrides = json.loads(raw_overrides)
except (TypeError, json.JSONDecodeError):
    overrides = {}

print("providers=" + ", ".join(sorted(config.providers)))
print(f"thinking={config.thinking}")
for route_name in ("decision", "reply", "utility", "jargon", "memory", "style", "member_profile"):
    active = overrides.get(route_name, config.routes[route_name].label)
    marker = " override" if route_name in overrides else ""
    print(f"{route_name}={active}{marker}")
    print(f"  config={config.routes[route_name].label}")
    print(f"  fallback={config.fallback_routes[route_name].label}")
print("available_models=" + ", ".join(route.label for route in config.model_catalog))
PY
else
  echo "bot service is not running"
fi

echo
echo "Recent bot logs:"
"${compose[@]}" logs --tail=20 bot

echo
echo "Recent NapCat connection/login events:"
"${compose[@]}" logs --tail=120 napcat 2>&1 \
  | grep -Ei 'websocket|reverse|connect|连接|login|登录|kick|下线|risk|风险|error|failed|失败' \
  | tail -20 \
  || true
