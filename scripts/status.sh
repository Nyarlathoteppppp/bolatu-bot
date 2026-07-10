#!/usr/bin/env bash
set -euo pipefail

cd /Users/ywbw/qq-social-agent

echo "Project: /Users/ywbw/qq-social-agent"
echo "Venv:    /Users/ywbw/qq-social-agent/.venv"
echo

echo "LLM routes:"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from qq_social_agent.config import load_config
from qq_social_agent.memory import MemoryStore

config = load_config().deepseek
memory = MemoryStore(Path("/Users/ywbw/qq-social-agent/data/bot.sqlite3"))
raw_overrides = memory.app_kv_get("llm_model_route_overrides") or "{}"
try:
    overrides = json.loads(raw_overrides)
except json.JSONDecodeError:
    overrides = {}

print("providers=" + ", ".join(sorted(config.providers)))
print(f"thinking={config.thinking}")
for route_name, title in (
    ("decision", "decision"),
    ("reply", "reply"),
    ("jargon", "jargon"),
    ("memory", "memory"),
    ("style", "style"),
    ("member_profile", "member_profile"),
):
    active = overrides.get(route_name, config.routes[route_name].label)
    marker = " override" if route_name in overrides else ""
    print(f"{title}={active}{marker}")
    print(f"  config={config.routes[route_name].label}")
    print(f"  fallback={config.fallback_routes[route_name].label}")
print("available_models=" + ", ".join(route.label for route in config.model_catalog))
PY
echo

echo "Bot listener on 127.0.0.1:8080:"
lsof -n -P -iTCP@127.0.0.1:8080 || true
listener_pid="$(lsof -n -P -tiTCP@127.0.0.1:8080 -sTCP:LISTEN | head -n 1 || true)"
if [ -n "$listener_pid" ]; then
  echo
  echo "Bot listener process:"
  ps -p "$listener_pid" -o pid,ppid,stat,tty,comm,args || true
else
  bot_pid="$(pgrep -f "/Users/ywbw/qq-social-agent/.venv/bin/python -u bot.py|/Users/ywbw/qq-social-agent/bot.py" | head -n 1 || true)"
  if [ -n "$bot_pid" ]; then
  echo
  echo "Bot process exists but is not listening on 8080:"
  ps -p "$bot_pid" -o pid,ppid,stat,tty,comm,args || true
  fi
fi
echo

echo "NapCat WebUI listeners:"
webui_listeners="$(lsof -n -P -iTCP -sTCP:LISTEN | awk '/QQ\\x20Hel/ && /TCP \*:(6099|61[0-9][0-9]) / {print}')"
if [ -n "$webui_listeners" ]; then
  echo "$webui_listeners"
  echo
  echo "NapCat WebUI URLs:"
  echo "$webui_listeners" | awk '{
    sub(/^.*TCP \*:/, "");
    sub(/ .*/, "");
    print "http://127.0.0.1:" $0 "/webui"
  }'
else
  echo "(none)"
fi
