#!/usr/bin/env bash
set -euo pipefail

cd /Users/ywbw/qq-social-agent

echo "Project: /Users/ywbw/qq-social-agent"
echo "Venv:    /Users/ywbw/qq-social-agent/.venv"
echo

echo "DeepSeek model:"
.venv/bin/python - <<'PY'
from qq_social_agent.config import load_config
config = load_config().deepseek
print(f"base={config.model} / thinking={config.thinking}")
print(f"decision={config.decision_model}")
print(f"reply={config.reply_model}")
print(f"utility={config.utility_model}")
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
