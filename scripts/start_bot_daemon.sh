#!/usr/bin/env bash
set -euo pipefail

cd /Users/ywbw/qq-social-agent
mkdir -p logs

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e '.[dev]'"
  exit 1
fi

uid="$(id -u)"
label="com.ywbw.qq-social-agent"
plist="/Users/ywbw/qq-social-agent/scripts/${label}.plist"
listener_pid="$(lsof -n -P -tiTCP@127.0.0.1:8080 -sTCP:LISTEN | head -n 1 || true)"

if [ -n "$listener_pid" ]; then
  echo "Bot already listening on 8080."
  ps -p "$listener_pid" -o pid,ppid,stat,comm,args || true
  exit 0
fi

launchctl bootout "gui/${uid}/${label}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${uid}" "$plist"
launchctl kickstart -k "gui/${uid}/${label}"
echo "Started bot launch agent: ${label}"
