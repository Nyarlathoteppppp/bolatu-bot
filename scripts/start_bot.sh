#!/usr/bin/env bash
set -euo pipefail

cd /Users/ywbw/qq-social-agent

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e '.[dev]'"
  exit 1
fi

source .venv/bin/activate
exec python bot.py
