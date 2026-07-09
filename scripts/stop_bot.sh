#!/usr/bin/env bash
set -euo pipefail

uid="$(id -u)"
label="com.ywbw.qq-social-agent"

launchctl bootout "gui/${uid}/${label}" >/dev/null 2>&1 || true

screen -S qq-social-agent -X quit >/dev/null 2>&1 || true

for pid in $(lsof -n -P -tiTCP@127.0.0.1:8080 -sTCP:LISTEN || true); do
  echo "Stopping bot listener process: ${pid}"
  kill "$pid" 2>/dev/null || true
done

sleep 1

for pid in $(lsof -n -P -tiTCP@127.0.0.1:8080 -sTCP:LISTEN || true); do
  echo "Force stopping bot listener process: ${pid}"
  kill -KILL "$pid" 2>/dev/null || true
done

echo "Stopped bot launch agent: ${label}"
