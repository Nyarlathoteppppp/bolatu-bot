#!/usr/bin/env bash
set -euo pipefail

uid="$(id -u)"
label="com.ywbw.qq-social-agent"
plist="/Users/ywbw/qq-social-agent/scripts/${label}.plist"

launchctl bootout "gui/${uid}/${label}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${uid}" "$plist"
launchctl kickstart -k "gui/${uid}/${label}"
echo "Restarted bot launch agent: ${label}"
