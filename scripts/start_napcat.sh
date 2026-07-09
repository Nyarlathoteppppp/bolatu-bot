#!/usr/bin/env bash
set -euo pipefail

qq_pattern="/Applications/QQ[.]app/Contents/(MacOS/QQ|Frameworks/QQ Helper[.]app)"
node_pattern="/Applications/QQ[.]app/Contents/Frameworks/QQ Helper[.]app/Contents/MacOS/QQ Helper.*NodeService"
webui_url="http://127.0.0.1:6099/webui"

print_webui_urls() {
  local urls
  urls="$(lsof -n -P -iTCP -sTCP:LISTEN 2>/dev/null | awk '/QQ\\x20Hel/ && /TCP \*:(6099|61[0-9][0-9]) / {
    sub(/^.*TCP \*:/, "");
    sub(/ .*/, "");
    print "http://127.0.0.1:" $0 "/webui"
  }' | sort -u)"
  if [ -n "$urls" ]; then
    echo "$urls"
  else
    echo "$webui_url"
  fi
}

cleanup_duplicate_instances() {
  local keep_helpers helper parent
  keep_helpers="$(lsof -n -P -iTCP@127.0.0.1:8080 -sTCP:ESTABLISHED 2>/dev/null | awk '/QQ\\x20Hel/ {print $2}' | sort -u | xargs || true)"
  if [ -z "$keep_helpers" ]; then
    return 1
  fi

  while read -r helper; do
    [ -n "$helper" ] || continue
    case " $keep_helpers " in
      *" $helper "*) continue ;;
    esac

    parent="$(ps -o ppid= -p "$helper" 2>/dev/null | tr -d ' ')"
    echo "Stopping stale QQ/NapCat instance: qq_pid=${parent:-unknown} helper_pid=${helper}"
    [ -n "$parent" ] && kill -TERM "$parent" >/dev/null 2>&1 || true
    kill -TERM "$helper" >/dev/null 2>&1 || true
  done < <(pgrep -f "$node_pattern" || true)

  sleep 2

  while read -r helper; do
    [ -n "$helper" ] || continue
    case " $keep_helpers " in
      *" $helper "*) continue ;;
    esac

    parent="$(ps -o ppid= -p "$helper" 2>/dev/null | tr -d ' ')"
    [ -n "$parent" ] && kill -KILL "$parent" >/dev/null 2>&1 || true
    kill -KILL "$helper" >/dev/null 2>&1 || true
  done < <(pgrep -f "$node_pattern" || true)

  return 0
}

echo "Stopping existing QQ/NapCat processes..."
osascript -e 'tell application "QQ" to quit' >/dev/null 2>&1 || true

for _ in {1..20}; do
  if ! pgrep -f "$qq_pattern" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if pgrep -f "$qq_pattern" >/dev/null 2>&1; then
  pkill -TERM -f "$qq_pattern" >/dev/null 2>&1 || true
  sleep 2
fi

if pgrep -f "$qq_pattern" >/dev/null 2>&1; then
  pkill -KILL -f "$qq_pattern" >/dev/null 2>&1 || true
  sleep 1
fi

echo "Starting QQ/NapCat..."
open -a /Applications/QQ.app --args --no-sandbox
echo "NapCat WebUI: ${webui_url}"

echo "Waiting for NapCat to connect to bot..."
for _ in {1..75}; do
  if cleanup_duplicate_instances; then
    echo "NapCat is connected to bot."
    echo "NapCat WebUI:"
    print_webui_urls
    exit 0
  fi
  sleep 1
done

echo "NapCat started, but no bot websocket connection was detected yet."
echo "Open one of these WebUI URLs, finish login, then run: /Users/ywbw/qq-social-agent/scripts/status.sh"
print_webui_urls
