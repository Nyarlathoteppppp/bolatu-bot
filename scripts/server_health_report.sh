#!/usr/bin/env bash
set -uo pipefail

project_dir="${PROJECT_DIR:-/opt/qq-social-agent}"
cd "$project_dir" || exit 2
mkdir -p reports logs

timestamp="$(date +%Y%m%d_%H%M%S)"
report_path="reports/server_health_${timestamp}.md"
cos_env="${COS_ENV_PATH:-/etc/qq-social-agent/cos-backup.env}"
cos_timeout="${HEALTH_COS_TIMEOUT_SECONDS:-60}"

section() {
  printf '\n## %s\n\n' "$1"
}

run_block() {
  local title="$1"
  shift
  section "$title"
  printf '```text\n'
  "$@" 2>&1 || printf 'command_failed: %s\n' "$*"
  printf '```\n'
}

safe_sizes() {
  for path in data data/backups server-data server-data/ntqq reports logs /home/ubuntu/.vscode-server /var/log; do
    output="$(du -sh "$path" 2>/dev/null | head -1 || true)"
    if [[ -n "$output" ]]; then
      printf '%s\n' "$output"
    else
      printf '%s\tunreadable_or_missing\n' "$path"
    fi
  done
}

{
  printf '# QQ Social Agent Health Report\n\n'
  printf 'Generated: %s\n\n' "$(date -Iseconds)"
  printf 'Project: `%s`\n\n' "$project_dir"

  section "Summary"
  printf '```text\n'
  curl -fsS --max-time 5 http://127.0.0.1:8080/readyz || printf 'readyz_failed\n'
  printf '\n'
  df -h / | tail -1
  du -sh data data/backups server-data server-data/ntqq 2>/dev/null || true
  printf '```\n'

  run_block "Docker" docker compose -p qq-social-agent -f docker-compose.server.yml ps
  run_block "Disk" df -h /
  run_block "Memory" free -h

  section "Project Sizes"
  printf '```text\n'
  safe_sizes
  printf '```\n'

  run_block "Journal" journalctl --disk-usage

  section "Cron"
  printf '```text\n'
  cat /etc/cron.d/qq-social-agent-hygiene 2>/dev/null || true
  cat /etc/cron.d/qq-social-agent-cos-backup 2>/dev/null || true
  cat /etc/cron.d/qq-social-agent-health 2>/dev/null || true
  cat /etc/cron.d/qq-social-agent-history-archive 2>/dev/null || true
  printf '```\n'

  section "Recent Logs"
  printf '```text\n'
  for file in logs/system_hygiene_cron.log logs/cos_backup_cron.log logs/cos_ntqq_gradual_cron.log logs/daily_history_archive_cron.log logs/server_health_cron.log; do
    if [[ -f "$file" ]]; then
      echo "--- $file"
      tail -40 "$file"
    fi
  done
  printf '```\n'



  section "History Archive"
  printf '```text\n'
  if compgen -G "data/history/manifests/daily_history_*_manifest.json" >/dev/null; then
    latest_manifest="$(ls -1t data/history/manifests/daily_history_*_manifest.json | head -1)"
    echo "latest_manifest=$latest_manifest"
    python3 - "$latest_manifest" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
print(f"date={data.get('archive_date')} messages={data.get('message_count')} bot={data.get('bot_message_count')} raw_bytes={data.get('raw_archive',{}).get('bytes')} memory_bytes={data.get('memory_archive',{}).get('bytes')}")
PY
  else
    echo "no_history_manifest_yet"
  fi
  du -sh data/history 2>/dev/null || true
  printf '```\n'

  section "COS"
  printf '```text\n'
  if [[ -f "$cos_env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$cos_env"
    set +a
    echo "COS_BACKUP_DEST=${COS_BACKUP_DEST:-unset}"
    if command -v coscli >/dev/null 2>&1 && [[ -n "${COS_BACKUP_DEST:-}" ]]; then
      timeout "$cos_timeout" coscli ls "${COS_BACKUP_DEST%/}/data/backups/" --limit 20 --init-skip --disable-log 2>&1 || echo "cos_backup_list_failed_or_timed_out"
      timeout "$cos_timeout" coscli ls "${COS_BACKUP_DEST%/}/server-data/napcat/config/" --limit 20 --init-skip --disable-log 2>&1 || echo "cos_napcat_config_list_failed_or_timed_out"
      timeout "$cos_timeout" coscli ls "${COS_BACKUP_DEST%/}/server-data/ntqq/" --limit 20 --init-skip --disable-log 2>&1 || echo "cos_ntqq_list_failed_or_timed_out"
    else
      echo "coscli_or_dest_missing"
    fi
  else
    echo "cos_env_missing: $cos_env"
  fi
  printf '```\n'
} > "$report_path"

ln -sfn "$(basename "$report_path")" reports/server_health_latest.md
cat "$report_path"
