#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/opt/qq-social-agent}"
mode="${1:---dry-run}"

if [[ "$mode" != "--dry-run" && "$mode" != "--apply" ]]; then
  echo "Usage: $0 [--dry-run|--apply]" >&2
  exit 2
fi

dry_run=1
if [[ "$mode" == "--apply" ]]; then
  dry_run=0
fi

cd "$project_dir"
mkdir -p reports logs

timestamp="$(date +%Y%m%d_%H%M%S)"
report_path="reports/db_hygiene_${timestamp}.md"

echo "== QQ bot system hygiene =="
echo "project: $project_dir"
echo "mode: $mode"
echo

echo "== disk before =="
df -h / "$project_dir" || true
echo

echo "== docker before =="
docker system df || true
echo

echo "== database hygiene =="
db_args=(scripts/db_hygiene.py --report "$report_path")
if [[ "$dry_run" == "1" ]]; then
  db_args+=(--dry-run --no-vacuum)
elif [[ "${DB_VACUUM:-0}" != "1" ]]; then
  db_args+=(--no-vacuum)
fi
python3 "${db_args[@]}"
echo

echo "== backup hygiene =="
backup_compress_days="${BACKUP_COMPRESS_DAYS:-1}"
backup_delete_days="${BACKUP_DELETE_DAYS:-180}"
if [[ "$dry_run" == "1" ]]; then
  echo "Backups to compress (*.bak older than ${backup_compress_days}d):"
  find data -maxdepth 1 -type f -name "*.bak" -mtime +"$backup_compress_days" -print || true
  echo "Compressed backups to delete (*.bak.gz older than ${backup_delete_days}d):"
  find data -maxdepth 1 -type f -name "*.bak.gz" -mtime +"$backup_delete_days" -print || true
else
  find data -maxdepth 1 -type f -name "*.bak" -mtime +"$backup_compress_days" -exec gzip -9 {} \; || true
  find data -maxdepth 1 -type f -name "*.bak.gz" -mtime +"$backup_delete_days" -delete || true
fi
echo

if [[ "${DOCKER_PRUNE:-1}" == "1" ]]; then
  echo "== docker prune =="
  prune_until="${DOCKER_PRUNE_UNTIL:-24h}"
  if [[ "$dry_run" == "1" ]]; then
    echo "Would prune Docker builder/image cache older than ${prune_until}."
  else
    docker builder prune -af --filter "until=${prune_until}" || true
    docker image prune -af --filter "until=${prune_until}" || true
  fi
  echo
fi

echo "== NapCat temp/log hygiene =="
napcat_temp_days="${NAPCAT_TEMP_DAYS:-14}"
napcat_log_days="${NAPCAT_LOG_DAYS:-30}"
napcat_media_days="${NAPCAT_MEDIA_DAYS:-180}"
if [[ "$dry_run" == "1" ]]; then
  echo "Temp files older than ${napcat_temp_days}d:"
  find server-data/ntqq -path "*/NapCat/temp/*" -type f -mtime +"$napcat_temp_days" -print 2>/dev/null || true
  echo "Log files older than ${napcat_log_days}d:"
  find server-data/ntqq -path "*/nt_data/log/*" -type f -mtime +"$napcat_log_days" -print 2>/dev/null || true
  if [[ "${NAPCAT_MEDIA_CLEAN:-0}" == "1" ]]; then
    echo "Media files older than ${napcat_media_days}d:"
    find server-data/ntqq \( -path "*/nt_data/Pic/*" -o -path "*/nt_data/Video/*" -o -path "*/nt_data/Ptt/*" \) -type f -mtime +"$napcat_media_days" -print 2>/dev/null || true
  else
    echo "Media cleanup disabled. Set NAPCAT_MEDIA_CLEAN=1 to include old Pic/Video/Ptt files."
  fi
else
  find server-data/ntqq -path "*/NapCat/temp/*" -type f -mtime +"$napcat_temp_days" -delete 2>/dev/null || true
  find server-data/ntqq -path "*/nt_data/log/*" -type f -mtime +"$napcat_log_days" -delete 2>/dev/null || true
  if [[ "${NAPCAT_MEDIA_CLEAN:-0}" == "1" ]]; then
    find server-data/ntqq \( -path "*/nt_data/Pic/*" -o -path "*/nt_data/Video/*" -o -path "*/nt_data/Ptt/*" \) -type f -mtime +"$napcat_media_days" -delete 2>/dev/null || true
  fi
fi
echo

echo "== docker after =="
docker system df || true
echo

echo "== disk after =="
df -h / "$project_dir" || true
echo

echo "DB report: $project_dir/$report_path"
