#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/opt/qq-social-agent}"
mode="${1:---dry-run}"

if [[ "$mode" != "--dry-run" && "$mode" != "--apply" ]]; then
  echo "Usage: $0 [--dry-run|--apply]" >&2
  exit 2
fi

cd "$project_dir"

cos_dest="${COS_BACKUP_DEST:-}"
if [[ -z "$cos_dest" ]]; then
  cat >&2 <<'EOF'
COS_BACKUP_DEST is required.
Example:
  COS_BACKUP_DEST=cos://your-bucket-1250000000/qq-social-agent scripts/cos_backup.sh --apply
Configure COSCLI first with:
  coscli config init
EOF
  exit 2
fi

if [[ "$mode" == "--apply" ]] && ! command -v coscli >/dev/null 2>&1; then
  echo "coscli not found. Install it first with scripts/install_coscli.sh." >&2
  exit 127
fi

if [[ "$mode" == "--apply" ]] && ! command -v timeout >/dev/null 2>&1; then
  echo "timeout command not found; refusing to run unbounded COS backup." >&2
  exit 127
fi

dry_run=1
if [[ "$mode" == "--apply" ]]; then
  dry_run=0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${BACKUP_DIR:-data/backups}"
local_keep_days="${COS_LOCAL_SNAPSHOT_KEEP_DAYS:-14}"
manual_keep_days="${COS_LOCAL_MANUAL_BACKUP_KEEP_DAYS:-30}"
coscli_timeout_seconds="${COSCLI_TIMEOUT_SECONDS:-900}"
coscli_ntqq_timeout_seconds="${COSCLI_NTQQ_TIMEOUT_SECONDS:-7200}"
coscli_err_retry_num="${COSCLI_ERR_RETRY_NUM:-2}"
coscli_routines="${COSCLI_ROUTINES:-2}"
mkdir -p "$backup_dir" logs

snapshot_sqlite="$backup_dir/bot.sqlite3.cos_${timestamp}.sqlite3"
snapshot_gz="$snapshot_sqlite.gz"
metadata_tar="$backup_dir/project_metadata_${timestamp}.tar.gz"

coscli_sync() {
  local timeout_seconds="$1"
  shift
  timeout "$timeout_seconds" coscli sync "$@" \
    --init-skip \
    --disable-log \
    --err-retry-num "$coscli_err_retry_num" \
    --routines "$coscli_routines" \
    --fail-output=false \
    --process-log=false
}

echo "== COS backup =="
echo "project: $project_dir"
echo "dest:    $cos_dest"
echo "mode:    $mode"
echo "timeout: normal=${coscli_timeout_seconds}s ntqq=${coscli_ntqq_timeout_seconds}s"

if [[ "$dry_run" == "1" ]]; then
  echo "Would create SQLite online backup: $snapshot_gz"
  echo "Would create metadata archive:      $metadata_tar"
  echo "Would sync $backup_dir to ${cos_dest%/}/data/backups"
  if [[ "${COS_INCLUDE_NAPCAT_CONFIG:-1}" == "1" ]]; then
    echo "Would sync server-data/napcat/config to ${cos_dest%/}/server-data/napcat/config"
  fi
  if [[ "${COS_INCLUDE_NAPCAT_NTQQ:-0}" == "1" ]]; then
    echo "Would sync server-data/ntqq to ${cos_dest%/}/server-data/ntqq"
  else
    echo "NapCat ntqq full sync disabled. Set COS_INCLUDE_NAPCAT_NTQQ=1 only for a deliberate full cold backup."
  fi
  echo "Would delete local COS snapshots older than ${local_keep_days}d after a successful upload."
  echo "Would delete uploaded manual backups older than ${manual_keep_days}d."
  exit 0
fi

python3 - "$snapshot_sqlite" <<'PY'
import sqlite3
import sys
from pathlib import Path
src = Path("data/bot.sqlite3")
dst = Path(sys.argv[1])
with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
    source.backup(target)
PY

gzip -9 "$snapshot_sqlite"
tar -czf "$metadata_tar" \
  config.yaml prompts scripts README.md pyproject.toml \
  server-data/napcat/config \
  2>/dev/null || true

echo "created: $snapshot_gz"
echo "created: $metadata_tar"

coscli_sync "$coscli_timeout_seconds" "$backup_dir" "${cos_dest%/}/data/backups" -r

if [[ "${COS_INCLUDE_NAPCAT_CONFIG:-1}" == "1" ]]; then
  coscli_sync "$coscli_timeout_seconds" server-data/napcat/config "${cos_dest%/}/server-data/napcat/config" -r || true
fi

if [[ "${COS_INCLUDE_NAPCAT_NTQQ:-0}" == "1" ]]; then
  echo "NapCat ntqq sync enabled. This may upload private QQ cache/media and can be large."
  coscli_sync "$coscli_ntqq_timeout_seconds" server-data/ntqq "${cos_dest%/}/server-data/ntqq" -r || true
fi

find "$backup_dir" -maxdepth 1 -type f -name "bot.sqlite3.cos_*.sqlite3.gz" -mtime +"$local_keep_days" -delete || true
find "$backup_dir" -maxdepth 1 -type f -name "project_metadata_*.tar.gz" -mtime +"$local_keep_days" -delete || true
# Manual snapshots are only cleaned after the successful backup-dir sync above.
find "$backup_dir/manual" -type f -mtime +"$manual_keep_days" -delete 2>/dev/null || true
find "$backup_dir/manual" -type d -empty -delete 2>/dev/null || true
find coscli_output -type f -mtime +14 -delete 2>/dev/null || true
find coscli_output -type d -empty -delete 2>/dev/null || true

echo "COS backup done."
