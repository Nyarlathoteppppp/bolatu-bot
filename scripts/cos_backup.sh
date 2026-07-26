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

dry_run=1
if [[ "$mode" == "--apply" ]]; then
  dry_run=0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${BACKUP_DIR:-data/backups}"
local_keep_days="${COS_LOCAL_SNAPSHOT_KEEP_DAYS:-14}"
mkdir -p "$backup_dir"

snapshot_sqlite="$backup_dir/bot.sqlite3.cos_${timestamp}.sqlite3"
snapshot_gz="$snapshot_sqlite.gz"
metadata_tar="$backup_dir/project_metadata_${timestamp}.tar.gz"

echo "== COS backup =="
echo "project: $project_dir"
echo "dest:    $cos_dest"
echo "mode:    $mode"

if [[ "$dry_run" == "1" ]]; then
  echo "Would create SQLite online backup: $snapshot_gz"
  echo "Would create metadata archive:      $metadata_tar"
  echo "Would run: coscli sync $backup_dir ${cos_dest%/}/data/backups -r"
  if [[ "${COS_INCLUDE_NAPCAT_CONFIG:-1}" == "1" ]]; then
    echo "Would sync server-data/napcat/config to ${cos_dest%/}/server-data/napcat/config"
  fi
  if [[ "${COS_INCLUDE_NAPCAT_NTQQ:-0}" == "1" ]]; then
    echo "Would sync server-data/ntqq to ${cos_dest%/}/server-data/ntqq"
  else
    echo "NapCat ntqq full sync disabled. Set COS_INCLUDE_NAPCAT_NTQQ=1 only for a deliberate full cold backup."
  fi
  echo "Would delete local COS snapshots older than ${local_keep_days}d after a successful upload."
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
tar -czf "$metadata_tar"   config.yaml prompts scripts README.md pyproject.toml   server-data/napcat/config   2>/dev/null || true

echo "created: $snapshot_gz"
echo "created: $metadata_tar"

coscli sync "$backup_dir" "${cos_dest%/}/data/backups" -r

if [[ "${COS_INCLUDE_NAPCAT_CONFIG:-1}" == "1" ]]; then
  coscli sync server-data/napcat/config "${cos_dest%/}/server-data/napcat/config" -r || true
fi

if [[ "${COS_INCLUDE_NAPCAT_NTQQ:-0}" == "1" ]]; then
  echo "NapCat ntqq sync enabled. This may upload private QQ cache/media and can be large."
  coscli sync server-data/ntqq "${cos_dest%/}/server-data/ntqq" -r || true
fi

find "$backup_dir" -maxdepth 1 -type f -name "bot.sqlite3.cos_*.sqlite3.gz" -mtime +"$local_keep_days" -delete || true
find "$backup_dir" -maxdepth 1 -type f -name "project_metadata_*.tar.gz" -mtime +"$local_keep_days" -delete || true

echo "COS backup done."
