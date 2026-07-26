#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/opt/qq-social-agent}"
cos_env="${COS_ENV_PATH:-/etc/qq-social-agent/cos-backup.env}"
cd "$project_dir"
mkdir -p logs data/history/raw data/history/memory data/history/manifests

if [[ -f "$cos_env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$cos_env"
  set +a
fi

python3 scripts/daily_history_archive.py "$@"
