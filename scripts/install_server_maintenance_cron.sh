#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/opt/qq-social-agent}"
hygiene_cron="${HYGIENE_CRON_PATH:-/etc/cron.d/qq-social-agent-hygiene}"
cos_cron="${COS_CRON_PATH:-/etc/cron.d/qq-social-agent-cos-backup}"
health_cron="${HEALTH_CRON_PATH:-/etc/cron.d/qq-social-agent-health}"
cos_env="${COS_ENV_PATH:-/etc/qq-social-agent/cos-backup.env}"

sudo tee "$hygiene_cron" >/dev/null <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Daily local cleanup for qq-social-agent. Keeps logs, backups, Docker cache and NapCat temp files bounded.
17 4 * * * ubuntu cd $project_dir && BACKUP_DELETE_DAYS=90 JOURNAL_VACUUM_SIZE=200M scripts/system_hygiene.sh --apply >> logs/system_hygiene_cron.log 2>&1
EOF
sudo chmod 0644 "$hygiene_cron"

sudo tee "$health_cron" >/dev/null <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Weekly health report. The latest report is reports/server_health_latest.md.
12 6 * * 1 ubuntu cd $project_dir && scripts/server_health_report.sh >> logs/server_health_cron.log 2>&1
EOF
sudo chmod 0644 "$health_cron"

if [[ -f "$cos_env" ]]; then
  sudo tee "$cos_cron" >/dev/null <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Daily COS cold backup. Configure $cos_env first.
42 4 * * * ubuntu set -a; source $cos_env; set +a; cd $project_dir && scripts/cos_backup.sh --apply >> logs/cos_backup_cron.log 2>&1
# Daily gradual NapCat ntqq cold backup. Uploads only a bounded file batch each run.
42 5 * * * ubuntu set -a; source $cos_env; set +a; cd $project_dir && scripts/cos_ntqq_gradual_backup.sh --apply >> logs/cos_ntqq_gradual_cron.log 2>&1
EOF
  sudo chmod 0644 "$cos_cron"
  echo "Installed COS backup cron: $cos_cron"
else
  echo "COS env not found, skipped COS backup cron: $cos_env"
  echo "Create it from scripts/cos-backup.env.example, then rerun this script."
fi

echo "Installed hygiene cron: $hygiene_cron"
echo "Installed health cron: $health_cron"
