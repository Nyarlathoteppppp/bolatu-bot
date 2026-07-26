# Storage Maintenance

This server is intended to run the QQ bot for a year or longer. Runtime data stays local; Tencent COS is used as cold backup storage.

## Current Layout

- `/opt/qq-social-agent/data/bot.sqlite3`: live SQLite database. Do not put this directly on COS.
- `/opt/qq-social-agent/data/backups`: local compressed DB backups and COS upload snapshots.
- `/opt/qq-social-agent/server-data`: NapCat and QQ runtime data.
- `/swapfile`: 2G disk-backed swap for memory spikes.

COS is object storage, not a mounted block disk. Use it for backup and recovery, not live SQLite writes.

## Automatic Schedule

Install or refresh cron files:

```bash
cd /opt/qq-social-agent
scripts/install_server_maintenance_cron.sh
```

Installed jobs:

- Daily 04:17: local cleanup with `scripts/system_hygiene.sh --apply`.
- Daily 04:42: COS backup with `scripts/cos_backup.sh --apply`.
- Weekly Monday 06:12: health report with `scripts/server_health_report.sh`.
- Monthly day 1 05:42: NapCat `server-data/ntqq` cold sync to COS.

## What Is Backed Up

Daily COS backup:

- online SQLite snapshot: messages, memory, RAG, approval data, member profiles, feedback
- project metadata archive: `config.yaml`, `prompts`, `scripts`, docs and project metadata
- `server-data/napcat/config`

Monthly COS cold sync:

- `server-data/ntqq`

`server-data/ntqq` may contain private QQ cache/media and is large, so it is monthly rather than daily.

## Timeout Policy

COS commands are bounded so maintenance does not hang forever:

- Normal COS sync: `COSCLI_TIMEOUT_SECONDS=900`
- Monthly NapCat ntqq sync: `COSCLI_NTQQ_TIMEOUT_SECONDS=7200`
- Health-check COS listing: `HEALTH_COS_TIMEOUT_SECONDS=60`

## Manual Commands

Preview daily COS backup:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/cos_backup.sh --dry-run
```

Run daily COS backup now:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/cos_backup.sh --apply
```

Run monthly NapCat cold sync now:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
COS_INCLUDE_NAPCAT_NTQQ=1 COSCLI_NTQQ_TIMEOUT_SECONDS=7200 scripts/cos_backup.sh --apply
```

Generate health report now:

```bash
cd /opt/qq-social-agent
scripts/server_health_report.sh
```

Latest report:

```bash
/opt/qq-social-agent/reports/server_health_latest.md
```

## Year-Long Policy

- Keep live DB local for correctness and speed.
- Keep local DB backups for about 90 days.
- Upload DB snapshots and metadata to COS daily.
- Keep local COS upload snapshots for 14 days after upload.
- Sync full NapCat `ntqq` monthly, not daily.
- Keep COS objects for the year; bucket space is enough for this project.
- Clean old VSCode server versions manually when `.vscode-server` exceeds a few GB.
- Do not delete `server-data/ntqq` blindly; it contains QQ runtime/login/cache data.
