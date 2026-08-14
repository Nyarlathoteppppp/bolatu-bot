# Storage Maintenance

This server is intended to run the QQ bot for a year or longer. Runtime data stays local; Tencent COS is used as cold backup storage.

## Current Layout

- `/opt/qq-social-agent/data/bot.sqlite3`: live SQLite database. Do not put this directly on COS.
- `/opt/qq-social-agent/data/backups`: local compressed DB backups, COS upload snapshots and `manual/` pre-change snapshots.
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
- Daily 00:20: raw chat archive and deterministic historical memory library with `HISTORY_MEMORY_ROUTE=utility scripts/daily_history_archive.sh --date yesterday --skip-llm`.
- Daily 05:42: gradual NapCat `server-data/ntqq` cold backup to COS, bounded by file count and bytes.

## What Is Backed Up

Daily COS backup:

- online SQLite snapshot: messages, memory, RAG, approval data, member profiles, feedback
- manually created pre-change snapshots under `data/backups/manual`
- project metadata archive: `config.yaml`, `prompts`, `scripts`, docs and project metadata
- `server-data/napcat/config`

Daily history archive:

- `data/history/raw/daily_raw_YYYY-MM-DD.jsonl.gz`: canonical chat text preserved from SQLite.
- `data/history/memory/daily_memory_YYYY-MM-DD.json`: deterministic historical recall JSON with speaker attribution, source message ids, active members, long-message entry points and bot samples. LLM enhancement can be run manually when needed.
- `data/history/manifests/daily_history_YYYY-MM-DD_manifest.json`: count, checksum and file metadata.

Gradual COS cold backup:

- `server-data/ntqq`, uploaded in bounded daily batches.

`server-data/ntqq` may contain private QQ cache/media and is large, so it is not synced all at once. The gradual script keeps a cursor in `data/maintenance/ntqq_gradual_state.json` and uploads at most `COS_NTQQ_MAX_BYTES` / `COS_NTQQ_MAX_FILES` per run. Temp/log/crash files are excluded by default via `COS_NTQQ_EXCLUDE_GLOBS`.

## Timeout Policy

COS commands are bounded so maintenance does not hang forever:

- Normal COS sync: `COSCLI_TIMEOUT_SECONDS=900`
- Gradual NapCat ntqq file upload: `COS_NTQQ_FILE_TIMEOUT_SECONDS=300` per file
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

Preview gradual NapCat cold backup:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/cos_ntqq_gradual_backup.sh --dry-run
```

Run one gradual NapCat batch now:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/cos_ntqq_gradual_backup.sh --apply
```

Run daily history archive now:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/daily_history_archive.sh --date yesterday
# If memory route is slow, use a faster route:
# HISTORY_MEMORY_ROUTE=utility scripts/daily_history_archive.sh --date yesterday
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
- Upload DB snapshots, metadata, raw daily chat archives and historical memory JSON to COS daily.
- Keep local COS upload snapshots for 14 days after upload.
- Keep manual pre-change snapshots locally for 30 days after their successful COS upload.
- Sync full NapCat `ntqq` gradually in bounded daily batches, not in one large monthly burst.
- Keep COS objects for the year; bucket space is enough for this project.
- Daily hygiene keeps only the two most recently used VSCode Remote Server builds; stale versions are removed automatically.
- Do not delete `server-data/ntqq` blindly; it contains QQ runtime/login/cache data.
