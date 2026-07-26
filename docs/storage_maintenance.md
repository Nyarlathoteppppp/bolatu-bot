# Storage Maintenance

This server is intended to run the QQ bot for a long time. Keep runtime data local, and use Tencent COS as cold backup storage.

## Current Layout

- `/opt/qq-social-agent/data/bot.sqlite3`: live SQLite database. Do not put this directly on COS.
- `/opt/qq-social-agent/data/backups`: local compressed DB backups and COS upload snapshots.
- `/opt/qq-social-agent/server-data`: NapCat and QQ runtime data.
- `/swapfile`: 2G disk-backed swap for memory spikes.

COS is object storage, not a mounted block disk. Use it for backups and old artifacts, not live SQLite writes.

## Local Hygiene

Install the daily cron:

```bash
cd /opt/qq-social-agent
scripts/install_server_maintenance_cron.sh
```

The cron runs daily at 04:17:

```bash
BACKUP_DELETE_DAYS=90 JOURNAL_VACUUM_SIZE=200M scripts/system_hygiene.sh --apply
```

It bounds:

- database/RAG garbage
- local compressed backups
- system journal
- Docker unused cache/images
- NapCat temp/log files

NapCat media cleanup is disabled by default because QQ media can be useful history. Run deliberately:

```bash
cd /opt/qq-social-agent
NAPCAT_MEDIA_CLEAN=1 NAPCAT_MEDIA_DAYS=180 scripts/system_hygiene.sh --dry-run
NAPCAT_MEDIA_CLEAN=1 NAPCAT_MEDIA_DAYS=180 scripts/system_hygiene.sh --apply
```

## COS Backup

Install COSCLI:

```bash
cd /opt/qq-social-agent
scripts/install_coscli.sh
```

Configure COSCLI interactively:

```bash
coscli config init
```

Create COS backup env:

```bash
sudo mkdir -p /etc/qq-social-agent
sudo cp scripts/cos-backup.env.example /etc/qq-social-agent/cos-backup.env
sudo chmod 600 /etc/qq-social-agent/cos-backup.env
sudo nano /etc/qq-social-agent/cos-backup.env
```

Set at least:

```bash
COS_BACKUP_DEST=cos://your-bucket-1250000000/qq-social-agent
```

Then enable daily COS cron:

```bash
cd /opt/qq-social-agent
scripts/install_server_maintenance_cron.sh
```

Manual preview and upload:

```bash
cd /opt/qq-social-agent
set -a; source /etc/qq-social-agent/cos-backup.env; set +a
scripts/cos_backup.sh --dry-run
scripts/cos_backup.sh --apply
```

Default COS backup includes:

- online SQLite snapshot
- project metadata archive
- `server-data/napcat/config`

Full `server-data/ntqq` sync is disabled by default because it can include private QQ cache/media and grows quickly. Enable only for deliberate cold backups:

```bash
COS_INCLUDE_NAPCAT_NTQQ=1 scripts/cos_backup.sh --apply
```

## Year-Long Policy

- Keep live DB local for correctness and speed.
- Keep local DB backups for about 90 days.
- Upload DB snapshots and metadata to COS daily after credentials are configured.
- Keep local COS upload snapshots for 14 days after upload.
- Clean old VSCode server versions manually when `.vscode-server` exceeds a few GB.
- Do not delete `server-data/ntqq` blindly; it contains QQ runtime/login/cache data.
