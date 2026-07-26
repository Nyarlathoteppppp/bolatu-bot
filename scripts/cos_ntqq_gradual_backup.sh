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
  echo "COS_BACKUP_DEST is required." >&2
  exit 2
fi

if [[ "$mode" == "--apply" ]] && ! command -v coscli >/dev/null 2>&1; then
  echo "coscli not found. Install it first with scripts/install_coscli.sh." >&2
  exit 127
fi

if [[ "$mode" == "--apply" ]] && ! command -v timeout >/dev/null 2>&1; then
  echo "timeout command not found; refusing to run unbounded COS upload." >&2
  exit 127
fi

mkdir -p data/maintenance logs

python3 - "$mode" <<'PY'
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
root = Path(os.environ.get("COS_NTQQ_ROOT", "server-data/ntqq"))
state_path = Path(os.environ.get("COS_NTQQ_STATE", "data/maintenance/ntqq_gradual_state.json"))
cos_dest = os.environ["COS_BACKUP_DEST"].rstrip("/")
max_bytes = int(float(os.environ.get("COS_NTQQ_MAX_BYTES", str(160 * 1024 * 1024))))
max_files = int(os.environ.get("COS_NTQQ_MAX_FILES", "250"))
timeout_seconds = int(os.environ.get("COS_NTQQ_FILE_TIMEOUT_SECONDS", "300"))
rate_limit = os.environ.get("COS_NTQQ_RATE_LIMIT_MBPS", "2")
err_retry_num = os.environ.get("COSCLI_ERR_RETRY_NUM", "2")

if not root.exists():
    print(f"ntqq root missing: {root}", file=sys.stderr)
    raise SystemExit(2)

exclude_globs = [
    item.strip()
    for item in os.environ.get(
        "COS_NTQQ_EXCLUDE_GLOBS",
        "NapCat/temp/*:*/nt_temp/*:*/log/*:*/log-cache/*:Crashpad/*:crash_files/*:*/__MACOSX/*:*/._*:*/.DS_Store",
    ).split(":")
    if item.strip()
]

def included(path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return not any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_globs)

all_files = sorted(path for path in root.rglob("*") if path.is_file())
files = [path for path in all_files if included(path)]
state = {}
if state_path.exists():
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        state = {}

uploaded: dict[str, str] = state.get("uploaded", {}) if isinstance(state.get("uploaded"), dict) else {}
cursor = int(state.get("cursor", 0) or 0)
if files:
    cursor %= len(files)
else:
    cursor = 0

selected: list[tuple[Path, str, int, str]] = []
selected_bytes = 0
scanned = 0
index = cursor

while files and scanned < len(files) and len(selected) < max_files:
    path = files[index]
    try:
        st = path.stat()
    except FileNotFoundError:
        index = (index + 1) % len(files)
        scanned += 1
        continue
    rel = path.relative_to(root).as_posix()
    sig = f"{st.st_size}:{int(st.st_mtime)}"
    if uploaded.get(rel) != sig:
        if selected and selected_bytes + st.st_size > max_bytes:
            break
        selected.append((path, rel, st.st_size, sig))
        selected_bytes += st.st_size
    index = (index + 1) % len(files)
    scanned += 1

print("== COS ntqq gradual backup ==")
print(f"mode: {mode}")
print(f"root: {root}")
print(f"dest: {cos_dest}/server-data/ntqq")
print(f"files_total: {len(files)}")
print(f"files_excluded: {len(all_files) - len(files)}")
print(f"exclude_globs: {':'.join(exclude_globs)}")
print(f"cursor_before: {cursor}")
print(f"scanned: {scanned}")
print(f"selected_files: {len(selected)}")
print(f"selected_bytes: {selected_bytes}")
print(f"max_bytes: {max_bytes}")
print(f"max_files: {max_files}")
print(f"rate_limit_MBps: {rate_limit}")

for path, rel, size, _sig in selected[:40]:
    print(f"- {size}\t{rel}")
if len(selected) > 40:
    print(f"... {len(selected) - 40} more files")

if mode == "--dry-run":
    raise SystemExit(0)

ok = 0
failed = 0
uploaded_bytes = 0
for path, rel, size, sig in selected:
    dest = f"{cos_dest}/server-data/ntqq/{rel}"
    cmd = [
        "timeout", str(timeout_seconds),
        "coscli", "cp", str(path), dest,
        "--init-skip",
        "--disable-log",
        "--err-retry-num", str(err_retry_num),
        "--rate-limiting", str(rate_limit),
        "--fail-output=false",
        "--process-log=false",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        uploaded[rel] = sig
        ok += 1
        uploaded_bytes += size
        print(f"uploaded\t{size}\t{rel}")
    else:
        failed += 1
        print(f"failed\t{size}\t{rel}\treturncode={result.returncode}")
        print(result.stdout[-1000:])

state = {
    "cursor": index if files else 0,
    "uploaded": uploaded,
    "last_run_at": time.time(),
    "last_run_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "last_files_total": len(files),
    "last_scanned": scanned,
    "last_selected": len(selected),
    "last_uploaded_ok": ok,
    "last_uploaded_failed": failed,
    "last_uploaded_bytes": uploaded_bytes,
}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
print(f"cursor_after: {state['cursor']}")
print(f"uploaded_ok: {ok}")
print(f"uploaded_failed: {failed}")
print(f"uploaded_bytes: {uploaded_bytes}")

if failed:
    raise SystemExit(1)
PY
