#!/usr/bin/env python3
"""Daily raw chat archive and historical memory builder."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TZ = "Asia/Shanghai"
RAW_SCHEMA_VERSION = 1
MEMORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    base_url: str
    api_key_env: str


def load_config() -> dict[str, Any]:
    with (PROJECT_DIR / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_DIR / ".env")
    env_path = os.environ.get("COS_ENV_PATH", "/etc/qq-social-agent/cos-backup.env")
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_archive_date(raw: str, tz: ZoneInfo) -> datetime:
    now = datetime.now(tz)
    if raw == "today":
        day = now.date()
    elif raw == "yesterday":
        day = (now - timedelta(days=1)).date()
    else:
        day = datetime.strptime(raw, "%Y-%m-%d").date()
    return datetime(day.year, day.month, day.day, tzinfo=tz)


def allowed_groups(config: dict[str, Any], cli_groups: str | None) -> list[int]:
    if cli_groups:
        return [int(x.strip()) for x in cli_groups.split(",") if x.strip()]
    env_groups = os.environ.get("HISTORY_ARCHIVE_GROUPS", "").strip()
    if env_groups:
        return [int(x.strip()) for x in env_groups.split(",") if x.strip()]
    return [int(x) for x in (config.get("access_control", {}).get("allowed_groups") or [])]


def db_path(config: dict[str, Any]) -> Path:
    raw = config.get("bot", {}).get("data_path", "data/bot.sqlite3")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_DIR / path


def connect_db(config: dict[str, Any]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path(config)))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_messages(conn: sqlite3.Connection, groups: list[int], start_ts: float, end_ts: float) -> list[sqlite3.Row]:
    if not groups:
        return []
    placeholders = ",".join("?" for _ in groups)
    query = f"""
        select id, group_id, user_id, nickname, text, is_bot, created_at,
               source_message_id, source_kind, correlation_id
        from messages
        where group_id in ({placeholders})
          and created_at >= ?
          and created_at < ?
        order by created_at asc, id asc
    """
    return list(conn.execute(query, [*groups, start_ts, end_ts]))


def ensure_dirs() -> None:
    for path in ["data/history/raw", "data/history/memory", "data/history/manifests", "logs"]:
        (PROJECT_DIR / path).mkdir(parents=True, exist_ok=True)


def iso_from_ts(ts: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(tz).isoformat()


def row_to_json(row: sqlite3.Row, archive_date: str, tz: ZoneInfo) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "archive_date": archive_date,
        "message_row_id": int(row["id"]),
        "source_message_id": row["source_message_id"],
        "group_id": int(row["group_id"]),
        "user_id": int(row["user_id"]),
        "display_name": row["nickname"],
        "is_bot": bool(row["is_bot"]),
        "created_at": float(row["created_at"]),
        "created_at_iso": iso_from_ts(row["created_at"], tz),
        "source_kind": row["source_kind"],
        "correlation_id": row["correlation_id"],
        "text": row["text"],
    }


def write_raw_archive(rows: list[sqlite3.Row], archive_date: str, tz: ZoneInfo) -> tuple[Path, str, int]:
    raw_path = PROJECT_DIR / f"data/history/raw/daily_raw_{archive_date}.jsonl.gz"
    sha = hashlib.sha256()
    count = 0
    with gzip.open(raw_path, "wt", encoding="utf-8") as fh:
        for row in rows:
            line = json.dumps(row_to_json(row, archive_date, tz), ensure_ascii=False, separators=(",", ":"))
            encoded = (line + "\n").encode("utf-8")
            sha.update(encoded)
            fh.write(line + "\n")
            count += 1
    return raw_path, sha.hexdigest(), count


def compact_text(text: str, limit: int = 220) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 18] + f"...[省略{len(normalized) - limit + 18}字]"


def build_memory_input(rows: list[sqlite3.Row], tz: ZoneInfo, max_messages: int, max_chars: int) -> str:
    if len(rows) > max_messages:
        head_count = max(40, max_messages // 5)
        tail_count = max(120, max_messages // 2)
        mid_count = max_messages - head_count - tail_count
        remaining = rows[head_count:-tail_count]
        step = max(1, len(remaining) // max(1, mid_count))
        selected = rows[:head_count] + remaining[::step][:mid_count] + rows[-tail_count:]
    else:
        selected = rows
    lines: list[str] = []
    total = 0
    for row in selected:
        dt = datetime.fromtimestamp(float(row["created_at"]), tz=timezone.utc).astimezone(tz)
        tail = str(row["user_id"])[-5:]
        speaker = "张风雪" if row["is_bot"] else (row["nickname"] or "群友")
        role = "bot" if row["is_bot"] else "member"
        line = f"[{row['id']}] [{dt:%H:%M}] [{role}] {speaker}[#{tail}]: {compact_text(row['text'])}"
        total += len(line)
        if total > max_chars:
            lines.append(f"[truncated] 后续内容未输入模型，但已在 raw archive 完整保存。已输入约 {len(lines)} 条。")
            break
        lines.append(line)
    return "\n".join(lines)


def resolve_model_route(config: dict[str, Any], flow: str = "memory") -> ModelRoute | None:
    deepseek_cfg = config.get("deepseek", {}) or {}
    route = str(deepseek_cfg.get(f"{flow}_model") or deepseek_cfg.get("model") or "").strip()
    if not route:
        return None
    provider, model = route.split("/", 1) if "/" in route else ("deepseek", route)
    provider_cfg = (deepseek_cfg.get("providers", {}) or {}).get(provider, {}) or {}
    base_url = provider_cfg.get("base_url") or deepseek_cfg.get("base_url")
    api_key_env = provider_cfg.get("api_key_env") or "DEEPSEEK_API_KEY"
    if not base_url:
        return None
    return ModelRoute(provider=provider, model=model, base_url=base_url, api_key_env=api_key_env)


def safe_json_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_memory_groups(memory: dict[str, Any], rows: list[sqlite3.Row]) -> dict[str, Any]:
    known_groups = sorted({str(row["group_id"]) for row in rows})
    if not known_groups:
        return memory
    groups = memory.get("groups")
    if not isinstance(groups, dict):
        memory["groups"] = {}
        groups = memory["groups"]
    present_known = [gid for gid in known_groups if gid in groups]
    if not present_known and len(known_groups) == 1 and len(groups) == 1:
        old_key = next(iter(groups.keys()))
        groups[known_groups[0]] = groups.pop(old_key)
        memory["group_key_normalized_from"] = old_key
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        gid = str(row["group_id"])
        item = counts.setdefault(gid, {"message_count": 0, "bot_message_count": 0, "member_message_count": 0})
        item["message_count"] += 1
        if row["is_bot"]:
            item["bot_message_count"] += 1
        else:
            item["member_message_count"] += 1
    for gid, count_info in counts.items():
        item = groups.setdefault(gid, {})
        if isinstance(item, dict):
            for key, value in count_info.items():
                item.setdefault(key, value)
    return memory


def fallback_memory(archive_date: str, rows: list[sqlite3.Row], status: str, reason: str) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for row in rows:
        gid = str(row["group_id"])
        item = groups.setdefault(
            gid,
            {
                "message_count": 0,
                "bot_message_count": 0,
                "member_message_count": 0,
                "member_count": 0,
                "members": {},
                "active_members": {},
                "activity_by_hour": {},
                "notable_long_messages": [],
                "bot_reply_samples": [],
            },
        )
        item["message_count"] += 1
        hour = datetime.fromtimestamp(float(row["created_at"]), tz=timezone.utc).astimezone(ZoneInfo(DEFAULT_TZ)).strftime("%H")
        item["activity_by_hour"][hour] = item["activity_by_hour"].get(hour, 0) + 1
        text = row["text"] or ""
        speaker = row["nickname"] or "群友"
        if row["is_bot"]:
            item["bot_message_count"] += 1
            if len(item["bot_reply_samples"]) < 20 and text.strip():
                item["bot_reply_samples"].append({
                    "message_id": int(row["id"]),
                    "source_message_id": row["source_message_id"],
                    "text": compact_text(text, 160),
                })
        else:
            uid = str(row["user_id"])
            item["member_message_count"] += 1
            item["members"][uid] = speaker
            item["active_members"][uid] = item["active_members"].get(uid, 0) + 1
            if len(text) >= 80 and len(item["notable_long_messages"]) < 30:
                item["notable_long_messages"].append({
                    "message_id": int(row["id"]),
                    "source_message_id": row["source_message_id"],
                    "user_id": uid,
                    "name": speaker,
                    "text": compact_text(text, 220),
                })
    for item in groups.values():
        item["member_count"] = len(item["members"])
        item["active_members"] = [
            {"user_id": uid, "name": item["members"].get(uid, ""), "message_count": count}
            for uid, count in sorted(item["active_members"].items(), key=lambda pair: pair[1], reverse=True)[:30]
        ]
        item["activity_by_hour"] = dict(sorted(item["activity_by_hour"].items()))
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "date": archive_date,
        "generation_status": status,
        "reason": reason,
        "model_route": None,
        "groups": groups,
        "topics": [],
        "member_updates": [],
        "relationship_updates": [],
        "jargon_updates": [],
        "good_style_examples": [],
        "bot_feedback": [],
        "open_threads": [],
        "uncertain": [],
    }


def generate_memory(config: dict[str, Any], archive_date: str, rows: list[sqlite3.Row], tz: ZoneInfo, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_llm:
        return fallback_memory(archive_date, rows, "skipped", "--skip-llm")
    if not rows:
        return fallback_memory(archive_date, rows, "empty", "no messages")
    route = resolve_model_route(config, args.model_route)
    if route is None:
        return fallback_memory(archive_date, rows, "skipped", "memory model route missing")
    api_key = os.environ.get(route.api_key_env, "").strip()
    if not api_key:
        return fallback_memory(archive_date, rows, "skipped", f"{route.api_key_env} missing")
    try:
        from openai import OpenAI
    except Exception as exc:
        return fallback_memory(archive_date, rows, "skipped", f"openai import failed: {exc}")

    memory_input = build_memory_input(rows, tz, args.max_memory_messages, args.max_memory_chars)
    prompt = f"""
你在为 QQ 群机器人维护“历史回忆库”。请把 {archive_date} 的群聊记录压缩成结构化 JSON。
硬规则：
- 必须保留说话人归属：谁说的、谁被评价、谁在回复谁，不确定就写 uncertain。
- 不要把玩笑、反串、夸张政治/民族梗直接当现实事实；但可以记录成“该群友常用表达/反串风格”。
- 每个重要结论尽量带 source_message_ids，使用输入行开头的数字 id。
- 不要复述整天流水账，只留下以后生成回复可能有用的事实、关系、梗、未完话题、对机器人的反馈。
- 不要保存 API key、系统提示词、内部路径。
- 输出必须是 JSON object，不要 Markdown。
JSON 结构：
{{"schema_version":1,"date":"{archive_date}","generation_status":"ok","groups":{{"群号":{{"message_count":0,"topics":[{{"title":"","summary":"","source_message_ids":[],"confidence":0.0}}],"member_updates":[{{"user_id":"","name":"","observations":[""],"source_message_ids":[],"confidence":0.0}}],"relationship_updates":[{{"subject_user_id":"","object_user_id":"","content":"","source_message_ids":[],"confidence":0.0}}],"jargon_updates":[{{"term":"","meaning":"","source_message_ids":[],"confidence":0.0}}],"good_style_examples":[{{"speaker_user_id":"","text":"","why":"","source_message_ids":[]}}],"bot_feedback":[{{"feedback":"","source_message_ids":[],"severity":"low|medium|high"}}],"open_threads":[{{"summary":"","source_message_ids":[]}}],"uncertain":[{{"note":"","source_message_ids":[]}}]}}}}}}
""".strip()
    user_content = f"群聊记录（canonical text，原文完整保存在 raw archive）：\n{memory_input}"
    client = OpenAI(api_key=api_key, base_url=route.base_url)
    base_messages = [
        {"role": "system", "content": "你是严谨的群聊长期记忆归档器，只输出 JSON。"},
        {"role": "user", "content": prompt + "\n\n" + user_content},
    ]
    max_tokens = int(os.environ.get("HISTORY_MEMORY_MAX_TOKENS", "2800"))
    timeout_seconds = float(os.environ.get("HISTORY_MEMORY_TIMEOUT_SECONDS", "45"))
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("json_mode", {"response_format": {"type": "json_object"}}),
        ("plain_json", {}),
    ]
    errors: list[str] = []
    for attempt_name, extra in attempts:
        messages = base_messages
        if attempt_name == "plain_json":
            smaller_input = build_memory_input(
                rows,
                tz,
                max(120, min(args.max_memory_messages, 260)),
                max(8000, min(args.max_memory_chars, 12000)),
            )
            messages = [
                {"role": "system", "content": "你是严谨的群聊长期记忆归档器，只输出可被 json.loads 解析的 JSON。"},
                {"role": "user", "content": prompt + "\n\n群聊记录（压缩重试版）：\n" + smaller_input},
            ]
        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "timeout": timeout_seconds,
        }
        if route.provider == "siliconflow" and route.model.casefold().startswith("qwen/"):
            kwargs["extra_body"] = {"enable_thinking": False}
        elif route.provider == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        kwargs.update(extra)
        try:
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            if not content.strip():
                errors.append(f"{attempt_name}: empty model output")
                continue
            parsed = safe_json_loads(content)
            parsed.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
            parsed.setdefault("date", archive_date)
            parsed.setdefault("generation_status", "ok")
            parsed.setdefault("generation_attempt", attempt_name)
            parsed.setdefault("model_route", args.model_route)
            return normalize_memory_groups(parsed, rows)
        except Exception as exc:
            errors.append(f"{attempt_name}: {type(exc).__name__}: {exc}")
    failed = fallback_memory(archive_date, rows, "failed", "llm memory failed; " + " | ".join(errors)[-1500:])
    failed["model_route"] = args.model_route
    return failed


def write_memory(memory: dict[str, Any], archive_date: str) -> Path:
    path = PROJECT_DIR / f"data/history/memory/daily_memory_{archive_date}.json"
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(archive_date: str, start: datetime, end: datetime, raw_path: Path, raw_line_sha: str, raw_count: int, memory_path: Path, rows: list[sqlite3.Row], groups: list[int]) -> Path:
    manifest = {
        "schema_version": 1,
        "archive_date": archive_date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timezone": str(start.tzinfo),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "groups": groups,
        "message_count": raw_count,
        "bot_message_count": sum(1 for row in rows if row["is_bot"]),
        "member_message_count": sum(1 for row in rows if not row["is_bot"]),
        "raw_archive": {
            "path": str(raw_path.relative_to(PROJECT_DIR)),
            "bytes": raw_path.stat().st_size,
            "sha256_lines": raw_line_sha,
            "sha256_file": sha256_file(raw_path),
        },
        "memory_archive": {
            "path": str(memory_path.relative_to(PROJECT_DIR)),
            "bytes": memory_path.stat().st_size,
            "sha256_file": sha256_file(memory_path),
        },
    }
    path = PROJECT_DIR / f"data/history/manifests/daily_history_{archive_date}_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def update_app_kv(config: dict[str, Any], archive_date: str, manifest_path: Path, upload_status: str) -> None:
    try:
        conn = connect_db(config)
        now = time.time()
        payload = {
            "date": archive_date,
            "manifest_path": str(manifest_path.relative_to(PROJECT_DIR)),
            "upload_status": upload_status,
            "updated_at": now,
        }
        conn.execute(
            "insert into app_kv(key, value, updated_at) values(?, ?, ?) "
            "on conflict(key) do update set value=excluded.value, updated_at=excluded.updated_at",
            ("history_archive:last_success", json.dumps(payload, ensure_ascii=False), now),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"warn: app_kv update failed: {exc}", file=sys.stderr)


def have_command(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def cos_sync(local_dir: Path, cos_dest: str, timeout_seconds: int) -> bool:
    if not local_dir.exists():
        return False
    cmd = [
        "timeout", str(timeout_seconds), "coscli", "sync", str(local_dir), cos_dest,
        "-r", "--init-skip", "--disable-log", "--err-retry-num", "2",
        "--routines", "2", "--fail-output=false", "--process-log=false",
    ]
    print("cos_sync:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(PROJECT_DIR), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_tail = completed.stdout[-4000:]
    print(output_tail)
    print(f"cos_sync_returncode: {completed.returncode}")
    return completed.returncode == 0


def upload_history(args: argparse.Namespace) -> str:
    if args.no_upload:
        return "skipped:no-upload"
    cos_dest = os.environ.get("COS_BACKUP_DEST", "").strip().rstrip("/")
    if not cos_dest:
        return "skipped:COS_BACKUP_DEST missing"
    if not have_command("coscli"):
        return "skipped:coscli missing"
    timeout_seconds = int(os.environ.get("HISTORY_ARCHIVE_COS_TIMEOUT_SECONDS", "300"))
    ok_raw = cos_sync(PROJECT_DIR / "data/history/raw", f"{cos_dest}/history/raw/", timeout_seconds)
    ok_memory = cos_sync(PROJECT_DIR / "data/history/memory", f"{cos_dest}/history/memory/", timeout_seconds)
    ok_manifest = cos_sync(PROJECT_DIR / "data/history/manifests", f"{cos_dest}/history/manifests/", timeout_seconds)
    return "ok" if (ok_raw and ok_memory and ok_manifest) else "failed"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one Beijing day of QQ group history.")
    parser.add_argument("--date", default="yesterday", help="today, yesterday, or YYYY-MM-DD in Asia/Shanghai")
    parser.add_argument("--timezone", default=os.environ.get("HISTORY_ARCHIVE_TIMEZONE", DEFAULT_TZ))
    parser.add_argument("--groups", default=None, help="comma separated group ids; default config access_control.allowed_groups")
    parser.add_argument("--no-upload", action="store_true", help="write local files only")
    parser.add_argument("--skip-llm", action="store_true", help="write raw archive and fallback memory only")
    parser.add_argument("--max-memory-messages", type=int, default=int(os.environ.get("HISTORY_MEMORY_MAX_MESSAGES", "260")))
    parser.add_argument("--max-memory-chars", type=int, default=int(os.environ.get("HISTORY_MEMORY_MAX_CHARS", "12000")))
    parser.add_argument("--model-route", default=os.environ.get("HISTORY_MEMORY_ROUTE", "memory"), help="LLM route name from config, default memory")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    load_env()
    config = load_config()
    tz = ZoneInfo(args.timezone)
    start = parse_archive_date(args.date, tz)
    end = start + timedelta(days=1)
    archive_date = start.date().isoformat()
    groups = allowed_groups(config, args.groups)
    ensure_dirs()
    conn = connect_db(config)
    rows = fetch_messages(conn, groups, start.timestamp(), end.timestamp())
    conn.close()
    raw_path, raw_line_sha, raw_count = write_raw_archive(rows, archive_date, tz)
    memory = generate_memory(config, archive_date, rows, tz, args)
    memory_path = write_memory(memory, archive_date)
    manifest_path = write_manifest(archive_date, start, end, raw_path, raw_line_sha, raw_count, memory_path, rows, groups)
    upload_status = upload_history(args)
    update_app_kv(config, archive_date, manifest_path, upload_status)
    summary = {
        "date": archive_date,
        "groups": groups,
        "messages": raw_count,
        "raw": str(raw_path.relative_to(PROJECT_DIR)),
        "memory": str(memory_path.relative_to(PROJECT_DIR)),
        "manifest": str(manifest_path.relative_to(PROJECT_DIR)),
        "memory_status": memory.get("generation_status"),
        "upload_status": upload_status,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
