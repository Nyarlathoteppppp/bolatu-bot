#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("data/bot.sqlite3")
DEFAULT_REPORT_DIR = Path("reports")
COUNT_TABLES = (
    "messages",
    "inbound_message_events",
    "bot_sent_messages",
    "bot_metric_events",
    "llm_usage_events",
    "rag_documents",
    "rag_embeddings",
    "rag_retrieval_events",
    "memory_summaries",
    "member_profile_summaries",
    "memory_atoms",
    "style_rules",
    "approval_feedback",
)


def _human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type in ('table','view') and name=?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row else 0


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except FileNotFoundError:
                pass
    return total


def _top_level_sizes(path: Path, limit: int = 12) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    items: list[tuple[str, int]] = []
    for child in path.iterdir():
        try:
            size = _dir_size(child) if child.is_dir() else child.stat().st_size
            items.append((str(child), size))
        except FileNotFoundError:
            continue
    return sorted(items, key=lambda item: item[1], reverse=True)[:limit]


def _iso(ts: float | int | None) -> str:
    if ts is None:
        return "-"
    return dt.datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in COUNT_TABLES:
        if _table_exists(conn, table):
            counts[table] = _count(conn, f"select count(*) from {table}")
    return counts


def _recency_counts(conn: sqlite3.Connection, table: str, now: float) -> dict[str, int]:
    cols = _columns(conn, table)
    if "created_at" not in cols:
        return {}
    return {
        "1d": _count(conn, f"select count(*) from {table} where created_at >= ?", (now - 86400,)),
        "7d": _count(conn, f"select count(*) from {table} where created_at >= ?", (now - 7 * 86400,)),
        "30d": _count(conn, f"select count(*) from {table} where created_at >= ?", (now - 30 * 86400,)),
    }


def _message_stats(conn: sqlite3.Connection, now: float) -> dict[str, Any]:
    if not _table_exists(conn, "messages"):
        return {}
    total = _count(conn, "select count(*) from messages")
    first_ts = _scalar(conn, "select min(created_at) from messages")
    last_ts = _scalar(conn, "select max(created_at) from messages")
    observed_days = 0.0
    if first_ts and last_ts and last_ts > first_ts:
        observed_days = max((float(last_ts) - float(first_ts)) / 86400, 0.01)
    last_24h = _count(conn, "select count(*) from messages where created_at >= ?", (now - 86400,))
    last_7d = _count(conn, "select count(*) from messages where created_at >= ?", (now - 7 * 86400,))
    avg_7d = last_7d / 7
    return {
        "total": total,
        "range": f"{_iso(first_ts)} -> {_iso(last_ts)}",
        "observed_days": round(observed_days, 2),
        "last_24h": last_24h,
        "last_7d": last_7d,
        "avg_per_day_7d": round(avg_7d, 1),
        "year_estimate_by_7d": int(avg_7d * 365),
        "long_over_100": _count(conn, "select count(*) from messages where length(text) > 100"),
        "long_over_300": _count(conn, "select count(*) from messages where length(text) > 300"),
        "media_like": _count(
            conn,
            """
            select count(*) from messages
            where text like '%[图片%' or text like '%[表情%' or text like '%[语音%' or text like '%[视频%'
            """,
        ),
        "unknown_reference_like": _count(conn, "select count(*) from messages where text like '%原消息内容未知%'"),
    }


def _pipeline_24h(conn: sqlite3.Connection, now: float) -> list[tuple[str, int]]:
    if not _table_exists(conn, "bot_metric_events"):
        return []
    rows = conn.execute(
        """
        select coalesce(stage, event_type) as name, count(*) as count
        from bot_metric_events
        where created_at >= ?
        group by name
        order by count desc
        limit 25
        """,
        (now - 86400,),
    ).fetchall()
    return [(str(row[0]), int(row[1])) for row in rows]


def _rag_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "rag_documents"):
        return {}
    by_type = {
        f"{row['doc_type']}:{row['status']}": int(row["count"])
        for row in conn.execute(
            """
            select doc_type, status, count(*) as count
            from rag_documents
            group by doc_type, status
            order by doc_type, status
            """
        )
    }
    embedding_bytes = 0
    if _table_exists(conn, "rag_embeddings"):
        embedding_bytes = int(
            _scalar(conn, "select coalesce(sum(length(vector_blob)), 0) from rag_embeddings") or 0
        )
    max_len = _scalar(conn, "select max(length(content)) from rag_documents") or 0
    avg_len = _scalar(conn, "select avg(length(content)) from rag_documents where status='active'") or 0
    return {
        "by_type": by_type,
        "embedding_blob_total": _human_bytes(embedding_bytes),
        "active_avg_content_len": round(float(avg_len), 1),
        "max_content_len": int(max_len),
    }


def render_report(conn: sqlite3.Connection, db_path: Path, report_path: Path | None) -> str:
    now = time.time()
    lines: list[str] = [
        "# Dirty Work Diagnostics",
        f"Generated: {_iso(now)}",
        f"DB: {db_path.resolve()}",
        "",
        "## Disk",
        f"- db: {_human_bytes(db_path.stat().st_size if db_path.exists() else 0)}",
        f"- wal: {_human_bytes(Path(str(db_path) + '-wal').stat().st_size if Path(str(db_path) + '-wal').exists() else 0)}",
        f"- data/: {_human_bytes(_dir_size(db_path.parent))}",
    ]
    backup_files = list(db_path.parent.glob("*.bak")) + list(db_path.parent.glob("*.bak.gz"))
    backup_bytes = sum(path.stat().st_size for path in backup_files if path.exists())
    lines.append(f"- backups: {len(backup_files)} / {_human_bytes(backup_bytes)}")
    lines.extend(["", "### Top Server Data"])
    for name, size in _top_level_sizes(Path("server-data"), 12):
        lines.append(f"- {name}: {_human_bytes(size)}")

    lines.extend(["", "## SQLite Tables"])
    for table, count in sorted(_table_counts(conn).items(), key=lambda item: item[1], reverse=True):
        recency = _recency_counts(conn, table, now)
        suffix = f" / recency {recency}" if recency else ""
        lines.append(f"- {table}: {count}{suffix}")

    lines.extend(["", "## Messages"])
    for key, value in _message_stats(conn, now).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Pipeline 24h"])
    pipeline = _pipeline_24h(conn, now)
    if pipeline:
        for name, count in pipeline:
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## RAG"])
    rag = _rag_stats(conn)
    if rag:
        for key, value in rag.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## One-Year Risk",
            "- memory: current bot/NapCat memory footprint is stable; watch for container restarts, not gradual Python growth.",
            "- messages: current growth is safe for SQLite for one year.",
            "- docker: build cache is the highest disk risk; keep restart-time and cron pruning enabled.",
            "- napcat: Pic/Emoji cache can grow into multi-GB over a year; clean temp/log automatically and media manually or with NAPCAT_MEDIA_CLEAN=1.",
            "- backups: keep compressed backups; do not accumulate raw .bak files forever.",
        ]
    )
    text = "\n".join(lines) + "\n"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a live DB/disk diagnostic report.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report_path = args.report
    if report_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = DEFAULT_REPORT_DIR / f"dirty_work_diagnostics_{stamp}.md"
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        text = render_report(conn, args.db, report_path)
    finally:
        conn.close()
    print(text)
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
