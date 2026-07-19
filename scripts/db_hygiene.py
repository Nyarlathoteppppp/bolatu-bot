#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("data/bot.sqlite3")
UNREADABLE_CONVERSATION_WHERE = """
status='active' and doc_type='conversation' and (
  content like '%[图片OCR:%'
  or content like '%原消息内容未知%'
  or content like '%消息ID：%'
  or content like '%[长消息%'
  or content like '%[转发消息]%'
  or (content like '%[表情包%' and length(content)<260)
)
"""


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0] or 0)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type in ('table','view') and name=?",
        (name,),
    ).fetchone()
    return row is not None


def _file_sizes(db_path: Path) -> dict[str, int]:
    sizes = {"db_bytes": db_path.stat().st_size if db_path.exists() else 0}
    for suffix in ("-wal", "-shm"):
        path = Path(str(db_path) + suffix)
        sizes[f"{suffix}_bytes"] = path.stat().st_size if path.exists() else 0
    return sizes


def _rag_counts(conn: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(conn, "rag_documents"):
        return {}
    return {
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


def _rag_garbage_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not _table_exists(conn, "rag_documents"):
        return counts
    counts["unreadable_active_conversation_docs"] = _count(
        conn, f"select count(*) from rag_documents where {UNREADABLE_CONVERSATION_WHERE}"
    )
    if _table_exists(conn, "rag_embeddings"):
        counts["nonactive_embeddings"] = _count(
            conn,
            """
            select count(*)
            from rag_embeddings e
            join rag_documents d on d.id=e.document_id
            where d.status != 'active'
            """,
        )
        counts["orphan_embeddings"] = _count(
            conn,
            """
            select count(*)
            from rag_embeddings e
            left join rag_documents d on d.id=e.document_id
            where d.id is null
            """,
        )
    if _table_exists(conn, "rag_documents_fts"):
        counts["nonactive_fts_rows"] = _count(
            conn,
            """
            select count(*)
            from rag_documents_fts f
            join rag_documents d on d.id=f.rowid
            where d.status != 'active'
            """,
        )
        counts["orphan_fts_rows"] = _count(
            conn,
            """
            select count(*)
            from rag_documents_fts f
            left join rag_documents d on d.id=f.rowid
            where d.id is null
            """,
        )
        counts["active_docs_without_fts"] = _count(
            conn,
            """
            select count(*)
            from rag_documents d
            left join rag_documents_fts f on f.rowid=d.id
            where d.status='active' and f.rowid is null
            """,
        )
    return counts


def _retrieval_counts(conn: sqlite3.Connection, since: float) -> list[dict[str, Any]]:
    if not _table_exists(conn, "rag_retrieval_events"):
        return []
    return [
        dict(row)
        for row in conn.execute(
            """
            select route, count(*) as calls,
                   sum(case when error != '' then 1 else 0 end) as errors,
                   round(avg(lexical_count), 2) as avg_lexical,
                   round(avg(semantic_count), 2) as avg_semantic,
                   round(avg(injected_count), 2) as avg_injected,
                   round(avg(elapsed_ms), 1) as avg_ms
            from rag_retrieval_events
            where created_at >= ?
            group by route
            order by calls desc
            """,
            (since,),
        )
    ]


def run_hygiene(
    db_path: Path,
    *,
    dry_run: bool,
    rag_cleanup: bool,
    vacuum: bool,
    report_path: Path | None,
) -> dict[str, Any]:
    started_at = time.time()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma busy_timeout=30000")

    report: dict[str, Any] = {
        "started_at": started_at,
        "started_at_iso": dt.datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "db_path": str(db_path),
        "dry_run": dry_run,
        "rag_cleanup": rag_cleanup,
        "vacuum_requested": vacuum,
        "sizes_before": _file_sizes(db_path),
        "integrity_before": conn.execute("pragma integrity_check").fetchone()[0],
        "quick_before": conn.execute("pragma quick_check").fetchone()[0],
        "rag_counts_before": _rag_counts(conn),
        "rag_garbage_before": _rag_garbage_counts(conn),
        "retrieval_counts_24h": _retrieval_counts(conn, started_at - 24 * 3600),
    }

    if not dry_run:
        now = time.time()
        conn.execute("begin immediate")
        if rag_cleanup and _table_exists(conn, "rag_documents"):
            cursor = conn.execute(
                f"""
                update rag_documents
                set status='inactive', valid_to=?, updated_at=?, embedding_status='skipped'
                where id in (select id from rag_documents where {UNREADABLE_CONVERSATION_WHERE})
                """,
                (now, now),
            )
            report["inactivated_unreadable_conversation_docs"] = cursor.rowcount
        if _table_exists(conn, "rag_embeddings") and _table_exists(conn, "rag_documents"):
            cursor = conn.execute(
                """
                delete from rag_embeddings
                where document_id in (select id from rag_documents where status != 'active')
                """
            )
            report["deleted_nonactive_embeddings"] = cursor.rowcount
            cursor = conn.execute(
                "delete from rag_embeddings where document_id not in (select id from rag_documents)"
            )
            report["deleted_orphan_embeddings"] = cursor.rowcount
        if _table_exists(conn, "rag_documents_fts") and _table_exists(conn, "rag_documents"):
            cursor = conn.execute(
                """
                delete from rag_documents_fts
                where rowid in (select id from rag_documents where status != 'active')
                """
            )
            report["deleted_nonactive_fts"] = cursor.rowcount
            cursor = conn.execute(
                "delete from rag_documents_fts where rowid not in (select id from rag_documents)"
            )
            report["deleted_orphan_fts"] = cursor.rowcount
        conn.commit()
        conn.execute("pragma optimize")
        report["wal_checkpoint"] = [tuple(row) for row in conn.execute("pragma wal_checkpoint(TRUNCATE)")]
        if vacuum:
            try:
                conn.execute("vacuum")
                report["vacuum"] = "ok"
            except Exception as exc:  # pragma: no cover - depends on live DB lock state
                report["vacuum"] = f"error: {exc}"
        report["wal_checkpoint_after_vacuum"] = [
            tuple(row) for row in conn.execute("pragma wal_checkpoint(TRUNCATE)")
        ]

    finished_at = time.time()
    report.update(
        {
            "finished_at": finished_at,
            "finished_at_iso": dt.datetime.fromtimestamp(finished_at).isoformat(timespec="seconds"),
            "elapsed_ms": round((finished_at - started_at) * 1000, 1),
            "integrity_after": conn.execute("pragma integrity_check").fetchone()[0],
            "quick_after": conn.execute("pragma quick_check").fetchone()[0],
            "rag_counts_after": _rag_counts(conn),
            "rag_garbage_after": _rag_garbage_counts(conn),
            "sizes_after": _file_sizes(db_path),
        }
    )
    if not dry_run and _table_exists(conn, "app_kv"):
        conn.execute(
            "insert or replace into app_kv(key,value,updated_at) values(?,?,?)",
            ("dirty_work_last_db_hygiene", json.dumps(report, ensure_ascii=False), time.time()),
        )
        conn.commit()
    conn.close()

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(report), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# DB Hygiene Report",
        f"Generated: {report.get('finished_at_iso')}",
        f"DB: {report.get('db_path')}",
        f"Dry run: {report.get('dry_run')}",
        "",
        "## Integrity",
        f"- before: {report.get('integrity_before')} / {report.get('quick_before')}",
        f"- after: {report.get('integrity_after')} / {report.get('quick_after')}",
        "",
        "## Sizes",
        f"- before: {report.get('sizes_before')}",
        f"- after: {report.get('sizes_after')}",
        "",
        "## RAG Garbage",
        f"- before: {report.get('rag_garbage_before')}",
        f"- after: {report.get('rag_garbage_after')}",
        "",
        "## Actions",
    ]
    for key in (
        "inactivated_unreadable_conversation_docs",
        "deleted_nonactive_embeddings",
        "deleted_orphan_embeddings",
        "deleted_nonactive_fts",
        "deleted_orphan_fts",
        "vacuum",
    ):
        if key in report:
            lines.append(f"- {key}: {report[key]}")
    lines.extend(["", "## Retrieval 24h"])
    retrievals = report.get("retrieval_counts_24h") or []
    if retrievals:
        for row in retrievals:
            lines.append(f"- {row}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean RAG garbage and run SQLite hygiene checks.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-rag-cleanup", action="store_true")
    parser.add_argument("--no-vacuum", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = run_hygiene(
        args.db,
        dry_run=args.dry_run,
        rag_cleanup=not args.no_rag_cleanup,
        vacuum=not args.no_vacuum,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
