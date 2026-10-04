"""Memory summaries, recall and manual edits on the shared connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time

from .memory_models import ChatMessage, MemorySummary
from .memory_repository_utils import _text_relevance_score


def _summary_from_row(row: sqlite3.Row) -> MemorySummary:
    try:
        raw_cues = json.loads(str(row["recall_cues_json"]))
    except json.JSONDecodeError:
        raw_cues = []
    cues = tuple(str(cue).strip() for cue in raw_cues if str(cue).strip())
    columns = set(row.keys())
    updated_at = row["updated_at"] if "updated_at" in columns else row["created_at"]
    status = row["status"] if "status" in columns else "active"
    locked = row["locked"] if "locked" in columns else 0
    summary_id = row["id"] if "id" in columns else 0
    return MemorySummary(
        group_id=int(row["group_id"]),
        summary=str(row["summary"]),
        recall_cues=cues,
        start_at=float(row["start_at"]),
        end_at=float(row["end_at"]),
        created_at=float(row["created_at"]),
        id=int(summary_id or 0),
        status=str(status or "active"),
        locked=bool(locked),
        updated_at=float(updated_at or row["created_at"]),
    )


class SummaryRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add_memory_summary(
        self,
        group_id: int,
        messages: list[ChatMessage],
        *,
        summary: str,
        recall_cues: list[str],
    ) -> None:
        if not messages or not summary.strip():
            return
        start = messages[0]
        end = messages[-1]
        import json

        now = time.time()
        self.conn.execute(
            """
            insert into memory_summaries(
              group_id, start_message_id, end_message_id, start_at, end_at,
              summary, recall_cues_json, created_at, updated_at, status, locked
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
            """,
            (
                group_id,
                start.id,
                end.id,
                start.created_at,
                end.created_at,
                summary.strip(),
                json.dumps(recall_cues[:5], ensure_ascii=False),
                now,
                now,
            ),
        )
        self.conn.execute(
            """
            insert into memory_summary_state(group_id, last_message_id)
            values (?, ?)
            on conflict(group_id) do update set last_message_id = excluded.last_message_id
            """,
            (group_id, end.id),
        )
        self.conn.commit()

    def advance_memory_summary_cursor(self, group_id: int, last_message_id: int) -> None:
        """Move the mid-memory window forward without writing a summary."""

        cursor = max(0, int(last_message_id))
        if cursor <= 0:
            return
        self.conn.execute(
            """
            insert into memory_summary_state(group_id, last_message_id)
            values (?, ?)
            on conflict(group_id) do update set last_message_id = max(
              memory_summary_state.last_message_id,
              excluded.last_message_id
            )
            """,
            (int(group_id), cursor),
        )
        self.conn.commit()

    def recent_memory_summaries(self, group_id: int, limit: int) -> list[MemorySummary]:
        rows = self.conn.execute(
            """
            select id, group_id, start_at, end_at, summary, recall_cues_json, created_at,
                   coalesce(updated_at, created_at) as updated_at,
                   coalesce(status, 'active') as status,
                   coalesce(locked, 0) as locked
            from memory_summaries
            where group_id = ? and coalesce(status, 'active') = 'active'
            order by created_at desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_summary_from_row(row) for row in reversed(rows)]

    def relevant_memory_summaries(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 80,
    ) -> list[MemorySummary]:
        rows = self.conn.execute(
            """
            select id, group_id, start_at, end_at, summary, recall_cues_json, created_at,
                   coalesce(updated_at, created_at) as updated_at,
                   coalesce(status, 'active') as status,
                   coalesce(locked, 0) as locked
            from memory_summaries
            where group_id = ? and coalesce(status, 'active') = 'active'
            order by created_at desc
            limit ?
            """,
            (group_id, candidate_limit),
        ).fetchall()
        scored: list[tuple[int, float, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['summary']} {row['recall_cues_json']}"
            score = _text_relevance_score(query, haystack)
            if score > 0:
                scored.append((score, float(row["created_at"]), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if scored:
            return [_summary_from_row(row) for _, _, row in scored[:limit]]
        fallback_limit = min(max(0, int(limit)), 2)
        return [_summary_from_row(row) for row in rows[:fallback_limit]]

    def admin_recent_memory_summaries(
        self,
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        query: str = "",
    ) -> list[MemorySummary]:
        bounded = max(1, min(500, int(limit)))
        clauses: list[str] = []
        params: list[object] = []
        if group_id is not None:
            clauses.append("group_id = ?")
            params.append(int(group_id))
        clean_status = status.strip().casefold()
        if clean_status and clean_status != "all":
            clauses.append("coalesce(status, 'active') = ?")
            params.append(clean_status)
        clean_query = re.sub(r"\s+", " ", query).strip()
        if clean_query:
            like = f"%{clean_query[:160]}%"
            clauses.append("(summary like ? or recall_cues_json like ? or cast(id as text) = ?)")
            params.extend((like, like, clean_query))
        where = "where " + " and ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"""
            select id, group_id, start_at, end_at, summary, recall_cues_json, created_at,
                   coalesce(updated_at, created_at) as updated_at,
                   coalesce(status, 'active') as status,
                   coalesce(locked, 0) as locked
            from memory_summaries
            {where}
            order by coalesce(status, 'active') = 'active' desc, locked desc, updated_at desc, id desc
            limit ?
            """,
            (*params, bounded),
        ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def memory_summary(self, summary_id: int) -> MemorySummary | None:
        row = self.conn.execute(
            """
            select id, group_id, start_at, end_at, summary, recall_cues_json, created_at,
                   coalesce(updated_at, created_at) as updated_at,
                   coalesce(status, 'active') as status,
                   coalesce(locked, 0) as locked
            from memory_summaries
            where id = ?
            """,
            (int(summary_id),),
        ).fetchone()
        return _summary_from_row(row) if row is not None else None

    def admin_update_memory_summary(
        self,
        summary_id: int,
        *,
        summary: str,
        recall_cues: list[str],
        status: str = "active",
        locked: bool | None = None,
    ) -> bool:
        clean_summary = re.sub(r"\s+", " ", summary).strip()
        if not clean_summary:
            return False
        clean_status = status.strip().casefold() or "active"
        if clean_status not in {"active", "archived", "expired"}:
            clean_status = "active"
        cues = [re.sub(r"\s+", " ", str(cue)).strip() for cue in recall_cues]
        cues = [cue for cue in cues if cue][:12]
        row = self.conn.execute("select locked from memory_summaries where id = ?", (int(summary_id),)).fetchone()
        if row is None:
            return False
        locked_value = int(bool(locked)) if locked is not None else int(row["locked"] or 0)
        self.conn.execute(
            """
            update memory_summaries
            set summary = ?, recall_cues_json = ?, status = ?, locked = ?, updated_at = ?
            where id = ?
            """,
            (clean_summary, json.dumps(cues, ensure_ascii=False), clean_status, locked_value, time.time(), int(summary_id)),
        )
        self.conn.commit()
        return True

    def admin_set_memory_summary_state(self, summary_id: int, *, action: str) -> bool:
        clean_action = action.strip().casefold()
        row = self.conn.execute("select id, locked, status from memory_summaries where id = ?", (int(summary_id),)).fetchone()
        if row is None:
            return False
        status = str(row["status"] or "active")
        locked = int(row["locked"] or 0)
        if clean_action in {"lock", "freeze", "pin"}:
            locked = 1
            status = "active"
        elif clean_action in {"unlock", "unfreeze"}:
            locked = 0
        elif clean_action in {"archive", "archived"}:
            status = "archived"
        elif clean_action in {"expire", "expired", "delete"}:
            status = "expired"
        elif clean_action in {"active", "restore", "keep"}:
            status = "active"
        else:
            return False
        self.conn.execute(
            """
            update memory_summaries
            set status = ?, locked = ?, updated_at = ?
            where id = ?
            """,
            (status, locked, time.time(), int(summary_id)),
        )
        self.conn.commit()
        return True

    def admin_add_memory_summary(
        self,
        *,
        group_id: int,
        summary: str,
        recall_cues: list[str],
        locked: bool = True,
    ) -> int:
        clean_summary = re.sub(r"\s+", " ", summary).strip()
        if not clean_summary:
            return 0
        row = self.conn.execute(
            """
            select id, created_at
            from messages
            where group_id = ?
            order by id desc
            limit 1
            """,
            (int(group_id),),
        ).fetchone()
        message_id = int(row["id"] or 0) if row is not None else 0
        observed_at = float(row["created_at"] or time.time()) if row is not None else time.time()
        cues = [re.sub(r"\s+", " ", str(cue)).strip() for cue in recall_cues]
        cues = [cue for cue in cues if cue][:12]
        now = time.time()
        cursor = self.conn.execute(
            """
            insert into memory_summaries(
              group_id, start_message_id, end_message_id, start_at, end_at,
              summary, recall_cues_json, created_at, updated_at, status, locked
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                int(group_id),
                message_id,
                message_id,
                observed_at,
                observed_at,
                clean_summary,
                json.dumps(cues, ensure_ascii=False),
                now,
                now,
                int(bool(locked)),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)
