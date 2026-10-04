"""Metric events and model usage over the shared SQLite connection."""
from __future__ import annotations

import json
import sqlite3
import time

from .memory_models import (
    BotMetricEvent,
    BotMetricSummary,
    LLMUsageEvent,
    LLMUsageSummary,
)


class MetricsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def admin_recent_metric_events(
        self,
        *,
        event_types: tuple[str, ...] = (),
        group_id: int | None = None,
        limit: int = 80,
    ) -> list[sqlite3.Row]:
        bounded = max(1, min(300, int(limit)))
        clauses: list[str] = []
        params: list[object] = []
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"event_type in ({placeholders})")
            params.extend(event_types)
        if group_id is not None:
            clauses.append("group_id = ?")
            params.append(int(group_id))
        where = "where " + " and ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"""
            select id, event_type, group_id, user_id, stage, action, metadata_json, created_at
            from bot_metric_events
            {where}
            order by id desc
            limit ?
            """,
            (*params, bounded),
        ).fetchall()
        return list(rows)

    def add_metric_event(
        self,
        *,
        event_type: str,
        group_id: int | None = None,
        user_id: int | None = None,
        stage: str = "",
        action: str = "",
        metadata: dict[str, object] | None = None,
        created_at: float | None = None,
    ) -> None:
        clean_event_type = event_type.strip() or "unknown"
        clean_stage = stage.strip()
        clean_action = action.strip()
        payload = metadata or {}
        self.conn.execute(
            """
            insert into bot_metric_events(
              event_type, group_id, user_id, stage, action, metadata_json, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_event_type[:60],
                group_id,
                user_id,
                clean_stage[:80],
                clean_action[:80],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def metric_summary(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 80,
    ) -> list[BotMetricSummary]:
        where, params = _metric_where(start_at=start_at, end_at=end_at, group_id=group_id)
        rows = self.conn.execute(
            f"""
            select event_type, stage, action, count(*) as count
            from bot_metric_events
            {where}
            group by event_type, stage, action
            order by count desc, event_type asc
            limit ?
            """,
            (*params, limit),
        ).fetchall()
        return [
            BotMetricSummary(
                event_type=str(row["event_type"]),
                stage=str(row["stage"]),
                action=str(row["action"]),
                count=int(row["count"]),
            )
            for row in rows
        ]

    def metric_event_count(self, event_type: str) -> int:
        row = self.conn.execute(
            "select count(*) as count from bot_metric_events where event_type = ?",
            (event_type.strip(),),
        ).fetchone()
        return int(row["count"] or 0) if row else 0

    def recent_metric_events(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 12,
    ) -> list[BotMetricEvent]:
        where, params = _metric_where(start_at=start_at, end_at=end_at, group_id=group_id)
        rows = self.conn.execute(
            f"""
            select event_type, group_id, user_id, stage, action, metadata_json, created_at
            from bot_metric_events
            {where}
            order by created_at desc, id desc
            limit ?
            """,
            (*params, limit),
        ).fetchall()
        return [_metric_event_from_row(row) for row in rows]

    def prune_metric_events(
        self,
        *,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> dict[str, int]:
        deleted_by_age = 0
        deleted_by_rows = 0
        if max_age_seconds is not None and max_age_seconds > 0:
            cutoff = time.time() - int(max_age_seconds)
            cursor = self.conn.execute(
                "delete from bot_metric_events where created_at < ?",
                (cutoff,),
            )
            deleted_by_age = int(cursor.rowcount or 0)
        if max_rows is not None and max_rows > 0:
            cutoff_row = self.conn.execute(
                """
                select created_at, id
                from bot_metric_events
                order by created_at desc, id desc
                limit 1 offset ?
                """,
                (int(max_rows) - 1,),
            ).fetchone()
            if cutoff_row is not None:
                cursor = self.conn.execute(
                    """
                    delete from bot_metric_events
                    where created_at < ?
                       or (created_at = ? and id < ?)
                    """,
                    (
                        float(cutoff_row["created_at"]),
                        float(cutoff_row["created_at"]),
                        int(cutoff_row["id"]),
                    ),
                )
                deleted_by_rows = int(cursor.rowcount or 0)
        self.conn.commit()
        row = self.conn.execute("select count(*) as count from bot_metric_events").fetchone()
        return {
            "deleted_by_age": deleted_by_age,
            "deleted_by_rows": deleted_by_rows,
            "remaining": int(row["count"] or 0) if row else 0,
        }

    def add_llm_usage(
        self,
        *,
        task: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        created_at: float | None = None,
        source_key: str | None = None,
    ) -> bool:
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return False
        cursor = self.conn.execute(
            """
            insert or ignore into llm_usage_events(
              task, model, prompt_tokens, completion_tokens, total_tokens, created_at, source_key
            )
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.strip() or "unknown",
                model.strip() or "unknown",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                created_at or time.time(),
                source_key.strip()[:240] if source_key else None,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def llm_usage_summary(
        self,
        *,
        since_seconds: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
    ) -> list[LLMUsageSummary]:
        where, params = _usage_time_where(
            since_seconds=since_seconds,
            start_at=start_at,
            end_at=end_at,
        )
        rows = self.conn.execute(
            f"""
            select
              task,
              model,
              count(*) as call_count,
              sum(coalesce(prompt_tokens, 0)) as prompt_tokens,
              sum(coalesce(completion_tokens, 0)) as completion_tokens,
              sum(coalesce(total_tokens, coalesce(prompt_tokens, 0) + coalesce(completion_tokens, 0))) as total_tokens,
              min(created_at) as first_at,
              max(created_at) as last_at
            from llm_usage_events
            {where}
            group by task, model
            order by total_tokens desc, call_count desc
            """,
            params,
        ).fetchall()
        return [_llm_usage_summary_from_row(row) for row in rows]

    def recent_llm_usage_events(
        self,
        *,
        since_seconds: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
        limit: int = 8,
    ) -> list[LLMUsageEvent]:
        where, time_params = _usage_time_where(
            since_seconds=since_seconds,
            start_at=start_at,
            end_at=end_at,
        )
        params: tuple[object, ...] = (*time_params, limit)
        rows = self.conn.execute(
            f"""
            select task, model, prompt_tokens, completion_tokens, total_tokens, created_at
            from llm_usage_events
            {where}
            order by created_at desc, id desc
            limit ?
            """,
            params,
        ).fetchall()
        return [_llm_usage_event_from_row(row) for row in rows]


def _metric_event_from_row(row: sqlite3.Row) -> BotMetricEvent:
    try:
        raw_metadata = json.loads(str(row["metadata_json"]))
    except json.JSONDecodeError:
        raw_metadata = {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    group_id = row["group_id"]
    user_id = row["user_id"]
    return BotMetricEvent(
        event_type=str(row["event_type"]),
        group_id=int(group_id) if group_id is not None else None,
        user_id=int(user_id) if user_id is not None else None,
        stage=str(row["stage"]),
        action=str(row["action"]),
        metadata=metadata,
        created_at=float(row["created_at"]),
    )


def _metric_where(
    *,
    start_at: float | None,
    end_at: float | None,
    group_id: int | None,
) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    params: list[object] = []
    if start_at is not None:
        clauses.append("created_at >= ?")
        params.append(start_at)
    if end_at is not None:
        clauses.append("created_at < ?")
        params.append(end_at)
    if group_id is not None:
        clauses.append("group_id = ?")
        params.append(group_id)
    if not clauses:
        return "", ()
    return "where " + " and ".join(clauses), tuple(params)


def _usage_time_where(
    *,
    since_seconds: int | None,
    start_at: float | None,
    end_at: float | None,
) -> tuple[str, tuple[float, ...]]:
    clauses: list[str] = []
    params: list[float] = []
    if start_at is None and since_seconds is not None:
        start_at = time.time() - since_seconds
    if start_at is not None:
        clauses.append("created_at >= ?")
        params.append(start_at)
    if end_at is not None:
        clauses.append("created_at < ?")
        params.append(end_at)
    if not clauses:
        return "", ()
    return "where " + " and ".join(clauses), tuple(params)


def _llm_usage_summary_from_row(row: sqlite3.Row) -> LLMUsageSummary:
    return LLMUsageSummary(
        task=str(row["task"]),
        model=str(row["model"]),
        call_count=int(row["call_count"] or 0),
        prompt_tokens=int(row["prompt_tokens"] or 0),
        completion_tokens=int(row["completion_tokens"] or 0),
        total_tokens=int(row["total_tokens"] or 0),
        first_at=float(row["first_at"] or 0.0),
        last_at=float(row["last_at"] or 0.0),
    )


def _llm_usage_event_from_row(row: sqlite3.Row) -> LLMUsageEvent:
    prompt_tokens = int(row["prompt_tokens"] or 0)
    completion_tokens = int(row["completion_tokens"] or 0)
    total_tokens = row["total_tokens"]
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return LLMUsageEvent(
        task=str(row["task"]),
        model=str(row["model"]),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(total_tokens or 0),
        created_at=float(row["created_at"]),
    )
