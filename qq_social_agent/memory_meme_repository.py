"""MemeRepository over the caller-owned SQLite connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time

from .memory_models import MemeAsset
from .memory_text import _compact_text


class MemeRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_meme_asset(
        self,
        *,
        sha256: str,
        source_group_id: int,
        source_user_id: int,
        source_message_id: str,
        file_path: str,
        mime_type: str,
        byte_size: int,
        description: str,
        tags: tuple[str, ...] = (),
        enabled: bool = False,
    ) -> MemeAsset:
        now = time.time()
        cleaned_tags = tuple(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
        tags_json = json.dumps(cleaned_tags, ensure_ascii=False)
        self.conn.execute(
            """
            insert into meme_assets(
              sha256, source_group_id, source_user_id, source_message_id,
              file_path, mime_type, byte_size, description, tags_json, enabled,
              created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(sha256) do update set
              source_message_id = excluded.source_message_id,
              file_path = excluded.file_path,
              mime_type = excluded.mime_type,
              byte_size = excluded.byte_size,
              description = case when excluded.description != '' then excluded.description else meme_assets.description end,
              tags_json = case when excluded.tags_json != '[]' then excluded.tags_json else meme_assets.tags_json end,
              enabled = max(meme_assets.enabled, excluded.enabled),
              updated_at = excluded.updated_at
            """,
            (
                sha256, int(source_group_id), int(source_user_id), str(source_message_id),
                file_path, mime_type, max(0, int(byte_size)), description.strip()[:900], tags_json,
                int(bool(enabled)), now, now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("select * from meme_assets where sha256 = ?", (sha256,)).fetchone()
        assert row is not None
        return _meme_asset_from_row(row)

    def meme_assets_for_private(
        self,
        *,
        query: str = "",
        limit: int = 6,
        same_meme_cooldown_seconds: float = 6 * 60 * 60,
    ) -> list[MemeAsset]:
        now = time.time()
        cooldown = max(0.0, float(same_meme_cooldown_seconds))
        if cooldown:
            rows = self.conn.execute(
                """
                select * from meme_assets
                where enabled = 1
                  and (last_used_at is null or last_used_at <= ?)
                order by coalesce(last_used_at, 0) asc, use_count asc, id asc
                limit 80
                """,
                (now - cooldown,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select * from meme_assets
                where enabled = 1
                order by coalesce(last_used_at, 0) asc, use_count asc, id asc
                limit 80
                """
            ).fetchall()
        assets = [_meme_asset_from_row(row) for row in rows]
        if not assets:
            return []
        terms = _meme_query_terms(query)
        ranked = sorted(
            assets,
            key=lambda asset: (
                -_meme_relevance_score(asset, terms),
                asset.use_count,
                asset.last_used_at or 0.0,
                asset.id,
            ),
        )
        return ranked[: max(1, min(12, int(limit)))]

    def mark_meme_asset_used(self, meme_id: int) -> bool:
        now = time.time()
        cursor = self.conn.execute(
            """
            update meme_assets
            set last_used_at = ?, use_count = use_count + 1, updated_at = ?
            where id = ? and enabled = 1
            """,
            (now, now, int(meme_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def meme_asset(self, meme_id: int) -> MemeAsset | None:
        row = self.conn.execute("select * from meme_assets where id = ?", (int(meme_id),)).fetchone()
        return _meme_asset_from_row(row) if row is not None else None


def _meme_asset_from_row(row: sqlite3.Row) -> MemeAsset:
    try:
        parsed_tags = json.loads(str(row["tags_json"] or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed_tags = []
    tags = tuple(str(tag).strip() for tag in parsed_tags if str(tag).strip()) if isinstance(parsed_tags, list) else ()
    raw_last_used = row["last_used_at"]
    return MemeAsset(
        id=int(row["id"]),
        sha256=str(row["sha256"]),
        source_message_id=str(row["source_message_id"]),
        file_path=str(row["file_path"]),
        mime_type=str(row["mime_type"]),
        byte_size=int(row["byte_size"]),
        description=str(row["description"] or ""),
        tags=tags,
        enabled=bool(row["enabled"]),
        created_at=float(row["created_at"]),
        last_used_at=float(raw_last_used) if raw_last_used is not None else None,
        use_count=int(row["use_count"] or 0),
    )


def _meme_query_terms(query: str) -> tuple[str, ...]:
    clean = _compact_text(query).casefold()
    terms = [item for item in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", clean) if item]
    return tuple(dict.fromkeys(terms))


def _meme_relevance_score(asset: MemeAsset, terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    haystack = f"{asset.description} {' '.join(asset.tags)}".casefold()
    return float(sum(1 for term in terms if term in haystack))
