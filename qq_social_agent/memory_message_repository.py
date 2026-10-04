"""Message history, inbound claims and reply feedback on the shared connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable

from .image_read_state import ImageReadStateStore
from .memory_models import ChatMessage, RawCorpusExample, BotSentMessage, RecalledReplyFeedback, ApprovedReplyFeedback
from .memory_corpus import _raw_corpus_tags, _is_low_value_raw_corpus_text
from .memory_text import _compact_text
from .memory_repository_utils import _message_from_row, _source_message_key, _text_relevance_score


class MessageRepository:
    def __init__(
        self,
        conn: sqlite3.Connection,
        images: ImageReadStateStore,
        *,
        upsert_member_profile: Callable[..., object],
        update_member_impression: Callable[..., object],
    ):
        self.conn = conn
        self.images = images
        self._upsert_member_profile = upsert_member_profile
        self._update_member_impression = update_member_impression

    def add_message(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
        *,
        is_bot: bool = False,
        created_at: float | None = None,
        source_message_id: int | str | None = None,
        source_kind: str = "live",
        correlation_id: str | None = None,
        session_id: str | None = None,
        message_segments_json: str | None = None,
        raw_message_json: str | None = None,
        sender_json: str | None = None,
    ) -> bool:
        created = created_at or time.time()
        source_key = _source_message_key(source_message_id)
        cursor = self.conn.execute(
            """
            insert or ignore into messages(
              group_id, user_id, nickname, text, is_bot, created_at,
              source_message_id, source_kind, correlation_id, session_id,
              message_segments_json, raw_message_json, sender_json
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                user_id,
                nickname,
                text,
                int(is_bot),
                created,
                source_key,
                source_kind.strip()[:32] or "live",
                correlation_id.strip()[:160] if correlation_id else None,
                session_id.strip()[:160] if session_id else None,
                message_segments_json if message_segments_json else None,
                raw_message_json if raw_message_json else None,
                sender_json if sender_json else None,
            ),
        )
        if cursor.rowcount <= 0:
            self.conn.commit()
            return False
        if not is_bot and message_segments_json:
            self.images.observe(int(cursor.lastrowid), message_segments_json)
        if not is_bot:
            self._upsert_member_profile(group_id, user_id, nickname, last_seen_at=created)
            self._update_member_impression(
                group_id,
                user_id,
                nickname,
                text,
                created_at=created,
            )
        self.conn.commit()
        return True

    def admin_recent_messages(self, *, group_id: int | None = None, limit: int = 80) -> list[sqlite3.Row]:
        bounded = max(1, min(300, int(limit)))
        where = ""
        params: list[object] = []
        if group_id is not None:
            where = "where group_id = ?"
            params.append(int(group_id))
        rows = self.conn.execute(
            f"""
            select id, group_id, user_id, nickname, text, is_bot, created_at,
                   source_message_id, source_kind, correlation_id, session_id,
                   message_segments_json, raw_message_json, sender_json
            from messages
            {where}
            order by id desc
            limit ?
            """,
            (*params, bounded),
        ).fetchall()
        return list(rows)

    def admin_message(self, message_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at,
                   source_message_id, source_kind, correlation_id, session_id,
                   message_segments_json, raw_message_json, sender_json
            from messages
            where id = ?
            """,
            (int(message_id),),
        ).fetchone()

    def update_message_context(self, message_id: int, text: str) -> None:
        """Enrich an existing message without moving its arrival time."""
        self.conn.execute("update messages set text = ? where id = ?", (text, int(message_id)))
        self.conn.commit()

    def fill_message_segments(self, message_id: int, segments: str) -> None:
        """Restore media data omitted by older history imports, in place."""
        cursor = self.conn.execute("""
            update messages set message_segments_json = ?
            where id = ? and (message_segments_json is null or message_segments_json = '')
        """, (segments, int(message_id)))
        if cursor.rowcount:
            self.images.observe(int(message_id), segments)
        self.conn.commit()

    def admin_message_by_source(self, group_id: int, source_message_id: int | str | None) -> sqlite3.Row | None:
        source_key = _source_message_key(source_message_id)
        if not source_key:
            return None
        return self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at,
                   source_message_id, source_kind, correlation_id, session_id,
                   message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and source_message_id = ?
            order by id desc
            limit 1
            """,
            (int(group_id), source_key),
        ).fetchone()

    def message_source_exists(self, group_id: int, source_message_id: int | str | None) -> bool:
        source_key = _source_message_key(source_message_id)
        if not source_key:
            return False
        row = self.conn.execute(
            """
            select 1 from messages
            where group_id = ? and source_message_id = ?
            limit 1
            """,
            (group_id, source_key),
        ).fetchone()
        return row is not None

    def claim_inbound_message(
        self,
        group_id: int,
        source_message_id: int | str | None,
        *,
        correlation_id: str | None = None,
        created_at: float | None = None,
    ) -> bool:
        source_key = _source_message_key(source_message_id)
        if not source_key:
            return True
        if self.message_source_exists(group_id, source_key):
            return False
        cursor = self.conn.execute(
            """
            insert or ignore into inbound_message_events(
              group_id, source_message_id, first_seen_at, correlation_id
            )
            values (?, ?, ?, ?)
            """,
            (
                group_id,
                source_key,
                created_at or time.time(),
                correlation_id.strip()[:160] if correlation_id else None,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def recent_messages(self, group_id: int, limit: int) -> list[ChatMessage]:
        safe_limit = max(0, int(limit))
        if safe_limit <= 0:
            return []
        fetch_limit = max(safe_limit * 4, safe_limit + 20)
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, fetch_limit),
        ).fetchall()
        rows = _dedupe_recent_message_rows(rows, safe_limit)
        return [_message_from_row(row) for row in reversed(rows)]

    def messages_since(self, group_id: int, *, since_at: float) -> list[ChatMessage]:
        """Read the short window excluded from RAG, without a message-count cutoff."""
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and created_at >= ?
            order by created_at desc, id desc
            """,
            (int(group_id), float(since_at)),
        ).fetchall()
        rows = _dedupe_recent_message_rows(rows, len(rows))
        return [_message_from_row(row) for row in reversed(rows)]

    def current_session_messages(self, group_id: int, *, gap_seconds: float) -> list[ChatMessage]:
        """Read back to the last private-chat gap instead of a fixed message count."""
        cursor = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages where group_id = ? order by created_at desc, id desc
            """,
            (int(group_id),),
        )
        rows: list[sqlite3.Row] = []
        newer_at: float | None = None
        for row in cursor:
            created_at = float(row["created_at"])
            if newer_at is not None and newer_at - created_at >= gap_seconds:
                break
            rows.append(row)
            newer_at = created_at
        rows = _dedupe_recent_message_rows(rows, len(rows))
        return [_message_from_row(row) for row in reversed(rows)]

    def message_by_source(self, group_id: int, source_message_id: int | str | None) -> ChatMessage | None:
        row = self.admin_message_by_source(group_id, source_message_id)
        return _message_from_row(row) if row is not None else None

    def message_by_id(self, group_id: int, message_id: int) -> ChatMessage | None:
        row = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages where group_id = ? and id = ?
            """,
            (int(group_id), int(message_id)),
        ).fetchone()
        return _message_from_row(row) if row is not None else None

    def messages_between(
        self,
        group_id: int,
        *,
        start_at: float,
        end_at: float,
        limit: int,
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and created_at >= ? and created_at < ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, start_at, end_at, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def messages_before(self, group_id: int, *, before_at: float, limit: int) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and created_at < ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, before_at, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def relevant_raw_corpus_examples(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 240,
        context_radius: int = 2,
        exclude_user_id: int | None = None,
        exclude_text: str = "",
        preferred_user_id: int | None = None,
        preferred_limit: int = 0,
        preferred_score_multiplier: float = 1.0,
        preferred_score_bonus: float = 0.0,
        per_user_limit: int = 1,
    ) -> list[RawCorpusExample]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and is_bot = 0 and length(trim(text)) >= 2
            order by id desc
            limit ?
            """,
            (group_id, candidate_limit),
        ).fetchall()
        scored: list[tuple[float, float, int, ChatMessage, tuple[str, ...]]] = []
        excluded_text_key = _compact_text(exclude_text)
        for row in rows:
            message = _message_from_row(row)
            if exclude_user_id is not None and message.user_id == exclude_user_id:
                if excluded_text_key and _compact_text(message.text) == excluded_text_key:
                    continue
            if _is_low_value_raw_corpus_text(message.text):
                continue
            tags = _raw_corpus_tags(message.text)
            haystack = f"{message.nickname} {message.text} {' '.join(tags)}"
            score = _text_relevance_score(query, haystack)
            if score <= 0:
                continue
            if preferred_user_id is not None and message.user_id == preferred_user_id:
                score = score * preferred_score_multiplier + preferred_score_bonus
            scored.append((score, message.created_at, message.id, message, tags))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        preferred_examples: list[RawCorpusExample] = []
        regular_examples: list[RawCorpusExample] = []
        seen_texts: set[str] = set()
        user_counts: dict[int, int] = {}
        for score, _, _, message, tags in scored:
            text_key = _compact_text(message.text)
            if text_key in seen_texts:
                continue
            allowed_for_user = preferred_limit if message.user_id == preferred_user_id else per_user_limit
            if allowed_for_user > 0 and user_counts.get(message.user_id, 0) >= allowed_for_user:
                continue
            seen_texts.add(text_key)
            user_counts[message.user_id] = user_counts.get(message.user_id, 0) + 1
            before, after = self._message_neighbors(
                group_id,
                message.id,
                radius=context_radius,
            )
            example = RawCorpusExample(
                message=message,
                before=tuple(before),
                after=tuple(after),
                tags=tags,
                score=score,
            )
            if (
                preferred_user_id is not None
                and message.user_id == preferred_user_id
                and preferred_limit > 0
            ):
                if len(preferred_examples) < preferred_limit:
                    preferred_examples.append(example)
                continue
            else:
                regular_examples.append(example)
            if len(regular_examples) >= limit and len(preferred_examples) >= preferred_limit:
                break
        return (preferred_examples + regular_examples)[:limit]

    def _message_neighbors(
        self,
        group_id: int,
        message_id: int,
        *,
        radius: int,
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        if radius <= 0:
            return [], []
        before_rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and id < ?
            order by id desc
            limit ?
            """,
            (group_id, message_id, radius),
        ).fetchall()
        after_rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and id > ?
            order by id asc
            limit ?
            """,
            (group_id, message_id, radius),
        ).fetchall()
        return (
            [_message_from_row(row) for row in reversed(before_rows)],
            [_message_from_row(row) for row in after_rows],
        )

    def recent_bot_replies(self, group_id: int, seconds: int) -> list[ChatMessage]:
        since = time.time() - seconds
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and is_bot = 1 and created_at >= ?
            order by created_at desc, id desc
            """,
            (group_id, since),
        ).fetchall()
        rows = _dedupe_recent_message_rows(rows, len(rows))
        return [_message_from_row(row) for row in rows]

    def messages_for_mid_summary(
        self,
        group_id: int,
        *,
        keep_recent: int,
        batch_size: int,
        include_bot: bool = True,
    ) -> list[ChatMessage]:
        cutoff = self.conn.execute(
            """
            select id from messages
            where group_id = ?
            order by id desc
            limit 1 offset ?
            """,
            (group_id, keep_recent),
        ).fetchone()
        if not cutoff:
            return []

        state = self.conn.execute(
            "select last_message_id from memory_summary_state where group_id = ?",
            (group_id,),
        ).fetchone()
        last_message_id = int(state["last_message_id"]) if state else 0
        bot_filter = "" if include_bot else "and is_bot = 0"
        rows = self.conn.execute(
            f"""
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and id > ? and id <= ? {bot_filter}
            order by id asc
            limit ?
            """,
            (group_id, last_message_id, int(cutoff["id"]), batch_size),
        ).fetchall()
        return [_message_from_row(row) for row in rows]

    def add_bot_sent_message(
        self,
        *,
        group_id: int,
        message_id: int,
        bot_reply: str,
        trigger_user_id: int,
        trigger_nickname: str,
        trigger_text: str,
        action: str,
        created_at: float | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert or replace into bot_sent_messages(
              group_id, message_id, bot_reply, trigger_user_id, trigger_nickname,
              trigger_text, action, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                message_id,
                bot_reply,
                trigger_user_id,
                trigger_nickname,
                trigger_text,
                action,
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def bot_sent_message(self, group_id: int, message_id: int) -> BotSentMessage | None:
        row = self.conn.execute(
            """
            select group_id, message_id, bot_reply, trigger_user_id, trigger_nickname,
                   trigger_text, action, created_at
            from bot_sent_messages
            where group_id = ? and message_id = ?
            """,
            (group_id, message_id),
        ).fetchone()
        if not row:
            return None
        return _bot_sent_from_row(row)

    def add_recalled_reply_feedback(
        self,
        *,
        group_id: int,
        message_id: int,
        bot_reply: str,
        trigger_user_id: int,
        trigger_nickname: str,
        trigger_text: str,
        action: str,
        owner_reason: str,
        scene_summary: str,
        bad_reply_problem: str,
        avoid_rule: str,
        better_direction: str,
        tags: list[str],
        operator_id: int,
        reason_user_id: int,
        recalled_at: float,
        reason_at: float,
    ) -> None:
        now = time.time()
        self.conn.execute(
            """
            insert into recalled_reply_feedback(
              group_id, message_id, bot_reply, trigger_user_id, trigger_nickname,
              trigger_text, action, owner_reason, scene_summary, bad_reply_problem,
              avoid_rule, better_direction, tags_json, operator_id, reason_user_id,
              recalled_at, reason_at, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                message_id,
                bot_reply,
                trigger_user_id,
                trigger_nickname,
                trigger_text,
                action,
                owner_reason,
                scene_summary,
                bad_reply_problem,
                avoid_rule,
                better_direction,
                json.dumps(tags[:8], ensure_ascii=False),
                operator_id,
                reason_user_id,
                recalled_at,
                reason_at,
                now,
            ),
        )
        self.conn.commit()

    def recent_recalled_reply_feedback(
        self,
        group_id: int,
        limit: int,
    ) -> list[RecalledReplyFeedback]:
        rows = self.conn.execute(
            """
            select group_id, message_id, bot_reply, trigger_user_id, trigger_nickname,
                   trigger_text, action, owner_reason, scene_summary, bad_reply_problem,
                   avoid_rule, better_direction, tags_json, operator_id, reason_user_id,
                   recalled_at, reason_at
            from recalled_reply_feedback
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_recalled_feedback_from_row(row) for row in reversed(rows)]

    def add_approved_reply_feedback(
        self,
        *,
        group_id: int,
        candidate_text: str,
        trigger_user_id: int,
        trigger_nickname: str,
        trigger_text: str,
        action: str,
        style: str,
        tags: list[str] | None = None,
        operator_id: int,
        created_at: float | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert into approved_reply_feedback(
              group_id, candidate_text, trigger_user_id, trigger_nickname,
              trigger_text, action, style, tags_json, operator_id, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                candidate_text.strip(),
                trigger_user_id,
                trigger_nickname,
                trigger_text,
                action,
                style.strip(),
                json.dumps((tags or [])[:8], ensure_ascii=False),
                operator_id,
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def recent_approved_reply_feedback(
        self,
        group_id: int,
        limit: int,
    ) -> list[ApprovedReplyFeedback]:
        rows = self.conn.execute(
            """
            select group_id, candidate_text, trigger_user_id, trigger_nickname,
                   trigger_text, action, style, tags_json, operator_id, created_at
            from approved_reply_feedback
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_approved_feedback_from_row(row) for row in reversed(rows)]



def _dedupe_recent_message_rows(rows: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    selected: list[sqlite3.Row] = []
    seen_bot_rows: list[tuple[tuple[int, int, str], float]] = []
    for row in rows:
        if bool(row["is_bot"]):
            text_key = _compact_text(str(row["text"]))
            if text_key:
                key = (int(row["group_id"]), int(row["user_id"]), text_key)
                created_at = float(row["created_at"])
                if any(
                    seen_key == key and abs(created_at - seen_at) <= 3.0
                    for seen_key, seen_at in seen_bot_rows
                ):
                    continue
                seen_bot_rows.append((key, created_at))
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _bot_sent_from_row(row: sqlite3.Row) -> BotSentMessage:
    return BotSentMessage(
        group_id=int(row["group_id"]),
        message_id=int(row["message_id"]),
        bot_reply=str(row["bot_reply"]),
        trigger_user_id=int(row["trigger_user_id"]),
        trigger_nickname=str(row["trigger_nickname"]),
        trigger_text=str(row["trigger_text"]),
        action=str(row["action"]),
        created_at=float(row["created_at"]),
    )


def _recalled_feedback_from_row(row: sqlite3.Row) -> RecalledReplyFeedback:
    try:
        raw_tags = json.loads(str(row["tags_json"]))
    except json.JSONDecodeError:
        raw_tags = []
    tags = ()
    if isinstance(raw_tags, list):
        tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())
    return RecalledReplyFeedback(
        group_id=int(row["group_id"]),
        message_id=int(row["message_id"]),
        bot_reply=str(row["bot_reply"]),
        trigger_user_id=int(row["trigger_user_id"]),
        trigger_nickname=str(row["trigger_nickname"]),
        trigger_text=str(row["trigger_text"]),
        action=str(row["action"]),
        owner_reason=str(row["owner_reason"]),
        scene_summary=str(row["scene_summary"]),
        bad_reply_problem=str(row["bad_reply_problem"]),
        avoid_rule=str(row["avoid_rule"]),
        better_direction=str(row["better_direction"]),
        tags=tags,
        operator_id=int(row["operator_id"]),
        reason_user_id=int(row["reason_user_id"]),
        recalled_at=float(row["recalled_at"]),
        reason_at=float(row["reason_at"]),
    )


def _approved_feedback_from_row(row: sqlite3.Row) -> ApprovedReplyFeedback:
    try:
        raw_tags = json.loads(str(row["tags_json"]))
    except (KeyError, json.JSONDecodeError):
        raw_tags = []
    tags = ()
    if isinstance(raw_tags, list):
        tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())
    return ApprovedReplyFeedback(
        group_id=int(row["group_id"]),
        candidate_text=str(row["candidate_text"]),
        trigger_user_id=int(row["trigger_user_id"]),
        trigger_nickname=str(row["trigger_nickname"]),
        trigger_text=str(row["trigger_text"]),
        action=str(row["action"]),
        style=str(row["style"]),
        tags=tags,
        operator_id=int(row["operator_id"]),
        created_at=float(row["created_at"]),
    )
