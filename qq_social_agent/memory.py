from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChatMessage:
    group_id: int
    user_id: int
    nickname: str
    text: str
    is_bot: bool
    created_at: float
    id: int = 0


@dataclass(frozen=True)
class MemorySummary:
    group_id: int
    summary: str
    recall_cues: tuple[str, ...]
    start_at: float
    end_at: float
    created_at: float


@dataclass(frozen=True)
class StyleRule:
    group_id: int
    situation: str
    style: str
    source_text: str
    created_at: float


@dataclass(frozen=True)
class MemberProfile:
    group_id: int
    user_id: int
    display_name: str
    aliases: tuple[str, ...]
    last_seen_at: float


@dataclass(frozen=True)
class BotSentMessage:
    group_id: int
    message_id: int
    bot_reply: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    created_at: float


@dataclass(frozen=True)
class RecalledReplyFeedback:
    group_id: int
    message_id: int
    bot_reply: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    owner_reason: str
    scene_summary: str
    bad_reply_problem: str
    avoid_rule: str
    better_direction: str
    tags: tuple[str, ...]
    operator_id: int
    reason_user_id: int
    recalled_at: float
    reason_at: float


@dataclass(frozen=True)
class ApprovedReplyFeedback:
    group_id: int
    candidate_text: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    style: str
    operator_id: int
    created_at: float


@dataclass(frozen=True)
class CustomJargonEntry:
    group_id: int
    term: str
    explanation: str
    created_by: int
    created_at: float


@dataclass(frozen=True)
class LLMUsageSummary:
    task: str
    model: str
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_at: float
    last_at: float


@dataclass(frozen=True)
class LLMUsageEvent:
    task: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: float


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists messages (
              id integer primary key autoincrement,
              group_id integer not null,
              user_id integer not null,
              nickname text not null,
              text text not null,
              is_bot integer not null default 0,
              created_at real not null
            );

            create index if not exists idx_messages_group_time
              on messages(group_id, created_at);

            create table if not exists memory_summaries (
              id integer primary key autoincrement,
              group_id integer not null,
              start_message_id integer not null,
              end_message_id integer not null,
              start_at real not null,
              end_at real not null,
              summary text not null,
              recall_cues_json text not null,
              created_at real not null
            );

            create index if not exists idx_memory_summaries_group_time
              on memory_summaries(group_id, created_at);

            create table if not exists memory_summary_state (
              group_id integer primary key,
              last_message_id integer not null default 0
            );

            create table if not exists style_rules (
              id integer primary key autoincrement,
              group_id integer not null,
              situation text not null,
              style text not null,
              source_text text not null,
              created_at real not null
            );

            create index if not exists idx_style_rules_group_time
              on style_rules(group_id, created_at);

            create table if not exists group_state (
              group_id integer primary key,
              enabled integer not null default 1,
              persona text,
              muted_until real not null default 0
            );

            create table if not exists member_profiles (
              group_id integer not null,
              user_id integer not null,
              display_name text not null,
              aliases_json text not null,
              last_seen_at real not null,
              primary key(group_id, user_id)
            );

            create table if not exists bot_sent_messages (
              group_id integer not null,
              message_id integer not null,
              bot_reply text not null,
              trigger_user_id integer not null,
              trigger_nickname text not null,
              trigger_text text not null,
              action text not null,
              created_at real not null,
              primary key(group_id, message_id)
            );

            create table if not exists recalled_reply_feedback (
              id integer primary key autoincrement,
              group_id integer not null,
              message_id integer not null,
              bot_reply text not null,
              trigger_user_id integer not null,
              trigger_nickname text not null,
              trigger_text text not null,
              action text not null,
              owner_reason text not null,
              scene_summary text not null,
              bad_reply_problem text not null,
              avoid_rule text not null,
              better_direction text not null,
              tags_json text not null,
              operator_id integer not null,
              reason_user_id integer not null,
              recalled_at real not null,
              reason_at real not null,
              created_at real not null
            );

            create index if not exists idx_recalled_feedback_group_time
              on recalled_reply_feedback(group_id, created_at);

            create table if not exists approved_reply_feedback (
              id integer primary key autoincrement,
              group_id integer not null,
              candidate_text text not null,
              trigger_user_id integer not null,
              trigger_nickname text not null,
              trigger_text text not null,
              action text not null,
              style text not null,
              operator_id integer not null,
              created_at real not null
            );

            create index if not exists idx_approved_feedback_group_time
              on approved_reply_feedback(group_id, created_at);

            create table if not exists custom_jargon_entries (
              id integer primary key autoincrement,
              group_id integer not null,
              term text not null,
              explanation text not null,
              created_by integer not null,
              created_at real not null,
              unique(group_id, term)
            );

            create index if not exists idx_custom_jargon_group_term
              on custom_jargon_entries(group_id, term);

            create table if not exists llm_usage_events (
              id integer primary key autoincrement,
              task text not null,
              model text not null,
              prompt_tokens integer,
              completion_tokens integer,
              total_tokens integer,
              created_at real not null
            );

            create index if not exists idx_llm_usage_events_time
              on llm_usage_events(created_at);

            """
        )
        self._backfill_member_profiles()
        self.conn.commit()

    def _backfill_member_profiles(self) -> None:
        existing = self.conn.execute("select 1 from member_profiles limit 1").fetchone()
        if existing:
            return
        rows = self.conn.execute(
            """
            select group_id, user_id, nickname, created_at
            from messages
            where is_bot = 0
            order by created_at asc, id asc
            """
        ).fetchall()
        for row in rows:
            self._upsert_member_profile(
                int(row["group_id"]),
                int(row["user_id"]),
                str(row["nickname"]),
                last_seen_at=float(row["created_at"]),
            )

    def add_message(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
        *,
        is_bot: bool = False,
        created_at: float | None = None,
    ) -> None:
        created = created_at or time.time()
        self.conn.execute(
            """
            insert into messages(group_id, user_id, nickname, text, is_bot, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (group_id, user_id, nickname, text, int(is_bot), created),
        )
        if not is_bot:
            self._upsert_member_profile(group_id, user_id, nickname, last_seen_at=created)
        self.conn.commit()

    def _upsert_member_profile(
        self,
        group_id: int,
        user_id: int,
        display_name: str,
        *,
        last_seen_at: float,
    ) -> None:
        clean_name = display_name.strip() or str(user_id)
        row = self.conn.execute(
            """
            select aliases_json from member_profiles
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        aliases: list[str] = []
        if row:
            try:
                raw_aliases = json.loads(str(row["aliases_json"]))
            except json.JSONDecodeError:
                raw_aliases = []
            if isinstance(raw_aliases, list):
                aliases = [str(alias).strip() for alias in raw_aliases if str(alias).strip()]
        aliases = _dedupe_names([clean_name, *aliases])[:8]
        self.conn.execute(
            """
            insert into member_profiles(group_id, user_id, display_name, aliases_json, last_seen_at)
            values (?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              display_name = excluded.display_name,
              aliases_json = excluded.aliases_json,
              last_seen_at = excluded.last_seen_at
            """,
            (group_id, user_id, clean_name, json.dumps(aliases, ensure_ascii=False), last_seen_at),
        )

    def recent_messages(self, group_id: int, limit: int) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ?
            order by created_at desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [
            ChatMessage(
                group_id=int(row["group_id"]),
                user_id=int(row["user_id"]),
                nickname=str(row["nickname"]),
                text=str(row["text"]),
                is_bot=bool(row["is_bot"]),
                created_at=float(row["created_at"]),
                id=int(row["id"]),
            )
            for row in reversed(rows)
        ]

    def messages_before(self, group_id: int, *, before_at: float, limit: int) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and created_at < ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, before_at, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def recent_bot_replies(self, group_id: int, seconds: int) -> list[ChatMessage]:
        since = time.time() - seconds
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and is_bot = 1 and created_at >= ?
            order by created_at desc
            """,
            (group_id, since),
        ).fetchall()
        return [
            ChatMessage(
                group_id=int(row["group_id"]),
                user_id=int(row["user_id"]),
                nickname=str(row["nickname"]),
                text=str(row["text"]),
                is_bot=True,
                created_at=float(row["created_at"]),
                id=int(row["id"]),
            )
            for row in rows
        ]

    def messages_for_mid_summary(
        self,
        group_id: int,
        *,
        keep_recent: int,
        batch_size: int,
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
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and id > ? and id <= ?
            order by id asc
            limit ?
            """,
            (group_id, last_message_id, int(cutoff["id"]), batch_size),
        ).fetchall()
        return [_message_from_row(row) for row in rows]

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
              summary, recall_cues_json, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?)
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

    def recent_memory_summaries(self, group_id: int, limit: int) -> list[MemorySummary]:
        rows = self.conn.execute(
            """
            select group_id, start_at, end_at, summary, recall_cues_json, created_at
            from memory_summaries
            where group_id = ?
            order by created_at desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_summary_from_row(row) for row in reversed(rows)]

    def messages_for_style_learning(
        self,
        group_id: int,
        *,
        limit: int,
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and is_bot = 0 and length(trim(text)) >= 2
            order by id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def last_style_rule_at(self, group_id: int) -> float:
        row = self.conn.execute(
            "select max(created_at) as ts from style_rules where group_id = ?",
            (group_id,),
        ).fetchone()
        return float(row["ts"] or 0.0) if row else 0.0

    def add_style_rules(
        self,
        group_id: int,
        rules: list[tuple[str, str, str]],
        *,
        keep: int = 80,
    ) -> None:
        now = time.time()
        clean_rules = [
            (situation.strip(), style.strip(), source_text.strip())
            for situation, style, source_text in rules
            if situation.strip() and style.strip()
        ]
        if not clean_rules:
            return
        self.conn.executemany(
            """
            insert into style_rules(group_id, situation, style, source_text, created_at)
            values (?, ?, ?, ?, ?)
            """,
            [
                (group_id, situation[:60], style[:80], source_text[:200], now)
                for situation, style, source_text in clean_rules
            ],
        )
        self.conn.execute(
            """
            delete from style_rules
            where group_id = ?
              and id not in (
                select id from style_rules
                where group_id = ?
                order by created_at desc, id desc
                limit ?
              )
            """,
            (group_id, group_id, keep),
        )
        self.conn.commit()

    def recent_style_rules(self, group_id: int, limit: int) -> list[StyleRule]:
        rows = self.conn.execute(
            """
            select group_id, situation, style, source_text, created_at
            from style_rules
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    def member_profiles_for_context(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        limit: int,
    ) -> list[MemberProfile]:
        ordered_user_ids = _dedupe_ints(user_ids)[:limit]
        if not ordered_user_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_user_ids)
        rows = self.conn.execute(
            f"""
            select group_id, user_id, display_name, aliases_json, last_seen_at
            from member_profiles
            where group_id = ? and user_id in ({placeholders})
            """,
            (group_id, *ordered_user_ids),
        ).fetchall()
        by_user_id = {int(row["user_id"]): _profile_from_row(row) for row in rows}
        return [by_user_id[user_id] for user_id in ordered_user_ids if user_id in by_user_id]

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
        operator_id: int,
        created_at: float | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert into approved_reply_feedback(
              group_id, candidate_text, trigger_user_id, trigger_nickname,
              trigger_text, action, style, operator_id, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                candidate_text.strip(),
                trigger_user_id,
                trigger_nickname,
                trigger_text,
                action,
                style.strip(),
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
                   trigger_text, action, style, operator_id, created_at
            from approved_reply_feedback
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_approved_feedback_from_row(row) for row in reversed(rows)]

    def upsert_custom_jargon(
        self,
        *,
        group_id: int,
        term: str,
        explanation: str,
        created_by: int,
    ) -> None:
        clean_term = term.strip()
        clean_explanation = explanation.strip()
        if not clean_term or not clean_explanation:
            return
        now = time.time()
        self.conn.execute(
            """
            insert into custom_jargon_entries(group_id, term, explanation, created_by, created_at)
            values (?, ?, ?, ?, ?)
            on conflict(group_id, term) do update set
              explanation = excluded.explanation,
              created_by = excluded.created_by,
              created_at = excluded.created_at
            """,
            (
                group_id,
                clean_term[:40],
                clean_explanation[:160],
                created_by,
                now,
            ),
        )
        self.conn.commit()

    def delete_custom_jargon(self, group_id: int, term: str) -> bool:
        cursor = self.conn.execute(
            "delete from custom_jargon_entries where group_id = ? and term = ?",
            (group_id, term.strip()),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def custom_jargon_entries(self, group_id: int) -> list[CustomJargonEntry]:
        rows = self.conn.execute(
            """
            select group_id, term, explanation, created_by, created_at
            from custom_jargon_entries
            where group_id = ?
            order by created_at desc, id desc
            """,
            (group_id,),
        ).fetchall()
        return [_custom_jargon_from_row(row) for row in rows]

    def add_llm_usage(
        self,
        *,
        task: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        created_at: float | None = None,
    ) -> None:
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return
        self.conn.execute(
            """
            insert into llm_usage_events(
              task, model, prompt_tokens, completion_tokens, total_tokens, created_at
            )
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                task.strip() or "unknown",
                model.strip() or "unknown",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def llm_usage_summary(self, *, since_seconds: int | None = None) -> list[LLMUsageSummary]:
        where = ""
        params: tuple[float, ...] = ()
        if since_seconds is not None:
            where = "where created_at >= ?"
            params = (time.time() - since_seconds,)
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
        limit: int = 8,
    ) -> list[LLMUsageEvent]:
        where = ""
        params: tuple[object, ...] = (limit,)
        if since_seconds is not None:
            where = "where created_at >= ?"
            params = (time.time() - since_seconds, limit)
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

    def set_group_enabled(self, group_id: int, enabled: bool) -> None:
        self.conn.execute(
            """
            insert into group_state(group_id, enabled)
            values (?, ?)
            on conflict(group_id) do update set enabled = excluded.enabled
            """,
            (group_id, int(enabled)),
        )
        self.conn.commit()

    def set_group_persona(self, group_id: int, persona: str) -> None:
        self.conn.execute(
            """
            insert into group_state(group_id, persona)
            values (?, ?)
            on conflict(group_id) do update set persona = excluded.persona
            """,
            (group_id, persona),
        )
        self.conn.commit()

    def mute_until(self, group_id: int, until_timestamp: float) -> None:
        self.conn.execute(
            """
            insert into group_state(group_id, muted_until)
            values (?, ?)
            on conflict(group_id) do update set muted_until = excluded.muted_until
            """,
            (group_id, until_timestamp),
        )
        self.conn.commit()

    def reset_group_messages(self, group_id: int) -> None:
        self.conn.execute("delete from messages where group_id = ?", (group_id,))
        self.conn.commit()

    def group_state(self, group_id: int) -> dict[str, object]:
        row = self.conn.execute(
            "select enabled, persona, muted_until from group_state where group_id = ?",
            (group_id,),
        ).fetchone()
        if not row:
            return {"enabled": True, "persona": None, "muted_until": 0.0}
        return {
            "enabled": bool(row["enabled"]),
            "persona": row["persona"],
            "muted_until": float(row["muted_until"]),
        }


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        nickname=str(row["nickname"]),
        text=str(row["text"]),
        is_bot=bool(row["is_bot"]),
        created_at=float(row["created_at"]),
        id=int(row["id"]),
    )


def _summary_from_row(row: sqlite3.Row) -> MemorySummary:
    try:
        raw_cues = json.loads(str(row["recall_cues_json"]))
    except json.JSONDecodeError:
        raw_cues = []
    cues = tuple(str(cue).strip() for cue in raw_cues if str(cue).strip())
    return MemorySummary(
        group_id=int(row["group_id"]),
        summary=str(row["summary"]),
        recall_cues=cues,
        start_at=float(row["start_at"]),
        end_at=float(row["end_at"]),
        created_at=float(row["created_at"]),
    )


def _profile_from_row(row: sqlite3.Row) -> MemberProfile:
    try:
        raw_aliases = json.loads(str(row["aliases_json"]))
    except json.JSONDecodeError:
        raw_aliases = []
    aliases = ()
    if isinstance(raw_aliases, list):
        aliases = tuple(_dedupe_names(str(alias).strip() for alias in raw_aliases if str(alias).strip()))
    return MemberProfile(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        display_name=str(row["display_name"]),
        aliases=aliases,
        last_seen_at=float(row["last_seen_at"]),
    )


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
    return ApprovedReplyFeedback(
        group_id=int(row["group_id"]),
        candidate_text=str(row["candidate_text"]),
        trigger_user_id=int(row["trigger_user_id"]),
        trigger_nickname=str(row["trigger_nickname"]),
        trigger_text=str(row["trigger_text"]),
        action=str(row["action"]),
        style=str(row["style"]),
        operator_id=int(row["operator_id"]),
        created_at=float(row["created_at"]),
    )


def _custom_jargon_from_row(row: sqlite3.Row) -> CustomJargonEntry:
    return CustomJargonEntry(
        group_id=int(row["group_id"]),
        term=str(row["term"]),
        explanation=str(row["explanation"]),
        created_by=int(row["created_by"]),
        created_at=float(row["created_at"]),
    )


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


def _dedupe_names(names: object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = str(name).strip()
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _dedupe_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
