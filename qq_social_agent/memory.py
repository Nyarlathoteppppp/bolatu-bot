from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
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
class RawCorpusExample:
    message: ChatMessage
    before: tuple[ChatMessage, ...]
    after: tuple[ChatMessage, ...]
    tags: tuple[str, ...]
    score: int


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
class MemberImpression:
    group_id: int
    user_id: int
    display_name: str
    aliases: tuple[str, ...]
    message_count: int
    top_tags: tuple[tuple[str, int], ...]
    top_keywords: tuple[tuple[str, int], ...]
    recent_texts: tuple[str, ...]
    ai_summary: str
    ai_interests: tuple[str, ...]
    ai_speaking_style: str
    ai_representative_texts: tuple[str, ...]
    ai_summary_at: float
    last_seen_at: float
    updated_at: float


@dataclass(frozen=True)
class MemberProfileSummary:
    group_id: int
    user_id: int
    profile_summary: str
    interests: tuple[str, ...]
    speaking_style: str
    representative_texts: tuple[str, ...]
    start_at: float
    end_at: float
    message_count: int
    created_at: float


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
        self._configure_connection()
        self._init_schema()

    def _configure_connection(self) -> None:
        self.conn.execute("pragma busy_timeout = 5000")
        self.conn.execute("pragma journal_mode = WAL")
        self.conn.execute("pragma synchronous = NORMAL")
        self.conn.execute("pragma temp_store = MEMORY")

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

            create index if not exists idx_messages_group_id
              on messages(group_id, id);

            create index if not exists idx_messages_group_bot_id
              on messages(group_id, is_bot, id);

            create index if not exists idx_messages_group_bot_text_id
              on messages(group_id, is_bot, id)
              where length(trim(text)) >= 2;

            create index if not exists idx_messages_group_user_time
              on messages(group_id, user_id, created_at);

            create index if not exists idx_messages_group_user_id
              on messages(group_id, user_id, id);

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

            create table if not exists app_kv (
              key text primary key,
              value text not null,
              updated_at real not null
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

            create index if not exists idx_member_profiles_group_seen
              on member_profiles(group_id, last_seen_at);

            create table if not exists member_impressions (
              group_id integer not null,
              user_id integer not null,
              message_count integer not null default 0,
              tag_counts_json text not null,
              keyword_counts_json text not null,
              recent_texts_json text not null,
              updated_at real not null,
              primary key(group_id, user_id)
            );

            create index if not exists idx_member_impressions_group_updated
              on member_impressions(group_id, updated_at);

            create table if not exists member_profile_summaries (
              id integer primary key autoincrement,
              group_id integer not null,
              user_id integer not null,
              profile_summary text not null,
              interests_json text not null,
              speaking_style text not null,
              representative_texts_json text not null,
              start_at real not null,
              end_at real not null,
              message_count integer not null,
              created_at real not null
            );

            create index if not exists idx_member_profile_summaries_group_user_time
              on member_profile_summaries(group_id, user_id, created_at);

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
              created_at real not null,
              source_key text
            );

            create index if not exists idx_llm_usage_events_time
              on llm_usage_events(created_at);

            """
        )
        self._ensure_llm_usage_source_key()
        self._backfill_member_profiles()
        self._backfill_member_impressions()
        self.conn.execute("pragma optimize")
        self.conn.commit()

    def _ensure_llm_usage_source_key(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(llm_usage_events)").fetchall()
        }
        if "source_key" not in columns:
            self.conn.execute("alter table llm_usage_events add column source_key text")
        self.conn.execute(
            """
            create unique index if not exists idx_llm_usage_events_source_key
              on llm_usage_events(source_key)
              where source_key is not null
            """
        )

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

    def _backfill_member_impressions(self) -> None:
        existing = self.conn.execute("select 1 from member_impressions limit 1").fetchone()
        if existing:
            return
        rows = self.conn.execute(
            """
            select group_id, user_id, nickname, text, created_at
            from messages
            where is_bot = 0
            order by created_at asc, id asc
            """
        ).fetchall()
        for row in rows:
            self._update_member_impression(
                int(row["group_id"]),
                int(row["user_id"]),
                str(row["nickname"]),
                str(row["text"]),
                created_at=float(row["created_at"]),
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
            self._update_member_impression(
                group_id,
                user_id,
                nickname,
                text,
                created_at=created,
            )
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
            select id, group_id, user_id, nickname, text, is_bot, created_at
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
            select id, group_id, user_id, nickname, text, is_bot, created_at
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
    ) -> list[RawCorpusExample]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and is_bot = 0 and length(trim(text)) >= 2
            order by id desc
            limit ?
            """,
            (group_id, candidate_limit),
        ).fetchall()
        scored: list[tuple[int, float, int, ChatMessage, tuple[str, ...]]] = []
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
            scored.append((score, message.created_at, message.id, message, tags))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        examples: list[RawCorpusExample] = []
        seen_texts: set[str] = set()
        for score, _, _, message, tags in scored:
            text_key = _compact_text(message.text)
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            before, after = self._message_neighbors(
                group_id,
                message.id,
                radius=context_radius,
            )
            examples.append(
                RawCorpusExample(
                    message=message,
                    before=tuple(before),
                    after=tuple(after),
                    tags=tags,
                    score=score,
                )
            )
            if len(examples) >= limit:
                break
        return examples

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
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and id < ?
            order by id desc
            limit ?
            """,
            (group_id, message_id, radius),
        ).fetchall()
        after_rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
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
            select group_id, start_at, end_at, summary, recall_cues_json, created_at
            from memory_summaries
            where group_id = ?
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
        return [_summary_from_row(row) for _, _, row in scored[:limit]]

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

    def relevant_style_rules(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 80,
    ) -> list[StyleRule]:
        rows = self.conn.execute(
            """
            select group_id, situation, style, source_text, created_at
            from style_rules
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, candidate_limit),
        ).fetchall()
        scored: list[tuple[int, float, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['situation']} {row['style']} {row['source_text']}"
            score = _text_relevance_score(query, haystack)
            if score > 0:
                scored.append((score, float(row["created_at"]), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
            )
            for _, _, row in scored[:limit]
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

    def member_impressions_for_context(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        limit: int,
    ) -> list[MemberImpression]:
        ordered_user_ids = _dedupe_ints(user_ids)[:limit]
        if not ordered_user_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_user_ids)
        rows = self.conn.execute(
            f"""
            select
              p.group_id,
              p.user_id,
              p.display_name,
              p.aliases_json,
              p.last_seen_at,
              coalesce(i.message_count, 0) as message_count,
              coalesce(i.tag_counts_json, '{{}}') as tag_counts_json,
              coalesce(i.keyword_counts_json, '{{}}') as keyword_counts_json,
              coalesce(i.recent_texts_json, '[]') as recent_texts_json,
              coalesce(i.updated_at, p.last_seen_at) as updated_at,
              coalesce(s.profile_summary, '') as ai_summary,
              coalesce(s.interests_json, '[]') as ai_interests_json,
              coalesce(s.speaking_style, '') as ai_speaking_style,
              coalesce(s.representative_texts_json, '[]') as ai_representative_texts_json,
              coalesce(s.created_at, 0) as ai_summary_at
            from member_profiles p
            left join member_impressions i
              on i.group_id = p.group_id and i.user_id = p.user_id
            left join member_profile_summaries s
              on s.id = (
                select latest.id
                from member_profile_summaries latest
                where latest.group_id = p.group_id and latest.user_id = p.user_id
                order by latest.created_at desc, latest.id desc
                limit 1
              )
            where p.group_id = ? and p.user_id in ({placeholders})
            """,
            (group_id, *ordered_user_ids),
        ).fetchall()
        by_user_id = {int(row["user_id"]): _member_impression_from_row(row) for row in rows}
        return [by_user_id[user_id] for user_id in ordered_user_ids if user_id in by_user_id]

    def recent_member_impressions(self, group_id: int, limit: int) -> list[MemberImpression]:
        rows = self.conn.execute(
            """
            select
              p.group_id,
              p.user_id,
              p.display_name,
              p.aliases_json,
              p.last_seen_at,
              coalesce(i.message_count, 0) as message_count,
              coalesce(i.tag_counts_json, '{}') as tag_counts_json,
              coalesce(i.keyword_counts_json, '{}') as keyword_counts_json,
              coalesce(i.recent_texts_json, '[]') as recent_texts_json,
              coalesce(i.updated_at, p.last_seen_at) as updated_at,
              coalesce(s.profile_summary, '') as ai_summary,
              coalesce(s.interests_json, '[]') as ai_interests_json,
              coalesce(s.speaking_style, '') as ai_speaking_style,
              coalesce(s.representative_texts_json, '[]') as ai_representative_texts_json,
              coalesce(s.created_at, 0) as ai_summary_at
            from member_profiles p
            left join member_impressions i
              on i.group_id = p.group_id and i.user_id = p.user_id
            left join member_profile_summaries s
              on s.id = (
                select latest.id
                from member_profile_summaries latest
                where latest.group_id = p.group_id and latest.user_id = p.user_id
                order by latest.created_at desc, latest.id desc
                limit 1
              )
            where p.group_id = ?
            order by p.last_seen_at desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_member_impression_from_row(row) for row in rows]

    def active_member_ids_since(
        self,
        group_id: int,
        *,
        since_at: float,
        limit: int,
        min_messages: int,
    ) -> list[int]:
        rows = self.conn.execute(
            """
            select user_id, count(*) as message_count, max(created_at) as last_seen_at
            from messages
            where group_id = ? and is_bot = 0 and created_at >= ? and length(trim(text)) >= 2
            group by user_id
            having count(*) >= ?
            order by message_count desc, last_seen_at desc
            limit ?
            """,
            (group_id, since_at, min_messages, limit),
        ).fetchall()
        return [int(row["user_id"]) for row in rows]

    def member_messages_between(
        self,
        group_id: int,
        user_id: int,
        *,
        start_at: float,
        end_at: float,
        limit: int,
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ?
              and user_id = ?
              and is_bot = 0
              and created_at >= ?
              and created_at < ?
              and length(trim(text)) >= 2
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, user_id, start_at, end_at, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def last_member_profile_summary_at(self, group_id: int, user_id: int) -> float:
        row = self.conn.execute(
            """
            select max(created_at) as ts
            from member_profile_summaries
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        return float(row["ts"] or 0.0) if row else 0.0

    def add_member_profile_summary(
        self,
        *,
        group_id: int,
        user_id: int,
        profile_summary: str,
        interests: list[str],
        speaking_style: str,
        representative_texts: list[str],
        start_at: float,
        end_at: float,
        message_count: int,
        keep_per_member: int = 14,
    ) -> None:
        summary = profile_summary.strip()
        if not summary:
            return
        now = time.time()
        self.conn.execute(
            """
            insert into member_profile_summaries(
              group_id, user_id, profile_summary, interests_json, speaking_style,
              representative_texts_json, start_at, end_at, message_count, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                user_id,
                summary[:420],
                json.dumps(_clean_text_list(interests, limit=8, item_limit=32), ensure_ascii=False),
                speaking_style.strip()[:260],
                json.dumps(_clean_text_list(representative_texts, limit=5, item_limit=140), ensure_ascii=False),
                start_at,
                end_at,
                max(0, int(message_count)),
                now,
            ),
        )
        self.conn.execute(
            """
            delete from member_profile_summaries
            where group_id = ? and user_id = ?
              and id not in (
                select id from member_profile_summaries
                where group_id = ? and user_id = ?
                order by created_at desc, id desc
                limit ?
              )
            """,
            (group_id, user_id, group_id, user_id, keep_per_member),
        )
        self.conn.commit()

    def recent_member_profile_summaries(
        self,
        group_id: int,
        user_id: int,
        limit: int,
    ) -> list[MemberProfileSummary]:
        rows = self.conn.execute(
            """
            select group_id, user_id, profile_summary, interests_json, speaking_style,
                   representative_texts_json, start_at, end_at, message_count, created_at
            from member_profile_summaries
            where group_id = ? and user_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, user_id, limit),
        ).fetchall()
        return [_member_profile_summary_from_row(row) for row in rows]

    def _update_member_impression(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
        *,
        created_at: float,
    ) -> None:
        clean_text = text.strip()
        row = self.conn.execute(
            """
            select message_count, tag_counts_json, keyword_counts_json, recent_texts_json
            from member_impressions
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        if row:
            message_count = int(row["message_count"]) + 1
            tag_counts = _counter_from_json(row["tag_counts_json"])
            keyword_counts = _counter_from_json(row["keyword_counts_json"])
            recent_texts = _recent_texts_from_json(row["recent_texts_json"])
        else:
            message_count = 1
            tag_counts = Counter()
            keyword_counts = Counter()
            recent_texts = []

        tags = _raw_corpus_tags(clean_text)
        tag_counts.update(tags)
        keyword_counts.update(_impression_keywords(clean_text))
        tag_counts = _cap_counter(tag_counts, 60)
        keyword_counts = _cap_counter(keyword_counts, 80)
        if clean_text and not _is_low_value_raw_corpus_text(clean_text):
            recent_texts.append(
                {
                    "text": clean_text[:140],
                    "tags": list(tags[:4]),
                    "at": created_at,
                }
            )
            recent_texts = recent_texts[-16:]

        self.conn.execute(
            """
            insert into member_impressions(
              group_id, user_id, message_count, tag_counts_json,
              keyword_counts_json, recent_texts_json, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              message_count = excluded.message_count,
              tag_counts_json = excluded.tag_counts_json,
              keyword_counts_json = excluded.keyword_counts_json,
              recent_texts_json = excluded.recent_texts_json,
              updated_at = excluded.updated_at
            """,
            (
                group_id,
                user_id,
                message_count,
                json.dumps(dict(tag_counts), ensure_ascii=False, sort_keys=True),
                json.dumps(dict(keyword_counts), ensure_ascii=False, sort_keys=True),
                json.dumps(recent_texts, ensure_ascii=False),
                created_at,
            ),
        )

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

    def app_kv_get(self, key: str) -> str | None:
        row = self.conn.execute("select value from app_kv where key = ?", (key,)).fetchone()
        if not row:
            return None
        return str(row["value"])

    def app_kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            insert into app_kv(key, value, updated_at)
            values (?, ?, ?)
            on conflict(key) do update set
              value = excluded.value,
              updated_at = excluded.updated_at
            """,
            (key, value, time.time()),
        )
        self.conn.commit()


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


def _text_relevance_score(query: str, haystack: str) -> int:
    query_terms = _relevance_terms(query)
    if not query_terms:
        return 0
    haystack_lower = haystack.casefold()
    score = 0
    for term in query_terms:
        term_lower = term.casefold()
        if term_lower not in haystack_lower:
            continue
        score += 3 if len(term_lower) >= 4 else 1
    return score


def _relevance_terms(text: str) -> set[str]:
    lowered = text.casefold()
    terms = {
        match.group(0)
        for match in re.finditer(r"[a-z0-9_]{2,}", lowered)
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 8:
            terms.add(chunk)
        for size in (2, 3, 4):
            if len(chunk) < size:
                continue
            for index in range(0, len(chunk) - size + 1):
                terms.add(chunk[index : index + size])
    stop_terms = {
        "这个",
        "那个",
        "什么",
        "怎么",
        "就是",
        "然后",
        "可以",
        "不是",
        "没有",
        "一下",
        "感觉",
        "时候",
    }
    return {term for term in terms if term not in stop_terms}


def _raw_corpus_tags(text: str) -> tuple[str, ...]:
    compact = _compact_text(text)
    tag_patterns = (
        ("玩梗", ("草", "哈哈", "笑死", "绷", "典", "麻了", "乐", "抽象", "梗", "开宰")),
        ("互损", ("傻逼", "弱智", "废物", "滚", "爹", "别学", "不如", "唐")),
        ("安慰", ("难受", "不开心", "顶不住", "撑不住", "压力", "破防", "烦死", "累")),
        ("反串", ("建议", "支持", "感觉不如", "这下", "赢", "什么成分")),
        ("政治", ("政治", "资本", "无产", "阶级", "粉红", "神友", "咱妈", "霓虹", "美国", "日本")),
        ("代码", ("代码", "bug", "报错", "炸了", "python", "java", "ai", "模型", "api")),
        ("倒霉", ("亏", "完蛋", "坏了", "炸了", "寄", "崩", "没人理")),
        ("恋爱", ("老婆", "喜欢", "暧昧", "女友", "男朋友", "宝宝")),
        ("行情", ("股票", "美股", "比特币", "btc", "eth", "亏钱", "涨", "跌")),
    )
    tags: list[str] = []
    for tag, patterns in tag_patterns:
        if any(pattern in compact for pattern in patterns):
            tags.append(tag)
    return tuple(tags)


def _impression_keywords(text: str) -> list[str]:
    compact = _compact_text(text)
    if not compact or _is_low_value_raw_corpus_text(compact):
        return []
    stop_terms = {
        "真的",
        "现在",
        "今天",
        "这个",
        "那个",
        "还是",
        "感觉",
        "不是",
        "没有",
        "怎么",
        "什么",
        "因为",
        "所以",
        "但是",
        "然后",
        "自己",
        "他们",
        "我们",
        "你们",
        "一样",
        "直接",
    }
    terms = [
        term
        for term in _relevance_terms(text)
        if 2 <= len(term) <= 12 and term not in stop_terms and not term.isdigit()
    ]
    terms.sort(key=lambda term: (len(term), term), reverse=True)
    return terms[:12]


def _is_low_value_raw_corpus_text(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return True
    if len(compact) <= 1:
        return True
    return compact in {
        "6",
        "66",
        "666",
        "草",
        "哈哈",
        "哈哈哈",
        "嗯",
        "哦",
        "好",
        "好的",
        "可以",
        "绷",
    }


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _counter_from_json(value: object) -> Counter[str]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = {}
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        text = str(key).strip()
        if not text:
            continue
        try:
            counter[text] = max(0, int(count))
        except (TypeError, ValueError):
            continue
    return counter


def _recent_texts_from_json(value: object) -> list[dict[str, object]]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = []
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        tags_raw = item.get("tags", [])
        tags = (
            [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            if isinstance(tags_raw, list)
            else []
        )
        try:
            created_at = float(item.get("at", 0.0) or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        result.append({"text": text[:140], "tags": tags[:4], "at": created_at})
    return result


def _cap_counter(counter: Counter[str], limit: int) -> Counter[str]:
    return Counter(dict(counter.most_common(limit)))


def _top_counter_items(counter: Counter[str], limit: int) -> tuple[tuple[str, int], ...]:
    return tuple((key, int(count)) for key, count in counter.most_common(limit) if count > 0)


def _json_text_list(value: object, *, limit: int) -> list[str]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = []
    if not isinstance(raw, list):
        return []
    return _clean_text_list([str(item) for item in raw], limit=limit, item_limit=140)


def _clean_text_list(items: list[str], *, limit: int, item_limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text[:item_limit])
        if len(cleaned) >= limit:
            break
    return cleaned


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


def _member_impression_from_row(row: sqlite3.Row) -> MemberImpression:
    profile = _profile_from_row(row)
    recent_texts = _recent_texts_from_json(row["recent_texts_json"])
    return MemberImpression(
        group_id=profile.group_id,
        user_id=profile.user_id,
        display_name=profile.display_name,
        aliases=profile.aliases,
        message_count=int(row["message_count"] or 0),
        top_tags=_top_counter_items(_counter_from_json(row["tag_counts_json"]), limit=6),
        top_keywords=_top_counter_items(_counter_from_json(row["keyword_counts_json"]), limit=8),
        recent_texts=tuple(
            str(item["text"])
            for item in recent_texts[-5:]
            if str(item.get("text", "")).strip()
        ),
        ai_summary=str(row["ai_summary"] or "").strip(),
        ai_interests=tuple(_json_text_list(row["ai_interests_json"], limit=8)),
        ai_speaking_style=str(row["ai_speaking_style"] or "").strip(),
        ai_representative_texts=tuple(_json_text_list(row["ai_representative_texts_json"], limit=5)),
        ai_summary_at=float(row["ai_summary_at"] or 0.0),
        last_seen_at=profile.last_seen_at,
        updated_at=float(row["updated_at"] or profile.last_seen_at),
    )


def _member_profile_summary_from_row(row: sqlite3.Row) -> MemberProfileSummary:
    return MemberProfileSummary(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        profile_summary=str(row["profile_summary"]),
        interests=tuple(_json_text_list(row["interests_json"], limit=8)),
        speaking_style=str(row["speaking_style"]),
        representative_texts=tuple(_json_text_list(row["representative_texts_json"], limit=5)),
        start_at=float(row["start_at"]),
        end_at=float(row["end_at"]),
        message_count=int(row["message_count"]),
        created_at=float(row["created_at"]),
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
