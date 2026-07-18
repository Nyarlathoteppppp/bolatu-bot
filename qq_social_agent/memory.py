from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MEMORY_ATOM_EVIDENCE_TYPES = frozenset({"message", "event", "manual"})
MEMORY_ATOM_STATUSES = frozenset({"active", "superseded", "disputed", "expired"})
_MEMORY_ATOM_SELECT_COLUMNS = """
    id, atom_type, group_id, subject_user_id, object_user_id, content,
    source, evidence_type, source_message_id, observed_at,
    valid_from, valid_to, confidence, importance, status,
    supersedes_id, expires_at, created_at, updated_at
"""


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
    score: float


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
    scope: str = "group"
    source_user_ids: tuple[int, ...] = ()
    source_message_ids: tuple[int, ...] = ()
    support_user_count: int = 1
    evidence_count: int = 1
    confidence: float = 0.6
    status: str = "active"
    valid_to: float | None = None


@dataclass(frozen=True)
class MemberProfile:
    group_id: int
    user_id: int
    display_name: str
    aliases: tuple[str, ...]
    last_seen_at: float


@dataclass(frozen=True)
class GroupInfo:
    group_id: int
    group_name: str
    member_count: int
    max_member_count: int
    last_synced_at: float


@dataclass(frozen=True)
class GroupMember:
    group_id: int
    user_id: int
    nickname: str
    card: str
    role: str
    title: str
    joined_at: float
    last_sent_at: float
    last_synced_at: float
    active: bool


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
    tags: tuple[str, ...]
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


@dataclass(frozen=True)
class BotMetricEvent:
    event_type: str
    group_id: int | None
    user_id: int | None
    stage: str
    action: str
    metadata: dict[str, object]
    created_at: float


@dataclass(frozen=True)
class BotMetricSummary:
    event_type: str
    stage: str
    action: str
    count: int


@dataclass(frozen=True)
class MemoryAtom:
    id: int
    atom_type: str
    group_id: int
    subject_user_id: int | None
    object_user_id: int | None
    content: str
    source: str
    confidence: float
    importance: float
    expires_at: float | None
    created_at: float
    updated_at: float
    evidence_type: str = "manual"
    source_message_id: str | None = None
    observed_at: float = 0.0
    valid_from: float | None = None
    valid_to: float | None = None
    status: str = "active"
    supersedes_id: int | None = None

    @property
    def evidence_source(self) -> str:
        return self.evidence_type


@dataclass(frozen=True)
class MemoryAtomAuditEvent:
    id: int
    atom_id: int
    action: str
    evidence_type: str
    source: str
    source_message_id: str | None
    actor_user_id: int | None
    detail: str
    observed_at: float
    created_at: float
    metadata: dict[str, object]


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
              created_at real not null,
              source_message_id text,
              source_kind text not null default 'live',
              correlation_id text
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

            create table if not exists inbound_message_events (
              group_id integer not null,
              source_message_id text not null,
              first_seen_at real not null,
              correlation_id text,
              primary key(group_id, source_message_id)
            );

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

            create table if not exists group_info (
              group_id integer primary key,
              group_name text not null,
              member_count integer not null default 0,
              max_member_count integer not null default 0,
              last_synced_at real not null
            );

            create table if not exists group_members (
              group_id integer not null,
              user_id integer not null,
              nickname text not null,
              card text not null default '',
              role text not null default '',
              title text not null default '',
              joined_at real not null default 0,
              last_sent_at real not null default 0,
              last_synced_at real not null,
              active integer not null default 1,
              primary key(group_id, user_id)
            );

            create index if not exists idx_group_members_group_active
              on group_members(group_id, active, user_id);

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
              tags_json text not null default '[]',
              operator_id integer not null,
              created_at real not null
            );

            create index if not exists idx_approved_feedback_group_time
              on approved_reply_feedback(group_id, created_at);

            create table if not exists bot_metric_events (
              id integer primary key autoincrement,
              event_type text not null,
              group_id integer,
              user_id integer,
              stage text not null,
              action text not null,
              metadata_json text not null,
              created_at real not null
            );

            create index if not exists idx_bot_metric_events_time
              on bot_metric_events(created_at);

            create index if not exists idx_bot_metric_events_group_time
              on bot_metric_events(group_id, created_at);

            create table if not exists memory_atoms (
              id integer primary key autoincrement,
              atom_type text not null,
              group_id integer not null,
              subject_user_id integer,
              object_user_id integer,
              content text not null,
              source text not null,
              evidence_type text not null default 'manual',
              source_message_id text,
              observed_at real,
              valid_from real,
              valid_to real,
              confidence real not null default 0.7,
              importance real not null default 0.5,
              status text not null default 'active',
              supersedes_id integer,
              expires_at real,
              created_at real not null,
              updated_at real not null
            );

            create index if not exists idx_memory_atoms_group_type_time
              on memory_atoms(group_id, atom_type, updated_at);

            create index if not exists idx_memory_atoms_group_subject
              on memory_atoms(group_id, subject_user_id, updated_at);

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
        self._ensure_message_source_columns()
        self._ensure_group_directory_tables()
        self._ensure_approved_feedback_tags()
        self._ensure_llm_usage_source_key()
        self._ensure_memory_atom_v2()
        self._ensure_style_rule_v2()
        self.expire_due_memory_atoms()
        self._backfill_member_profiles()
        self._backfill_member_impressions()
        self.conn.execute("pragma optimize")
        self.conn.commit()

    def _ensure_approved_feedback_tags(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(approved_reply_feedback)").fetchall()
        }
        if "tags_json" not in columns:
            self.conn.execute("alter table approved_reply_feedback add column tags_json text not null default '[]'")

    def _ensure_message_source_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(messages)").fetchall()
        }
        if "source_message_id" not in columns:
            self.conn.execute("alter table messages add column source_message_id text")
        if "source_kind" not in columns:
            self.conn.execute("alter table messages add column source_kind text not null default 'live'")
        if "correlation_id" not in columns:
            self.conn.execute("alter table messages add column correlation_id text")
        self.conn.execute(
            """
            create unique index if not exists idx_messages_group_source_message
              on messages(group_id, source_message_id)
              where source_message_id is not null and source_message_id != ''
            """
        )
        self.conn.execute(
            """
            create table if not exists inbound_message_events (
              group_id integer not null,
              source_message_id text not null,
              first_seen_at real not null,
              correlation_id text,
              primary key(group_id, source_message_id)
            )
            """
        )

    def _ensure_group_directory_tables(self) -> None:
        self.conn.executescript(
            """
            create table if not exists group_info (
              group_id integer primary key,
              group_name text not null,
              member_count integer not null default 0,
              max_member_count integer not null default 0,
              last_synced_at real not null
            );

            create table if not exists group_members (
              group_id integer not null,
              user_id integer not null,
              nickname text not null,
              card text not null default '',
              role text not null default '',
              title text not null default '',
              joined_at real not null default 0,
              last_sent_at real not null default 0,
              last_synced_at real not null,
              active integer not null default 1,
              primary key(group_id, user_id)
            );

            create index if not exists idx_group_members_group_active
              on group_members(group_id, active, user_id);
            """
        )

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

    def _ensure_style_rule_v2(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(style_rules)").fetchall()
        }
        additions = (
            ("scope", "text not null default 'legacy'"),
            ("source_user_ids_json", "text not null default '[]'"),
            ("source_message_ids_json", "text not null default '[]'"),
            ("support_user_count", "integer not null default 1"),
            ("evidence_count", "integer not null default 1"),
            ("confidence", "real not null default 0.6"),
            ("status", "text not null default 'active'"),
            ("valid_to", "real"),
        )
        for name, declaration in additions:
            if name not in columns:
                self.conn.execute(f"alter table style_rules add column {name} {declaration}")
        self.conn.execute(
            "create index if not exists idx_style_rules_scope_status on style_rules(group_id, scope, status, created_at)"
        )

    def _ensure_memory_atom_v2(self) -> None:
        savepoint = "memory_atom_v2_migration"
        self.conn.execute(f"savepoint {savepoint}")
        try:
            self._ensure_memory_atom_v2_schema()
            self.conn.execute(f"release savepoint {savepoint}")
        except Exception:
            self.conn.execute(f"rollback to savepoint {savepoint}")
            self.conn.execute(f"release savepoint {savepoint}")
            raise

    def _ensure_memory_atom_v2_schema(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(memory_atoms)").fetchall()
        }
        additions = (
            ("evidence_type", "text not null default 'manual'"),
            ("source_message_id", "text"),
            ("observed_at", "real"),
            ("valid_from", "real"),
            ("valid_to", "real"),
            ("status", "text not null default 'active'"),
            ("supersedes_id", "integer"),
        )
        evidence_type_added = "evidence_type" not in columns
        for name, declaration in additions:
            if name not in columns:
                self.conn.execute(f"alter table memory_atoms add column {name} {declaration}")

        if evidence_type_added:
            self.conn.execute(
                """
                update memory_atoms
                set evidence_type = case
                  when source_message_id is not null and source_message_id != '' then 'message'
                  when source like 'message:%' then 'message'
                  when source = 'manual'
                    or source like 'manual:%'
                    or source like 'manual_%'
                    or source like 'builtin%'
                    then 'manual'
                  else 'event'
                end
                """
            )
        else:
            self.conn.execute(
                """
                update memory_atoms
                set evidence_type = 'manual'
                where evidence_type not in ('message', 'event', 'manual')
                   or evidence_type is null
                   or evidence_type = ''
                """
            )
        self.conn.execute(
            """
            update memory_atoms
            set observed_at = coalesce(observed_at, created_at),
                valid_from = coalesce(valid_from, created_at),
                valid_to = coalesce(valid_to, expires_at),
                status = case
                  when status is null or status = '' then 'active'
                  when status not in ('active', 'superseded', 'disputed', 'expired') then 'active'
                  else status
                end
            where observed_at is null
               or valid_from is null
               or (valid_to is null and expires_at is not null)
               or status not in ('active', 'superseded', 'disputed', 'expired')
               or status is null
            """
        )
        statements = (
            """
            create index if not exists idx_memory_atoms_group_status_validity
              on memory_atoms(group_id, status, valid_from, valid_to, updated_at)
            """,
            """
            create index if not exists idx_memory_atoms_source_message
              on memory_atoms(group_id, source_message_id)
              where source_message_id is not null and source_message_id != ''
            """,
            """
            create index if not exists idx_memory_atoms_status_expiry
              on memory_atoms(status, valid_to, expires_at)
            """,
            "create index if not exists idx_memory_atoms_supersedes on memory_atoms(supersedes_id)",
            """
            create table if not exists memory_atom_audit_events (
              id integer primary key autoincrement,
              atom_id integer not null,
              action text not null,
              evidence_type text not null,
              source text not null,
              source_message_id text,
              actor_user_id integer,
              detail text not null default '',
              observed_at real not null,
              created_at real not null,
              metadata_json text not null default '{}'
            )
            """,
            """
            create index if not exists idx_memory_atom_audit_atom_time
              on memory_atom_audit_events(atom_id, created_at, id)
            """,
            """
            create index if not exists idx_memory_atom_audit_source_message
              on memory_atom_audit_events(source_message_id)
              where source_message_id is not null and source_message_id != ''
            """,
        )
        for statement in statements:
            self.conn.execute(statement)
        self.conn.execute(
            """
            insert into memory_atom_audit_events(
              atom_id, action, evidence_type, source, source_message_id,
              actor_user_id, detail, observed_at, created_at, metadata_json
            )
            select atom.id, 'migrated', atom.evidence_type, atom.source,
                   atom.source_message_id, null, 'legacy memory atom',
                   coalesce(atom.observed_at, atom.created_at), atom.created_at, '{}'
            from memory_atoms as atom
            where not exists (
              select 1 from memory_atom_audit_events as audit
              where audit.atom_id = atom.id
            )
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
        source_message_id: int | str | None = None,
        source_kind: str = "live",
        correlation_id: str | None = None,
    ) -> bool:
        created = created_at or time.time()
        source_key = _source_message_key(source_message_id)
        cursor = self.conn.execute(
            """
            insert or ignore into messages(
              group_id, user_id, nickname, text, is_bot, created_at,
              source_message_id, source_kind, correlation_id
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        if cursor.rowcount <= 0:
            self.conn.commit()
            return False
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

    def upsert_group_info(
        self,
        *,
        group_id: int,
        group_name: str,
        member_count: int,
        max_member_count: int,
        last_synced_at: float | None = None,
    ) -> None:
        synced = last_synced_at or time.time()
        self.conn.execute(
            """
            insert into group_info(group_id, group_name, member_count, max_member_count, last_synced_at)
            values (?, ?, ?, ?, ?)
            on conflict(group_id) do update set
              group_name = excluded.group_name,
              member_count = excluded.member_count,
              max_member_count = excluded.max_member_count,
              last_synced_at = excluded.last_synced_at
            """,
            (group_id, group_name.strip()[:120], max(0, member_count), max(0, max_member_count), synced),
        )
        self.conn.commit()

    def group_info(self, group_id: int) -> GroupInfo | None:
        row = self.conn.execute(
            """
            select group_id, group_name, member_count, max_member_count, last_synced_at
            from group_info
            where group_id = ?
            """,
            (group_id,),
        ).fetchone()
        return _group_info_from_row(row) if row else None

    def replace_group_members(
        self,
        group_id: int,
        members: list[dict[str, object]],
        *,
        synced_at: float | None = None,
    ) -> int:
        synced = synced_at or time.time()
        self.conn.execute(
            "update group_members set active = 0, last_synced_at = ? where group_id = ?",
            (synced, group_id),
        )
        clean_members: list[tuple[object, ...]] = []
        for member in members:
            user_id = int(member.get("user_id") or 0)
            if user_id <= 0:
                continue
            clean_members.append(
                (
                    group_id,
                    user_id,
                    str(member.get("nickname") or user_id).strip()[:120],
                    str(member.get("card") or "").strip()[:120],
                    str(member.get("role") or "").strip()[:32],
                    str(member.get("title") or "").strip()[:120],
                    float(member.get("joined_at") or 0.0),
                    float(member.get("last_sent_at") or 0.0),
                    synced,
                    1,
                )
            )
        self.conn.executemany(
            """
            insert into group_members(
              group_id, user_id, nickname, card, role, title,
              joined_at, last_sent_at, last_synced_at, active
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              nickname = excluded.nickname,
              card = excluded.card,
              role = excluded.role,
              title = excluded.title,
              joined_at = excluded.joined_at,
              last_sent_at = excluded.last_sent_at,
              last_synced_at = excluded.last_synced_at,
              active = excluded.active
            """,
            clean_members,
        )
        self.conn.commit()
        return len(clean_members)

    def group_member(self, group_id: int, user_id: int) -> GroupMember | None:
        row = self.conn.execute(
            """
            select group_id, user_id, nickname, card, role, title,
                   joined_at, last_sent_at, last_synced_at, active
            from group_members
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        return _group_member_from_row(row) if row else None

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
        safe_limit = max(0, int(limit))
        if safe_limit <= 0:
            return []
        fetch_limit = max(safe_limit * 4, safe_limit + 20)
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, fetch_limit),
        ).fetchall()
        rows = _dedupe_recent_message_rows(rows, safe_limit)
        return [_message_from_row(row) for row in reversed(rows)]

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
        preferred_user_id: int | None = None,
        preferred_limit: int = 0,
        preferred_score_multiplier: float = 1.0,
        preferred_score_bonus: float = 0.0,
        per_user_limit: int = 1,
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
            select id, group_id, user_id, nickname, text, is_bot, created_at
            from messages
            where group_id = ? and id > ? and id <= ? {bot_filter}
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
        rules: list[tuple],
        *,
        keep: int = 80,
    ) -> None:
        now = time.time()
        clean_rules: list[tuple[str, str, str, tuple[int, ...], tuple[int, ...]]] = []
        for raw_rule in rules:
            if len(raw_rule) < 3:
                continue
            situation, style, source_text = (str(raw_rule[0]), str(raw_rule[1]), str(raw_rule[2]))
            source_user_ids = tuple(int(value) for value in (raw_rule[3] if len(raw_rule) > 3 else ()) if int(value) > 0)
            source_message_ids = tuple(int(value) for value in (raw_rule[4] if len(raw_rule) > 4 else ()) if int(value) > 0)
            if situation.strip() and style.strip():
                clean_rules.append((situation.strip(), style.strip(), source_text.strip(), source_user_ids, source_message_ids))
        if not clean_rules:
            return
        self.conn.executemany(
            """
            insert into style_rules(
              group_id, situation, style, source_text, created_at, scope,
              source_user_ids_json, source_message_ids_json, support_user_count,
              evidence_count, confidence, status, valid_to
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            [
                (
                    group_id, situation[:60], style[:80], source_text[:200], now,
                    "group" if not user_ids or len(set(user_ids)) >= 2 else "personal",
                    json.dumps(list(dict.fromkeys(user_ids)), ensure_ascii=False),
                    json.dumps(list(dict.fromkeys(message_ids)), ensure_ascii=False),
                    max(1, len(set(user_ids))), max(1, len(set(message_ids))),
                    0.82 if len(set(user_ids)) >= 2 else (0.65 if not user_ids else 0.58),
                    now + (90 if len(set(user_ids)) >= 2 or not user_ids else 30) * 24 * 60 * 60,
                )
                for situation, style, source_text, user_ids, message_ids in clean_rules
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
            select group_id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to
            from style_rules
            where group_id = ? and status = 'active' and (valid_to is null or valid_to > ?)
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, time.time(), limit),
        ).fetchall()
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
                scope=str(row["scope"]),
                source_user_ids=tuple(_loads_int_list(row["source_user_ids_json"])),
                source_message_ids=tuple(_loads_int_list(row["source_message_ids_json"])),
                support_user_count=int(row["support_user_count"]),
                evidence_count=int(row["evidence_count"]),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
                valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
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
        speaker_user_id: int | None = None,
    ) -> list[StyleRule]:
        rows = self.conn.execute(
            """
            select group_id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to
            from style_rules
            where group_id = ? and status = 'active' and (valid_to is null or valid_to > ?)
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, time.time(), candidate_limit),
        ).fetchall()
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            source_user_ids = tuple(_loads_int_list(row["source_user_ids_json"]))
            if str(row["scope"]) == "personal" and speaker_user_id not in source_user_ids:
                continue
            haystack = f"{row['situation']} {row['style']} {row['source_text']}"
            score = _text_relevance_score(query, haystack)
            if score > 0:
                confidence = max(0.1, min(1.0, float(row["confidence"])))
                score = score * (0.5 + 0.5 * confidence) + min(2, max(0, int(row["support_user_count"]) - 1)) * 0.5
                scored.append((score, float(row["created_at"]), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
                scope=str(row["scope"]),
                source_user_ids=tuple(_loads_int_list(row["source_user_ids_json"])),
                source_message_ids=tuple(_loads_int_list(row["source_message_ids_json"])),
                support_user_count=int(row["support_user_count"]),
                evidence_count=int(row["evidence_count"]),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
                valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
            )
            for _, _, row in scored[:limit]
        ]

    def migrate_focused_style_rules(self, group_id: int, focused_user_id: int) -> dict[str, int]:
        """Conservatively scope legacy rules from one prolific speaker without deleting useful group style."""
        rows = self.conn.execute(
            """
            select id, situation, style, source_text
            from style_rules
            where group_id = ? and scope = 'legacy' and status = 'active'
            order by id desc
            """,
            (group_id,),
        ).fetchall()
        personal = kept_group = expired_duplicates = expired_literal = 0
        seen_focused: set[tuple[str, str]] = set()
        marker = f"[#{str(focused_user_id)[-5:]}]"
        for row in rows:
            source_text = str(row["source_text"])
            exact = self.conn.execute(
                """
                select id, user_id from messages
                where group_id = ? and is_bot = 0 and text = ?
                order by id desc limit 1
                """,
                (group_id, source_text),
            ).fetchone()
            is_focused = bool(exact and int(exact["user_id"]) == focused_user_id) or source_text.startswith(marker) or marker in source_text[:80]
            if not is_focused:
                continue
            message_ids = [int(exact["id"])] if exact else []
            key = (_compact_text(str(row["situation"])), _compact_text(str(row["style"])))
            preserve_as_group = "目移" in f"{row['situation']} {row['style']} {source_text}"
            if preserve_as_group:
                self.conn.execute(
                    """
                    update style_rules
                    set scope = 'group', source_user_ids_json = ?, source_message_ids_json = ?,
                        confidence = 0.78, support_user_count = 1, evidence_count = 1
                    where id = ?
                    """,
                    (json.dumps([focused_user_id]), json.dumps(message_ids), int(row["id"])),
                )
                kept_group += 1
                continue
            literal_personal = any(token in str(row["style"]).casefold() for token in ("xhn", "扶她"))
            if literal_personal:
                self.conn.execute(
                    "update style_rules set status = 'expired', valid_to = ? where id = ?",
                    (time.time(), int(row["id"])),
                )
                expired_literal += 1
                continue
            if key in seen_focused:
                self.conn.execute(
                    "update style_rules set status = 'expired', valid_to = ? where id = ?",
                    (time.time(), int(row["id"])),
                )
                expired_duplicates += 1
                continue
            seen_focused.add(key)
            self.conn.execute(
                """
                    update style_rules
                    set scope = 'personal', source_user_ids_json = ?, source_message_ids_json = ?,
                    confidence = 0.55, support_user_count = 1, evidence_count = 1,
                    valid_to = ?
                    where id = ?
                """,
                (
                    json.dumps([focused_user_id]), json.dumps(message_ids),
                    time.time() + 60 * 24 * 60 * 60, int(row["id"]),
                ),
            )
            personal += 1
        self.conn.commit()
        return {
            "personal": personal,
            "kept_group": kept_group,
            "expired_duplicates": expired_duplicates,
            "expired_literal": expired_literal,
        }

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
            select
              p.group_id,
              p.user_id,
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at
            from member_profiles p
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
            where p.group_id = ? and p.user_id in ({placeholders})
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
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at,
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
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
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
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at,
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
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
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

    def upsert_memory_atom(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        expires_at: float | None = None,
        evidence_type: str | None = None,
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        status: str = "active",
        supersedes_id: int | None = None,
    ) -> int:
        clean_content = re.sub(r"\s+", " ", content).strip()
        if not clean_content:
            return 0
        clean_type = atom_type.strip()[:32] or "note"
        clean_source = source.strip()[:80] or "manual"
        source_key = _source_message_key(source_message_id)
        clean_evidence_type = _normalize_memory_evidence_type(
            evidence_type or _infer_memory_evidence_type(clean_source, source_key)
        )
        clean_status = _normalize_memory_atom_status(status)
        effective_valid_to = valid_to if valid_to is not None else expires_at
        now = time.time()
        existing = None
        if supersedes_id is None:
            existing = self.conn.execute(
                """
                select id, source, confidence, importance, expires_at,
                       evidence_type, source_message_id, observed_at,
                       valid_from, valid_to, status, supersedes_id
                from memory_atoms
                where group_id = ?
                  and atom_type = ?
                  and coalesce(subject_user_id, -1) = coalesce(?, -1)
                  and coalesce(object_user_id, -1) = coalesce(?, -1)
                  and content = ?
                  and status = 'active'
                order by updated_at desc
                limit 1
                """,
                (group_id, clean_type, subject_user_id, object_user_id, clean_content[:420]),
            ).fetchone()
        if existing:
            atom_id = int(existing["id"])
            before = {
                "source": str(existing["source"]),
                "confidence": float(existing["confidence"]),
                "importance": float(existing["importance"]),
                "expires_at": existing["expires_at"],
                "evidence_type": str(existing["evidence_type"]),
                "source_message_id": existing["source_message_id"],
                "observed_at": existing["observed_at"],
                "valid_from": existing["valid_from"],
                "valid_to": existing["valid_to"],
                "status": str(existing["status"]),
                "supersedes_id": existing["supersedes_id"],
            }
            after = {
                "source": clean_source,
                "confidence": _clamp_float(confidence, 0.0, 1.0),
                "importance": _clamp_float(importance, 0.0, 1.0),
                "expires_at": effective_valid_to,
                "evidence_type": clean_evidence_type,
                "source_message_id": source_key,
                "observed_at": observed_at if observed_at is not None else existing["observed_at"],
                "valid_from": valid_from if valid_from is not None else existing["valid_from"],
                "valid_to": effective_valid_to,
                "status": clean_status,
                "supersedes_id": supersedes_id if supersedes_id is not None else existing["supersedes_id"],
            }
            resulting_valid_from = after["valid_from"]
            resulting_valid_to = after["valid_to"]
            if (
                resulting_valid_from is not None
                and resulting_valid_to is not None
                and float(resulting_valid_to) < float(resulting_valid_from)
            ):
                raise ValueError("memory atom valid_to cannot be earlier than valid_from")
            if before == after:
                return atom_id
            try:
                self.conn.execute(
                    """
                    update memory_atoms
                    set source = ?, confidence = ?, importance = ?,
                        expires_at = ?, evidence_type = ?, source_message_id = ?,
                        observed_at = coalesce(?, observed_at),
                        valid_from = coalesce(?, valid_from), valid_to = ?,
                        status = ?, supersedes_id = coalesce(?, supersedes_id),
                        updated_at = ?
                    where id = ?
                    """,
                    (
                        clean_source,
                        _clamp_float(confidence, 0.0, 1.0),
                        _clamp_float(importance, 0.0, 1.0),
                        effective_valid_to,
                        clean_evidence_type,
                        source_key,
                        observed_at,
                        valid_from,
                        effective_valid_to,
                        clean_status,
                        supersedes_id,
                        now,
                        atom_id,
                    ),
                )
                self._insert_memory_atom_audit_event(
                    atom_id=atom_id,
                    action="refreshed",
                    evidence_type=clean_evidence_type,
                    source=clean_source,
                    source_message_id=source_key,
                    actor_user_id=None,
                    detail="legacy upsert refreshed existing atom",
                    observed_at=observed_at if observed_at is not None else now,
                    metadata={"before": before, "after": after},
                )
                self.conn.commit()
                return atom_id
            except Exception:
                self.conn.rollback()
                raise
        return self.add_memory_atom(
            atom_type=clean_type,
            group_id=group_id,
            content=clean_content,
            source=clean_source,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            evidence_type=clean_evidence_type,
            source_message_id=source_key,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=effective_valid_to,
            confidence=confidence,
            importance=importance,
            status=clean_status,
            supersedes_id=supersedes_id,
        )

    def add_memory_atom(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str = "manual",
        evidence_type: str | None = None,
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        status: str = "active",
        supersedes_id: int | None = None,
        actor_user_id: int | None = None,
        audit_detail: str = "",
    ) -> int:
        clean_content = re.sub(r"\s+", " ", content).strip()
        if not clean_content:
            return 0
        source_key = _source_message_key(source_message_id)
        clean_evidence_type = _normalize_memory_evidence_type(
            evidence_type or _infer_memory_evidence_type(source, source_key)
        )
        clean_status = _normalize_memory_atom_status(status)
        now = time.time()
        observed = float(observed_at) if observed_at is not None else now
        starts = float(valid_from) if valid_from is not None else observed
        ends = float(valid_to) if valid_to is not None else None
        if clean_status == "expired" and ends is None:
            ends = observed
        if ends is not None and ends < starts:
            raise ValueError("memory atom valid_to cannot be earlier than valid_from")
        try:
            superseded_event_id = 0
            if supersedes_id is not None:
                target = self.memory_atom(supersedes_id)
                if target is None:
                    raise ValueError(f"superseded memory atom does not exist: {supersedes_id}")
                if target.group_id != group_id:
                    raise ValueError("superseded memory atom must belong to the same group")
                if target.status not in {"active", "disputed"}:
                    raise ValueError(f"memory atom {supersedes_id} is already {target.status}")
                if clean_status != "active":
                    raise ValueError("a replacement memory atom must start as active")
                if target.valid_from is not None and observed < target.valid_from:
                    raise ValueError("replacement cannot predate the superseded atom's valid_from")
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'superseded', valid_to = ?, expires_at = ?, updated_at = ?
                    where id = ?
                    """,
                    (observed, observed, now, supersedes_id),
                )
                superseded_event_id = self._insert_memory_atom_audit_event(
                    atom_id=supersedes_id,
                    action="superseded",
                    evidence_type=clean_evidence_type,
                    source=source,
                    source_message_id=source_key,
                    actor_user_id=actor_user_id,
                    detail=audit_detail or "superseded by replacement atom",
                    observed_at=observed,
                )
            atom_id = self._insert_memory_atom_record(
                atom_type=atom_type,
                group_id=group_id,
                content=clean_content,
                source=source,
                evidence_type=clean_evidence_type,
                source_message_id=source_key,
                observed_at=observed,
                valid_from=starts,
                valid_to=ends,
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                confidence=confidence,
                importance=importance,
                status=clean_status,
                supersedes_id=supersedes_id,
                actor_user_id=actor_user_id,
                audit_action="created",
                audit_detail=audit_detail or "memory atom created",
                now=now,
            )
            if superseded_event_id:
                self.conn.execute(
                    "update memory_atom_audit_events set metadata_json = ? where id = ?",
                    (json.dumps({"replacement_atom_id": atom_id}), superseded_event_id),
                )
            self.conn.commit()
            return atom_id
        except Exception:
            self.conn.rollback()
            raise

    def _insert_memory_atom_record(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str,
        evidence_type: str,
        source_message_id: str | None,
        observed_at: float,
        valid_from: float,
        valid_to: float | None,
        subject_user_id: int | None,
        object_user_id: int | None,
        confidence: float,
        importance: float,
        status: str,
        supersedes_id: int | None,
        actor_user_id: int | None,
        audit_action: str,
        audit_detail: str,
        now: float,
    ) -> int:
        clean_type = atom_type.strip()[:32] or "note"
        clean_source = source.strip()[:80] or evidence_type
        cursor = self.conn.execute(
            """
            insert into memory_atoms(
              atom_type, group_id, subject_user_id, object_user_id, content,
              source, evidence_type, source_message_id, observed_at,
              valid_from, valid_to, confidence, importance, status,
              supersedes_id, expires_at, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_type,
                group_id,
                subject_user_id,
                object_user_id,
                content[:420],
                clean_source,
                evidence_type,
                source_message_id,
                observed_at,
                valid_from,
                valid_to,
                _clamp_float(confidence, 0.0, 1.0),
                _clamp_float(importance, 0.0, 1.0),
                status,
                supersedes_id,
                valid_to,
                now,
                now,
            ),
        )
        atom_id = int(cursor.lastrowid or 0)
        self._insert_memory_atom_audit_event(
            atom_id=atom_id,
            action=audit_action,
            evidence_type=evidence_type,
            source=clean_source,
            source_message_id=source_message_id,
            actor_user_id=actor_user_id,
            detail=audit_detail,
            observed_at=observed_at,
        )
        return atom_id

    def add_memory_counter_evidence(
        self,
        atom_id: int,
        *,
        content: str,
        source: str,
        evidence_type: str = "message",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        confidence: float = 0.8,
        mark_disputed: bool = True,
    ) -> int:
        atom = self.memory_atom(atom_id)
        clean_content = re.sub(r"\s+", " ", content).strip()
        if atom is None or not clean_content:
            return 0
        clean_evidence_type = _normalize_memory_evidence_type(evidence_type)
        observed = float(observed_at) if observed_at is not None else time.time()
        try:
            if mark_disputed and atom.status == "active":
                self.conn.execute(
                    "update memory_atoms set status = 'disputed', updated_at = ? where id = ?",
                    (time.time(), atom_id),
                )
            event_id = self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="counter_evidence",
                evidence_type=clean_evidence_type,
                source=source,
                source_message_id=_source_message_key(source_message_id),
                actor_user_id=actor_user_id,
                detail=clean_content[:420],
                observed_at=observed,
                metadata={"confidence": _clamp_float(confidence, 0.0, 1.0)},
            )
            self.conn.commit()
            return event_id
        except Exception:
            self.conn.rollback()
            raise

    def dispute_memory_atom(
        self,
        atom_id: int,
        *,
        content: str,
        source: str,
        evidence_type: str = "message",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        confidence: float = 0.8,
    ) -> bool:
        return bool(
            self.add_memory_counter_evidence(
                atom_id,
                content=content,
                source=source,
                evidence_type=evidence_type,
                source_message_id=source_message_id,
                observed_at=observed_at,
                actor_user_id=actor_user_id,
                confidence=confidence,
                mark_disputed=True,
            )
        )

    def expire_memory_atom(
        self,
        atom_id: int,
        *,
        reason: str = "",
        source: str = "manual",
        observed_at: float | None = None,
        actor_user_id: int | None = None,
    ) -> bool:
        atom = self.memory_atom(atom_id)
        if atom is None or atom.status not in {"active", "disputed"}:
            return False
        observed = float(observed_at) if observed_at is not None else time.time()
        if atom.valid_from is not None and observed < atom.valid_from:
            raise ValueError("expiry cannot predate the memory atom's valid_from")
        try:
            self.conn.execute(
                """
                update memory_atoms
                set status = 'expired', valid_to = ?, expires_at = ?, updated_at = ?
                where id = ?
                """,
                (observed, observed, time.time(), atom_id),
            )
            self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="expired",
                evidence_type="manual",
                source=source,
                source_message_id=None,
                actor_user_id=actor_user_id,
                detail=(reason.strip() or "memory atom expired")[:420],
                observed_at=observed,
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def correct_memory_atom(
        self,
        atom_id: int,
        *,
        content: str,
        source: str = "manual_correction",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        reason: str = "",
        confidence: float | None = None,
        importance: float | None = None,
        valid_to: float | None = None,
        atom_type: str | None = None,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
    ) -> int:
        old = self.memory_atom(atom_id)
        clean_content = re.sub(r"\s+", " ", content).strip()
        if old is None or old.status not in {"active", "disputed"} or not clean_content:
            return 0
        observed = float(observed_at) if observed_at is not None else time.time()
        if old.valid_from is not None and observed < old.valid_from:
            raise ValueError("correction cannot predate the memory atom's valid_from")
        if valid_to is not None and float(valid_to) < observed:
            raise ValueError("corrected memory valid_to cannot be earlier than observed_at")
        now = time.time()
        try:
            self.conn.execute(
                """
                update memory_atoms
                set status = 'superseded', valid_to = ?, expires_at = ?, updated_at = ?
                where id = ?
                """,
                (observed, observed, now, atom_id),
            )
            superseded_event_id = self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="superseded",
                evidence_type="manual",
                source=source,
                source_message_id=_source_message_key(source_message_id),
                actor_user_id=actor_user_id,
                detail=(reason.strip() or "replaced by manual correction")[:420],
                observed_at=observed,
            )
            new_atom_id = self._insert_memory_atom_record(
                atom_type=atom_type or old.atom_type,
                group_id=old.group_id,
                content=clean_content,
                source=source,
                evidence_type="manual",
                source_message_id=_source_message_key(source_message_id),
                observed_at=observed,
                valid_from=observed,
                valid_to=valid_to,
                subject_user_id=old.subject_user_id if subject_user_id is None else subject_user_id,
                object_user_id=old.object_user_id if object_user_id is None else object_user_id,
                confidence=old.confidence if confidence is None else confidence,
                importance=old.importance if importance is None else importance,
                status="active",
                supersedes_id=old.id,
                actor_user_id=actor_user_id,
                audit_action="manual_correction",
                audit_detail=(reason.strip() or f"manual correction of atom {old.id}")[:420],
                now=now,
            )
            self.conn.execute(
                "update memory_atom_audit_events set metadata_json = ? where id = ?",
                (json.dumps({"replacement_atom_id": new_atom_id}), superseded_event_id),
            )
            self.conn.commit()
            return new_atom_id
        except Exception:
            self.conn.rollback()
            raise

    def memory_atom(self, atom_id: int) -> MemoryAtom | None:
        row = self.conn.execute(
            f"select {_MEMORY_ATOM_SELECT_COLUMNS} from memory_atoms where id = ?",
            (atom_id,),
        ).fetchone()
        return _memory_atom_from_row(row) if row is not None else None

    def memory_atom_audit_trail(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        rows = self.conn.execute(
            """
            select * from (
              select id, atom_id, action, evidence_type, source, source_message_id,
                     actor_user_id, detail, observed_at, created_at, metadata_json
              from memory_atom_audit_events
              where atom_id = ?
              order by created_at desc, id desc
              limit ?
            )
            order by created_at asc, id asc
            """,
            (atom_id, max(1, int(limit))),
        ).fetchall()
        return [_memory_atom_audit_event_from_row(row) for row in rows]

    def memory_atom_events(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        return self.memory_atom_audit_trail(atom_id, limit=limit)

    def _insert_memory_atom_audit_event(
        self,
        *,
        atom_id: int,
        action: str,
        evidence_type: str,
        source: str,
        source_message_id: str | None,
        actor_user_id: int | None,
        detail: str,
        observed_at: float,
        metadata: dict[str, object] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            insert into memory_atom_audit_events(
              atom_id, action, evidence_type, source, source_message_id,
              actor_user_id, detail, observed_at, created_at, metadata_json
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom_id,
                action.strip()[:32] or "updated",
                _normalize_memory_evidence_type(evidence_type),
                source.strip()[:80] or evidence_type,
                source_message_id,
                actor_user_id,
                detail.strip()[:420],
                observed_at,
                time.time(),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid or 0)

    def delete_memory_atom(self, atom_id: int) -> bool:
        return self.expire_memory_atom(atom_id, reason="legacy delete_memory_atom")

    def recent_memory_atoms(self, group_id: int, limit: int) -> list[MemoryAtom]:
        now = time.time()
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            where group_id = ?
              and status = 'active'
              and (valid_from is null or valid_from <= ?)
              and (valid_to is null or valid_to > ?)
              and (expires_at is null or expires_at > ?)
            order by importance desc, updated_at desc, id desc
            limit ?
            """,
            (group_id, now, now, now, limit),
        ).fetchall()
        return [_memory_atom_from_row(row) for row in rows]

    def relevant_memory_atoms(
        self,
        group_id: int,
        query: str,
        *,
        subject_user_ids: list[int] | None = None,
        speaker_user_id: int | None = None,
        relationship_user_ids: list[int] | None = None,
        limit: int = 6,
        candidate_limit: int = 120,
        now: float | None = None,
    ) -> list[MemoryAtom]:
        current = time.time() if now is None else float(now)
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            where group_id = ?
              and status = 'active'
              and (valid_from is null or valid_from <= ?)
              and (valid_to is null or valid_to > ?)
              and (expires_at is null or expires_at > ?)
            """,
            (group_id, current, current, current),
        ).fetchall()
        subject_set = set(subject_user_ids or []) | set(relationship_user_ids or [])
        if speaker_user_id is not None:
            subject_set.add(int(speaker_user_id))
        has_query_terms = bool(_relevance_terms(query))
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            content = str(row["content"])
            lexical_score = float(_text_relevance_score(query, content))
            score = lexical_score
            subject = row["subject_user_id"]
            obj = row["object_user_id"]
            person_match = bool(subject_set and (subject in subject_set or obj in subject_set))
            if has_query_terms and lexical_score <= 0 and not person_match:
                continue
            if subject_set and subject in subject_set:
                score += 3.0
            if subject_set and obj in subject_set:
                score += 2.25
            if speaker_user_id is not None and subject == int(speaker_user_id):
                score += 1.5
            elif speaker_user_id is not None and obj == int(speaker_user_id):
                score += 0.75
            if str(row["atom_type"]).casefold() == "relation" and subject_set:
                if subject in subject_set or obj in subject_set:
                    score += 0.75
            score += float(row["importance"] or 0.0) * 2.0
            score += float(row["confidence"] or 0.0) * 0.75
            score += _memory_atom_recency_score(row, now=current)
            score += _memory_atom_feedback_score(row)
            if score <= 0:
                continue
            scored.append((score, float(row["updated_at"]), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if candidate_limit > 0:
            scored = scored[:candidate_limit]
        return [_memory_atom_from_row(row) for _, _, row in scored[:limit]]

    def expire_due_memory_atoms(
        self,
        *,
        now: float | None = None,
        group_id: int | None = None,
    ) -> int:
        current = time.time() if now is None else float(now)
        return self._expire_due_memory_atoms(current, group_id=group_id)

    def _expire_due_memory_atoms(self, now: float, *, group_id: int | None = None) -> int:
        group_clause = " and group_id = ?" if group_id is not None else ""
        params: tuple[object, ...] = (now, group_id) if group_id is not None else (now,)
        rows = self.conn.execute(
            f"""
            select id, evidence_type, source, source_message_id
            from memory_atoms
            where status = 'active'
              and coalesce(valid_to, expires_at) is not null
              and coalesce(valid_to, expires_at) <= ?
              {group_clause}
            """,
            params,
        ).fetchall()
        try:
            for row in rows:
                atom_id = int(row["id"])
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'expired',
                        valid_to = coalesce(valid_to, expires_at, ?),
                        expires_at = coalesce(expires_at, valid_to, ?),
                        updated_at = ?
                    where id = ?
                    """,
                    (now, now, now, atom_id),
                )
                self._insert_memory_atom_audit_event(
                    atom_id=atom_id,
                    action="expired",
                    evidence_type=str(row["evidence_type"] or "event"),
                    source="validity_window",
                    source_message_id=_source_message_key(row["source_message_id"]),
                    actor_user_id=None,
                    detail="validity window elapsed",
                    observed_at=now,
                )
            if rows:
                self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

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
        self.conn.execute("delete from inbound_message_events where group_id = ?", (group_id,))
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


def _group_info_from_row(row: sqlite3.Row) -> GroupInfo:
    return GroupInfo(
        group_id=int(row["group_id"]),
        group_name=str(row["group_name"]),
        member_count=int(row["member_count"] or 0),
        max_member_count=int(row["max_member_count"] or 0),
        last_synced_at=float(row["last_synced_at"] or 0.0),
    )


def _group_member_from_row(row: sqlite3.Row) -> GroupMember:
    return GroupMember(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        nickname=str(row["nickname"]),
        card=str(row["card"] or ""),
        role=str(row["role"] or ""),
        title=str(row["title"] or ""),
        joined_at=float(row["joined_at"] or 0.0),
        last_sent_at=float(row["last_sent_at"] or 0.0),
        last_synced_at=float(row["last_synced_at"] or 0.0),
        active=bool(row["active"]),
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


def _memory_atom_from_row(row: sqlite3.Row) -> MemoryAtom:
    columns = set(row.keys())
    subject = row["subject_user_id"]
    obj = row["object_user_id"]
    expires_at = row["expires_at"]
    source_message_id = row["source_message_id"] if "source_message_id" in columns else None
    observed_at = row["observed_at"] if "observed_at" in columns else row["created_at"]
    valid_from = row["valid_from"] if "valid_from" in columns else row["created_at"]
    valid_to = row["valid_to"] if "valid_to" in columns else expires_at
    supersedes_id = row["supersedes_id"] if "supersedes_id" in columns else None
    return MemoryAtom(
        id=int(row["id"]),
        atom_type=str(row["atom_type"]),
        group_id=int(row["group_id"]),
        subject_user_id=int(subject) if subject is not None else None,
        object_user_id=int(obj) if obj is not None else None,
        content=str(row["content"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        expires_at=float(expires_at) if expires_at is not None else None,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        evidence_type=str(row["evidence_type"] or "manual") if "evidence_type" in columns else "manual",
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        observed_at=float(observed_at) if observed_at is not None else float(row["created_at"]),
        valid_from=float(valid_from) if valid_from is not None else None,
        valid_to=float(valid_to) if valid_to is not None else None,
        status=str(row["status"] or "active") if "status" in columns else "active",
        supersedes_id=int(supersedes_id) if supersedes_id is not None else None,
    )


def _memory_atom_audit_event_from_row(row: sqlite3.Row) -> MemoryAtomAuditEvent:
    try:
        raw_metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, json.JSONDecodeError):
        raw_metadata = {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    source_message_id = row["source_message_id"]
    actor_user_id = row["actor_user_id"]
    return MemoryAtomAuditEvent(
        id=int(row["id"]),
        atom_id=int(row["atom_id"]),
        action=str(row["action"]),
        evidence_type=str(row["evidence_type"]),
        source=str(row["source"]),
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        actor_user_id=int(actor_user_id) if actor_user_id is not None else None,
        detail=str(row["detail"]),
        observed_at=float(row["observed_at"]),
        created_at=float(row["created_at"]),
        metadata=metadata,
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


def _clamp_float(value: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _normalize_memory_evidence_type(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in MEMORY_ATOM_EVIDENCE_TYPES:
        raise ValueError(
            f"unsupported memory evidence type: {value!r}; "
            f"expected one of {sorted(MEMORY_ATOM_EVIDENCE_TYPES)}"
        )
    return normalized


def _infer_memory_evidence_type(source: str, source_message_id: str | None) -> str:
    if source_message_id:
        return "message"
    normalized = str(source or "").strip().casefold()
    if normalized.startswith("message:"):
        return "message"
    if normalized == "manual" or normalized.startswith(("manual:", "manual_", "builtin")):
        return "manual"
    return "event"


def _normalize_memory_atom_status(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in MEMORY_ATOM_STATUSES:
        raise ValueError(
            f"unsupported memory atom status: {value!r}; "
            f"expected one of {sorted(MEMORY_ATOM_STATUSES)}"
        )
    return normalized


def _memory_atom_recency_score(row: sqlite3.Row, *, now: float) -> float:
    observed_at = row["observed_at"]
    timestamp = float(observed_at) if observed_at is not None else float(row["updated_at"] or 0.0)
    age_seconds = max(0.0, now - timestamp)
    return 1.5 / (1.0 + age_seconds / (7 * 24 * 60 * 60))


def _memory_atom_feedback_score(row: sqlite3.Row) -> float:
    atom_type = str(row["atom_type"] or "").casefold()
    source = str(row["source"] or "").casefold()
    content = str(row["content"] or "")
    score = 0.0
    if atom_type == "feedback":
        score += 1.5
    if source.startswith(("approval_", "recall_", "owner_feedback")):
        score += 0.75
    if "不准奏反馈" in content or "优质反馈" in content:
        score += 0.5
    return score


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


def _loads_int_list(value: object) -> list[int]:
    try:
        raw = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


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


def _source_message_key(source_message_id: int | str | None) -> str | None:
    if source_message_id is None:
        return None
    key = str(source_message_id).strip()
    return key or None
