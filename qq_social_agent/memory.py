from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .memory_identity import (
    PRIVATE_CHAT_ID_OFFSET,
    KEDAI_PRIMARY_USER_ID,
    KEDAI_ALT_USER_ID,
    LINKED_ACCOUNT_GROUPS,
    linked_account_ids,
    expand_linked_account_ids,
    linked_account_note,
)

# Re-export the shared records for existing callers of this module.
from .memory_models import (
    ChatMessage,
    RawCorpusExample,
    MemorySummary,
    StyleRule,
    MemberProfile,
    PrivateConversationState,
    GroupInfo,
    GroupMember,
    MemberImpression,
    MemberProfileSummary,
    BotSentMessage,
    RecalledReplyFeedback,
    ApprovedReplyFeedback,
    CustomJargonEntry,
    LLMUsageSummary,
    LLMUsageEvent,
    BotMetricEvent,
    BotMetricSummary,
    MemeAsset,
    MemoryAtom,
    MemoryAtomAuditEvent,
)
from .memory_text import _compact_text, _json_text_list, _clean_text_list
from .memory_metrics_repository import (
    MetricsRepository,
    _metric_event_from_row,
    _metric_where,
    _usage_time_where,
    _llm_usage_summary_from_row,
    _llm_usage_event_from_row,
)
from .memory_private_state_repository import (
    PrivateStateRepository,
    _clean_private_state_text,
    _clean_private_threads,
    _clean_private_state_fields,
    _private_conversation_state_from_row,
)
from .memory_meme_repository import (
    MemeRepository,
    _meme_asset_from_row,
    _meme_query_terms,
    _meme_relevance_score,
)

from .memory_repository_utils import (
    _message_from_row,
    _clamp_float,
    _unique_recent_ints,
    _text_relevance_score,
    _relevance_terms,
    _loads_int_list,
    _source_message_key,
)
from .memory_atom_repository import (
    MemoryAtomRepository,
    _memory_atom_from_row,
    _memory_atom_audit_event_from_row,
    _normalize_memory_evidence_type,
    _infer_memory_evidence_type,
    _normalize_memory_atom_status,
    _memory_atom_recency_score,
    _memory_atom_feedback_score,
    MEMORY_ATOM_EVIDENCE_TYPES,
    MEMORY_ATOM_STATUSES,
    _MEMORY_ATOM_SELECT_COLUMNS,
)
from .memory_style_repository import (
    StyleRepository,
    _style_rule_fingerprint,
    _normalize_style_rule_text,
    _style_rule_semantic_markers,
    _style_rules_are_semantically_mergeable,
    _style_rule_confidence,
    _prefer_specific_style_text,
    _style_text_specificity,
    _style_rule_value,
    _STYLE_RULE_CANONICAL_REPLACEMENTS,
    _STYLE_RULE_SEMANTIC_MARKERS,
)

from .interaction_state import InteractionStateStore
from .image_read_state import ImageReadStateStore


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._atom_repository = MemoryAtomRepository(self.conn)
        self._init_schema()
        self.interactions = InteractionStateStore(self.conn)
        self.images = ImageReadStateStore(self.conn)

        self._metrics_repository = MetricsRepository(self.conn)
        self._private_state_repository = PrivateStateRepository(self.conn)
        self._meme_repository = MemeRepository(self.conn)
        self._style_repository = StyleRepository(self.conn)

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
              correlation_id text,
              session_id text,
              message_segments_json text,
              raw_message_json text,
              sender_json text
            );

            create index if not exists idx_messages_group_time
              on messages(group_id, created_at);

            create index if not exists idx_messages_time
              on messages(created_at);

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
              created_at real not null,
              updated_at real,
              status text not null default 'active',
              locked integer not null default 0
            );

            create index if not exists idx_memory_summaries_group_time
              on memory_summaries(group_id, created_at);

            create table if not exists private_conversation_states (
              chat_id integer primary key,
              user_id integer not null,
              display_name text not null default '',
              relationship_note text not null default '',
              interaction_tone text not null default '',
              current_topic text not null default '',
              open_threads_json text not null default '[]',
              frozen_fields_json text not null default '[]',
              updated_at real not null
            );

            create index if not exists idx_private_conversation_states_updated
              on private_conversation_states(updated_at desc);

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

            create index if not exists idx_bot_sent_messages_time
              on bot_sent_messages(created_at);

            create index if not exists idx_bot_sent_messages_group_time
              on bot_sent_messages(group_id, created_at);

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

            create table if not exists meme_assets (
              id integer primary key autoincrement,
              sha256 text not null unique,
              source_group_id integer not null,
              source_user_id integer not null,
              source_message_id text not null,
              file_path text not null,
              mime_type text not null,
              byte_size integer not null,
              description text not null default '',
              tags_json text not null default '[]',
              enabled integer not null default 0,
              created_at real not null,
              updated_at real not null,
              last_used_at real,
              use_count integer not null default 0
            );

            create index if not exists idx_meme_assets_enabled_used
              on meme_assets(enabled, last_used_at, id);

            """
        )
        self._ensure_message_source_columns()
        self._ensure_memory_summary_admin_columns()
        self._ensure_group_directory_tables()
        self._ensure_approved_feedback_tags()
        self._ensure_llm_usage_source_key()
        self._ensure_memory_atom_v2()
        self._ensure_private_conversation_state_columns()
        self._ensure_style_rule_v2()
        self._ensure_meme_asset_columns()
        self.expire_due_memory_atoms()
        self._backfill_member_profiles()
        self._backfill_member_impressions()
        self._backfill_private_conversation_states()
        self.conn.execute("pragma optimize")
        self.conn.commit()

    def _ensure_private_conversation_state_columns(self) -> None:
        """Keep the private-state migration additive for old databases."""

        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(private_conversation_states)").fetchall()
        }
        if not columns:
            return
        additions = (
            ("display_name", "text not null default ''"),
            ("relationship_note", "text not null default ''"),
            ("interaction_tone", "text not null default ''"),
            ("current_topic", "text not null default ''"),
            ("open_threads_json", "text not null default '[]'"),
            ("frozen_fields_json", "text not null default '[]'"),
            ("updated_at", "real not null default 0"),
        )
        for name, declaration in additions:
            if name not in columns:
                self.conn.execute(f"alter table private_conversation_states add column {name} {declaration}")
        self.conn.execute(
            "create index if not exists idx_private_conversation_states_updated "
            "on private_conversation_states(updated_at desc)"
        )

    def _backfill_private_conversation_states(self) -> None:
        """Create harmless shells for existing direct chats, without inferring facts."""

        rows = self.conn.execute(
            """
            select message.group_id, message.user_id, message.nickname, message.created_at
            from messages as message
            join (
              select group_id, max(id) as latest_id
              from messages
              where group_id >= ? and is_bot = 0
              group by group_id
            ) as latest on latest.latest_id = message.id
            """,
            (PRIVATE_CHAT_ID_OFFSET,),
        ).fetchall()
        now = time.time()
        for row in rows:
            self.conn.execute(
                """
                insert or ignore into private_conversation_states(
                  chat_id, user_id, display_name, updated_at
                ) values (?, ?, ?, ?)
                """,
                (int(row["group_id"]), int(row["user_id"]), str(row["nickname"] or ""), now),
            )

    def _ensure_meme_asset_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(meme_assets)").fetchall()
        }
        if not columns:
            return
        if "enabled" not in columns:
            self.conn.execute("alter table meme_assets add column enabled integer not null default 0")
        if "last_used_at" not in columns:
            self.conn.execute("alter table meme_assets add column last_used_at real")
        if "use_count" not in columns:
            self.conn.execute("alter table meme_assets add column use_count integer not null default 0")

    def _ensure_memory_summary_admin_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(memory_summaries)").fetchall()
        }
        if "updated_at" not in columns:
            self.conn.execute("alter table memory_summaries add column updated_at real")
        if "status" not in columns:
            self.conn.execute("alter table memory_summaries add column status text not null default 'active'")
        if "locked" not in columns:
            self.conn.execute("alter table memory_summaries add column locked integer not null default 0")
        self.conn.execute(
            """
            update memory_summaries
            set updated_at = coalesce(updated_at, created_at),
                status = coalesce(nullif(status, ''), 'active'),
                locked = coalesce(locked, 0)
            """
        )
        self.conn.execute(
            """
            create index if not exists idx_memory_summaries_group_status_time
              on memory_summaries(group_id, status, created_at)
            """
        )

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
        if "session_id" not in columns:
            self.conn.execute("alter table messages add column session_id text")
        if "message_segments_json" not in columns:
            self.conn.execute("alter table messages add column message_segments_json text")
        if "raw_message_json" not in columns:
            self.conn.execute("alter table messages add column raw_message_json text")
        if "sender_json" not in columns:
            self.conn.execute("alter table messages add column sender_json text")
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
            ("rule_fingerprint", "text not null default ''"),
            ("last_seen_at", "real not null default 0"),
            ("merged_count", "integer not null default 0"),
        )
        for name, declaration in additions:
            if name not in columns:
                self.conn.execute(f"alter table style_rules add column {name} {declaration}")
        rows = self.conn.execute(
            """
            select id, situation, style, created_at, rule_fingerprint, last_seen_at
            from style_rules
            where coalesce(rule_fingerprint, '') = ''
               or coalesce(last_seen_at, 0) = 0
            """
        ).fetchall()
        for row in rows:
            fingerprint = str(row["rule_fingerprint"] or "") or _style_rule_fingerprint(
                str(row["situation"] or ""),
                str(row["style"] or ""),
            )
            last_seen_at = float(row["last_seen_at"] or row["created_at"] or time.time())
            self.conn.execute(
                """
                update style_rules
                set rule_fingerprint = ?, last_seen_at = ?
                where id = ?
                """,
                (fingerprint, last_seen_at, int(row["id"])),
            )
        self.conn.execute(
            "create index if not exists idx_style_rules_scope_status on style_rules(group_id, scope, status, created_at)"
        )
        self.conn.execute(
            "create index if not exists idx_style_rules_fingerprint on style_rules(group_id, status, scope, rule_fingerprint)"
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

    def admin_recent_metric_events(
        self,
        *,
        event_types: tuple[str, ...] = (),
        group_id: int | None = None,
        limit: int = 80,
    ) -> list[sqlite3.Row]:
        return self._metrics_repository.admin_recent_metric_events(
            event_types=event_types,
            group_id=group_id,
            limit=limit,
        )

    def admin_recent_memory_atoms(
        self,
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        user_id: int | None = None,
        atom_type: str = "",
        query: str = "",
    ) -> list[MemoryAtom]:
        return self._atom_repository.admin_recent_memory_atoms(
            group_id=group_id,
            status=status,
            limit=limit,
            user_id=user_id,
            atom_type=atom_type,
            query=query,
        )

    def admin_review_memory_atom(
        self,
        atom_id: int,
        *,
        action: str,
        actor_user_id: int | None = None,
        note: str = "",
    ) -> bool:
        return self._atom_repository.admin_review_memory_atom(
            atom_id,
            action=action,
            actor_user_id=actor_user_id,
            note=note,
        )

    def admin_merge_memory_atoms(
        self,
        source_atom_id: int,
        target_atom_id: int,
        *,
        actor_user_id: int | None = None,
        note: str = "",
    ) -> bool:
        return self._atom_repository.admin_merge_memory_atoms(
            source_atom_id,
            target_atom_id,
            actor_user_id=actor_user_id,
            note=note,
        )

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

    def messages_for_style_learning(
        self,
        group_id: int,
        *,
        limit: int,
    ) -> list[ChatMessage]:
        return self._style_repository.messages_for_style_learning(group_id, limit=limit)

    def last_style_rule_at(self, group_id: int) -> float:
        return self._style_repository.last_style_rule_at(group_id)

    def add_style_rules(
        self,
        group_id: int,
        rules: list[tuple],
        *,
        keep: int = 45,
    ) -> dict[str, int]:
        return self._style_repository.add_style_rules(group_id, rules, keep=keep)

    def _find_mergeable_style_rule(
        self,
        group_id: int,
        fingerprint: str,
        *,
        situation: str,
        style: str,
        scope: str,
        source_user_ids: tuple[int, ...],
        now: float,
    ) -> sqlite3.Row | None:
        return self._style_repository._find_mergeable_style_rule(
            group_id,
            fingerprint,
            situation=situation,
            style=style,
            scope=scope,
            source_user_ids=source_user_ids,
            now=now,
        )

    def _merge_style_rule(
        self,
        row: sqlite3.Row,
        *,
        situation: str,
        style: str,
        source_text: str,
        source_user_ids: tuple[int, ...],
        source_message_ids: tuple[int, ...],
        now: float,
    ) -> None:
        return self._style_repository._merge_style_rule(
            row,
            situation=situation,
            style=style,
            source_text=source_text,
            source_user_ids=source_user_ids,
            source_message_ids=source_message_ids,
            now=now,
        )

    def _expire_low_value_style_rules(self, group_id: int, *, keep: int, now: float) -> int:
        return self._style_repository._expire_low_value_style_rules(group_id, keep=keep, now=now)

    def recent_style_rules(self, group_id: int, limit: int) -> list[StyleRule]:
        return self._style_repository.recent_style_rules(group_id, limit)

    def relevant_style_rules(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 80,
        speaker_user_id: int | None = None,
    ) -> list[StyleRule]:
        return self._style_repository.relevant_style_rules(
            group_id,
            query,
            limit=limit,
            candidate_limit=candidate_limit,
            speaker_user_id=speaker_user_id,
        )

    def migrate_focused_style_rules(self, group_id: int, focused_user_id: int) -> dict[str, int]:
        """Conservatively scope legacy rules from one prolific speaker without deleting useful group style."""

        return self._style_repository.migrate_focused_style_rules(group_id, focused_user_id)

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
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
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

    def latest_member_profile_summary(
        self,
        group_id: int,
        user_id: int,
    ) -> MemberProfileSummary | None:
        summaries = self.recent_member_profile_summaries(group_id, user_id, limit=1)
        return summaries[0] if summaries else None

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
        keep_per_member: int = 3,
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

    def prune_member_profile_summaries(self, *, keep_per_member: int = 3) -> int:
        keep = max(1, int(keep_per_member))
        pairs = self.conn.execute(
            "select distinct group_id, user_id from member_profile_summaries"
        ).fetchall()
        deleted = 0
        for row in pairs:
            cursor = self.conn.execute(
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
                (row["group_id"], row["user_id"], row["group_id"], row["user_id"], keep),
            )
            deleted += int(cursor.rowcount or 0)
        self.conn.commit()
        return deleted

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
        return self._metrics_repository.add_metric_event(
            event_type=event_type,
            group_id=group_id,
            user_id=user_id,
            stage=stage,
            action=action,
            metadata=metadata,
            created_at=created_at,
        )

    def metric_summary(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 80,
    ) -> list[BotMetricSummary]:
        return self._metrics_repository.metric_summary(start_at=start_at, end_at=end_at, group_id=group_id, limit=limit)

    def metric_event_count(self, event_type: str) -> int:
        return self._metrics_repository.metric_event_count(event_type)

    def recent_metric_events(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 12,
    ) -> list[BotMetricEvent]:
        return self._metrics_repository.recent_metric_events(
            start_at=start_at,
            end_at=end_at,
            group_id=group_id,
            limit=limit,
        )

    def prune_metric_events(
        self,
        *,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> dict[str, int]:
        return self._metrics_repository.prune_metric_events(max_age_seconds=max_age_seconds, max_rows=max_rows)

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
        return self._atom_repository.upsert_memory_atom(
            atom_type=atom_type,
            group_id=group_id,
            content=content,
            source=source,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            confidence=confidence,
            importance=importance,
            expires_at=expires_at,
            evidence_type=evidence_type,
            source_message_id=source_message_id,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            status=status,
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
        return self._atom_repository.add_memory_atom(
            atom_type=atom_type,
            group_id=group_id,
            content=content,
            source=source,
            evidence_type=evidence_type,
            source_message_id=source_message_id,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            confidence=confidence,
            importance=importance,
            status=status,
            supersedes_id=supersedes_id,
            actor_user_id=actor_user_id,
            audit_detail=audit_detail,
        )

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
        return self._atom_repository._insert_memory_atom_record(
            atom_type=atom_type,
            group_id=group_id,
            content=content,
            source=source,
            evidence_type=evidence_type,
            source_message_id=source_message_id,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            confidence=confidence,
            importance=importance,
            status=status,
            supersedes_id=supersedes_id,
            actor_user_id=actor_user_id,
            audit_action=audit_action,
            audit_detail=audit_detail,
            now=now,
        )

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
        return self._atom_repository.add_memory_counter_evidence(
            atom_id,
            content=content,
            source=source,
            evidence_type=evidence_type,
            source_message_id=source_message_id,
            observed_at=observed_at,
            actor_user_id=actor_user_id,
            confidence=confidence,
            mark_disputed=mark_disputed,
        )

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
        return self._atom_repository.dispute_memory_atom(
            atom_id,
            content=content,
            source=source,
            evidence_type=evidence_type,
            source_message_id=source_message_id,
            observed_at=observed_at,
            actor_user_id=actor_user_id,
            confidence=confidence,
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
        return self._atom_repository.expire_memory_atom(
            atom_id,
            reason=reason,
            source=source,
            observed_at=observed_at,
            actor_user_id=actor_user_id,
        )

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
        return self._atom_repository.correct_memory_atom(
            atom_id,
            content=content,
            source=source,
            source_message_id=source_message_id,
            observed_at=observed_at,
            actor_user_id=actor_user_id,
            reason=reason,
            confidence=confidence,
            importance=importance,
            valid_to=valid_to,
            atom_type=atom_type,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
        )

    def memory_atom(self, atom_id: int) -> MemoryAtom | None:
        return self._atom_repository.memory_atom(atom_id)

    def memory_atom_audit_trail(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        return self._atom_repository.memory_atom_audit_trail(atom_id, limit=limit)

    def memory_atom_events(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        return self._atom_repository.memory_atom_events(atom_id, limit=limit)

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
        return self._atom_repository._insert_memory_atom_audit_event(
            atom_id=atom_id,
            action=action,
            evidence_type=evidence_type,
            source=source,
            source_message_id=source_message_id,
            actor_user_id=actor_user_id,
            detail=detail,
            observed_at=observed_at,
            metadata=metadata,
        )

    def delete_memory_atom(self, atom_id: int) -> bool:
        return self._atom_repository.delete_memory_atom(atom_id)

    def recent_memory_atoms(self, group_id: int, limit: int) -> list[MemoryAtom]:
        return self._atom_repository.recent_memory_atoms(group_id, limit)

    def active_memory_atoms_for_subject(
        self,
        group_id: int,
        subject_user_id: int | None,
        *,
        atom_types: tuple[str, ...] | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        return self._atom_repository.active_memory_atoms_for_subject(
            group_id,
            subject_user_id,
            atom_types=atom_types,
            limit=limit,
        )

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
        return self._atom_repository.relevant_memory_atoms(
            group_id,
            query,
            subject_user_ids=subject_user_ids,
            speaker_user_id=speaker_user_id,
            relationship_user_ids=relationship_user_ids,
            limit=limit,
            candidate_limit=candidate_limit,
            now=now,
        )

    def expire_due_memory_atoms(
        self,
        *,
        now: float | None = None,
        group_id: int | None = None,
    ) -> int:
        return self._atom_repository.expire_due_memory_atoms(now=now, group_id=group_id)

    def _expire_due_memory_atoms(self, now: float, *, group_id: int | None = None) -> int:
        return self._atom_repository._expire_due_memory_atoms(now, group_id=group_id)

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
        return self._metrics_repository.add_llm_usage(
            task=task,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            created_at=created_at,
            source_key=source_key,
        )

    def llm_usage_summary(
        self,
        *,
        since_seconds: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
    ) -> list[LLMUsageSummary]:
        return self._metrics_repository.llm_usage_summary(since_seconds=since_seconds, start_at=start_at, end_at=end_at)

    def recent_llm_usage_events(
        self,
        *,
        since_seconds: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
        limit: int = 8,
    ) -> list[LLMUsageEvent]:
        return self._metrics_repository.recent_llm_usage_events(
            since_seconds=since_seconds,
            start_at=start_at,
            end_at=end_at,
            limit=limit,
        )

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
        self.images.clear_group(group_id)
        self.conn.execute("delete from interaction_events where group_id = ?", (group_id,))
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

    def private_conversation_state(self, chat_id: int) -> PrivateConversationState | None:
        return self._private_state_repository.private_conversation_state(chat_id)

    def recent_private_conversation_states(self, limit: int = 80) -> list[PrivateConversationState]:
        return self._private_state_repository.recent_private_conversation_states(limit)

    def update_private_conversation_state(
        self,
        *,
        chat_id: int,
        user_id: int,
        display_name: str | None = None,
        relationship_note: str | None = None,
        interaction_tone: str | None = None,
        current_topic: str | None = None,
        open_threads: list[str] | tuple[str, ...] | None = None,
        frozen_fields: list[str] | tuple[str, ...] | None = None,
    ) -> PrivateConversationState:
        """Create or edit direct-chat state without changing factual memory atoms.

        ``None`` preserves a field; an empty string/list deliberately clears it.
        This makes correction in the WebUI predictable and auditable facts remain
        in ``memory_atoms`` instead of becoming mutable prompt text.
        """

        return self._private_state_repository.update_private_conversation_state(
            chat_id=chat_id,
            user_id=user_id,
            display_name=display_name,
            relationship_note=relationship_note,
            interaction_tone=interaction_tone,
            current_topic=current_topic,
            open_threads=open_threads,
            frozen_fields=frozen_fields,
        )

    def refresh_private_conversation_learning(
        self,
        *,
        chat_id: int,
        user_id: int,
        display_name: str,
        open_threads: list[str] | tuple[str, ...],
    ) -> PrivateConversationState:
        """Refresh short-term private state while honouring manual field locks."""

        return self._private_state_repository.refresh_private_conversation_learning(
            chat_id=chat_id,
            user_id=user_id,
            display_name=display_name,
            open_threads=open_threads,
        )

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
        return self._meme_repository.upsert_meme_asset(
            sha256=sha256,
            source_group_id=source_group_id,
            source_user_id=source_user_id,
            source_message_id=source_message_id,
            file_path=file_path,
            mime_type=mime_type,
            byte_size=byte_size,
            description=description,
            tags=tags,
            enabled=enabled,
        )

    def meme_assets_for_private(
        self,
        *,
        query: str = "",
        limit: int = 6,
        same_meme_cooldown_seconds: float = 6 * 60 * 60,
    ) -> list[MemeAsset]:
        return self._meme_repository.meme_assets_for_private(
            query=query,
            limit=limit,
            same_meme_cooldown_seconds=same_meme_cooldown_seconds,
        )

    def mark_meme_asset_used(self, meme_id: int) -> bool:
        return self._meme_repository.mark_meme_asset_used(meme_id)

    def meme_asset(self, meme_id: int) -> MemeAsset | None:
        return self._meme_repository.meme_asset(meme_id)


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
