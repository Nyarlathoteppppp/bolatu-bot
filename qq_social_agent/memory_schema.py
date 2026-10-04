"""Existing schema creation, additive migrations and ordered backfills."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable

from .memory_identity import PRIVATE_CHAT_ID_OFFSET
from .memory_style_repository import _style_rule_fingerprint


class MemorySchema:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        expire_due_memory_atoms: Callable[..., object],
        upsert_member_profile: Callable[..., object],
        update_member_impression: Callable[..., object],
    ):
        self.conn = conn
        self.expire_due_memory_atoms = expire_due_memory_atoms
        self._upsert_member_profile = upsert_member_profile
        self._update_member_impression = update_member_impression

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
