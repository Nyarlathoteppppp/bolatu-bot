from __future__ import annotations

import sqlite3
import time
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

from .memory_corpus import _raw_corpus_tags, _is_low_value_raw_corpus_text
from .memory_message_repository import (
    MessageRepository,
    _dedupe_recent_message_rows,
    _bot_sent_from_row,
    _recalled_feedback_from_row,
    _approved_feedback_from_row,
)
from .memory_member_repository import (
    MemberRepository,
    _group_info_from_row,
    _group_member_from_row,
    _impression_keywords,
    _counter_from_json,
    _recent_texts_from_json,
    _cap_counter,
    _top_counter_items,
    _profile_from_row,
    _member_impression_from_row,
    _member_profile_summary_from_row,
    _dedupe_names,
    _dedupe_ints,
)
from .memory_summary_repository import (
    SummaryRepository,
    _summary_from_row,
)
from .memory_schema import (
    MemorySchema,
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
        self._member_repository = MemberRepository(self.conn)
        self._schema = MemorySchema(
            self.conn,
            expire_due_memory_atoms=self.expire_due_memory_atoms,
            upsert_member_profile=self._upsert_member_profile,
            update_member_impression=self._update_member_impression,
        )
        self._init_schema()
        self.interactions = InteractionStateStore(self.conn)
        self.images = ImageReadStateStore(self.conn)

        self._metrics_repository = MetricsRepository(self.conn)
        self._private_state_repository = PrivateStateRepository(self.conn)
        self._meme_repository = MemeRepository(self.conn)
        self._style_repository = StyleRepository(self.conn)
        self._summary_repository = SummaryRepository(self.conn)
        self._message_repository = MessageRepository(
            self.conn,
            self.images,
            upsert_member_profile=self._upsert_member_profile,
            update_member_impression=self._update_member_impression,
        )

    def _configure_connection(self) -> None:
        self.conn.execute("pragma busy_timeout = 5000")
        self.conn.execute("pragma journal_mode = WAL")
        self.conn.execute("pragma synchronous = NORMAL")
        self.conn.execute("pragma temp_store = MEMORY")

    def _init_schema(self) -> None:
        return self._schema._init_schema()

    def _ensure_private_conversation_state_columns(self) -> None:
        """Keep the private-state migration additive for old databases."""

        return self._schema._ensure_private_conversation_state_columns()

    def _backfill_private_conversation_states(self) -> None:
        """Create harmless shells for existing direct chats, without inferring facts."""

        return self._schema._backfill_private_conversation_states()

    def _ensure_meme_asset_columns(self) -> None:
        return self._schema._ensure_meme_asset_columns()

    def _ensure_memory_summary_admin_columns(self) -> None:
        return self._schema._ensure_memory_summary_admin_columns()

    def _ensure_approved_feedback_tags(self) -> None:
        return self._schema._ensure_approved_feedback_tags()

    def _ensure_message_source_columns(self) -> None:
        return self._schema._ensure_message_source_columns()

    def _ensure_group_directory_tables(self) -> None:
        return self._schema._ensure_group_directory_tables()

    def _ensure_llm_usage_source_key(self) -> None:
        return self._schema._ensure_llm_usage_source_key()

    def _ensure_style_rule_v2(self) -> None:
        return self._schema._ensure_style_rule_v2()

    def _ensure_memory_atom_v2(self) -> None:
        return self._schema._ensure_memory_atom_v2()

    def _ensure_memory_atom_v2_schema(self) -> None:
        return self._schema._ensure_memory_atom_v2_schema()

    def _backfill_member_profiles(self) -> None:
        return self._schema._backfill_member_profiles()

    def _backfill_member_impressions(self) -> None:
        return self._schema._backfill_member_impressions()

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
        return self._message_repository.add_message(
            group_id,
            user_id,
            nickname,
            text,
            is_bot=is_bot,
            created_at=created_at,
            source_message_id=source_message_id,
            source_kind=source_kind,
            correlation_id=correlation_id,
            session_id=session_id,
            message_segments_json=message_segments_json,
            raw_message_json=raw_message_json,
            sender_json=sender_json,
        )

    def admin_recent_messages(self, *, group_id: int | None = None, limit: int = 80) -> list[sqlite3.Row]:
        return self._message_repository.admin_recent_messages(group_id=group_id, limit=limit)

    def admin_message(self, message_id: int) -> sqlite3.Row | None:
        return self._message_repository.admin_message(message_id)

    def update_message_context(self, message_id: int, text: str) -> None:
        """Enrich an existing message without moving its arrival time."""

        return self._message_repository.update_message_context(message_id, text)

    def fill_message_segments(self, message_id: int, segments: str) -> None:
        """Restore media data omitted by older history imports, in place."""

        return self._message_repository.fill_message_segments(message_id, segments)

    def admin_message_by_source(self, group_id: int, source_message_id: int | str | None) -> sqlite3.Row | None:
        return self._message_repository.admin_message_by_source(group_id, source_message_id)

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
        return self._message_repository.message_source_exists(group_id, source_message_id)

    def claim_inbound_message(
        self,
        group_id: int,
        source_message_id: int | str | None,
        *,
        correlation_id: str | None = None,
        created_at: float | None = None,
    ) -> bool:
        return self._message_repository.claim_inbound_message(
            group_id,
            source_message_id,
            correlation_id=correlation_id,
            created_at=created_at,
        )

    def upsert_group_info(
        self,
        *,
        group_id: int,
        group_name: str,
        member_count: int,
        max_member_count: int,
        last_synced_at: float | None = None,
    ) -> None:
        return self._member_repository.upsert_group_info(
            group_id=group_id,
            group_name=group_name,
            member_count=member_count,
            max_member_count=max_member_count,
            last_synced_at=last_synced_at,
        )

    def group_info(self, group_id: int) -> GroupInfo | None:
        return self._member_repository.group_info(group_id)

    def replace_group_members(
        self,
        group_id: int,
        members: list[dict[str, object]],
        *,
        synced_at: float | None = None,
    ) -> int:
        return self._member_repository.replace_group_members(group_id, members, synced_at=synced_at)

    def group_member(self, group_id: int, user_id: int) -> GroupMember | None:
        return self._member_repository.group_member(group_id, user_id)

    def _upsert_member_profile(
        self,
        group_id: int,
        user_id: int,
        display_name: str,
        *,
        last_seen_at: float,
    ) -> None:
        return self._member_repository._upsert_member_profile(
            group_id,
            user_id,
            display_name,
            last_seen_at=last_seen_at,
        )

    def recent_messages(self, group_id: int, limit: int) -> list[ChatMessage]:
        return self._message_repository.recent_messages(group_id, limit)

    def messages_since(self, group_id: int, *, since_at: float) -> list[ChatMessage]:
        """Read the short window excluded from RAG, without a message-count cutoff."""

        return self._message_repository.messages_since(group_id, since_at=since_at)

    def current_session_messages(self, group_id: int, *, gap_seconds: float) -> list[ChatMessage]:
        """Read back to the last private-chat gap instead of a fixed message count."""

        return self._message_repository.current_session_messages(group_id, gap_seconds=gap_seconds)

    def message_by_source(self, group_id: int, source_message_id: int | str | None) -> ChatMessage | None:
        return self._message_repository.message_by_source(group_id, source_message_id)

    def message_by_id(self, group_id: int, message_id: int) -> ChatMessage | None:
        return self._message_repository.message_by_id(group_id, message_id)

    def messages_between(
        self,
        group_id: int,
        *,
        start_at: float,
        end_at: float,
        limit: int,
    ) -> list[ChatMessage]:
        return self._message_repository.messages_between(group_id, start_at=start_at, end_at=end_at, limit=limit)

    def messages_before(self, group_id: int, *, before_at: float, limit: int) -> list[ChatMessage]:
        return self._message_repository.messages_before(group_id, before_at=before_at, limit=limit)

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
        return self._message_repository.relevant_raw_corpus_examples(
            group_id,
            query,
            limit=limit,
            candidate_limit=candidate_limit,
            context_radius=context_radius,
            exclude_user_id=exclude_user_id,
            exclude_text=exclude_text,
            preferred_user_id=preferred_user_id,
            preferred_limit=preferred_limit,
            preferred_score_multiplier=preferred_score_multiplier,
            preferred_score_bonus=preferred_score_bonus,
            per_user_limit=per_user_limit,
        )

    def _message_neighbors(
        self,
        group_id: int,
        message_id: int,
        *,
        radius: int,
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        return self._message_repository._message_neighbors(group_id, message_id, radius=radius)

    def recent_bot_replies(self, group_id: int, seconds: int) -> list[ChatMessage]:
        return self._message_repository.recent_bot_replies(group_id, seconds)

    def messages_for_mid_summary(
        self,
        group_id: int,
        *,
        keep_recent: int,
        batch_size: int,
        include_bot: bool = True,
    ) -> list[ChatMessage]:
        return self._message_repository.messages_for_mid_summary(
            group_id,
            keep_recent=keep_recent,
            batch_size=batch_size,
            include_bot=include_bot,
        )

    def add_memory_summary(
        self,
        group_id: int,
        messages: list[ChatMessage],
        *,
        summary: str,
        recall_cues: list[str],
    ) -> None:
        return self._summary_repository.add_memory_summary(group_id, messages, summary=summary, recall_cues=recall_cues)

    def advance_memory_summary_cursor(self, group_id: int, last_message_id: int) -> None:
        """Move the mid-memory window forward without writing a summary."""

        return self._summary_repository.advance_memory_summary_cursor(group_id, last_message_id)

    def recent_memory_summaries(self, group_id: int, limit: int) -> list[MemorySummary]:
        return self._summary_repository.recent_memory_summaries(group_id, limit)

    def relevant_memory_summaries(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 80,
    ) -> list[MemorySummary]:
        return self._summary_repository.relevant_memory_summaries(
            group_id,
            query,
            limit=limit,
            candidate_limit=candidate_limit,
        )

    def admin_recent_memory_summaries(
        self,
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        query: str = "",
    ) -> list[MemorySummary]:
        return self._summary_repository.admin_recent_memory_summaries(
            group_id=group_id,
            status=status,
            limit=limit,
            query=query,
        )

    def memory_summary(self, summary_id: int) -> MemorySummary | None:
        return self._summary_repository.memory_summary(summary_id)

    def admin_update_memory_summary(
        self,
        summary_id: int,
        *,
        summary: str,
        recall_cues: list[str],
        status: str = "active",
        locked: bool | None = None,
    ) -> bool:
        return self._summary_repository.admin_update_memory_summary(
            summary_id,
            summary=summary,
            recall_cues=recall_cues,
            status=status,
            locked=locked,
        )

    def admin_set_memory_summary_state(self, summary_id: int, *, action: str) -> bool:
        return self._summary_repository.admin_set_memory_summary_state(summary_id, action=action)

    def admin_add_memory_summary(
        self,
        *,
        group_id: int,
        summary: str,
        recall_cues: list[str],
        locked: bool = True,
    ) -> int:
        return self._summary_repository.admin_add_memory_summary(
            group_id=group_id,
            summary=summary,
            recall_cues=recall_cues,
            locked=locked,
        )

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
        return self._member_repository.member_profiles_for_context(group_id, user_ids, limit=limit)

    def member_impressions_for_context(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        limit: int,
    ) -> list[MemberImpression]:
        return self._member_repository.member_impressions_for_context(group_id, user_ids, limit=limit)

    def recent_member_impressions(self, group_id: int, limit: int) -> list[MemberImpression]:
        return self._member_repository.recent_member_impressions(group_id, limit)

    def active_member_ids_since(
        self,
        group_id: int,
        *,
        since_at: float,
        limit: int,
        min_messages: int,
    ) -> list[int]:
        return self._member_repository.active_member_ids_since(
            group_id,
            since_at=since_at,
            limit=limit,
            min_messages=min_messages,
        )

    def member_messages_between(
        self,
        group_id: int,
        user_id: int,
        *,
        start_at: float,
        end_at: float,
        limit: int,
    ) -> list[ChatMessage]:
        return self._member_repository.member_messages_between(
            group_id,
            user_id,
            start_at=start_at,
            end_at=end_at,
            limit=limit,
        )

    def last_member_profile_summary_at(self, group_id: int, user_id: int) -> float:
        return self._member_repository.last_member_profile_summary_at(group_id, user_id)

    def latest_member_profile_summary(
        self,
        group_id: int,
        user_id: int,
    ) -> MemberProfileSummary | None:
        return self._member_repository.latest_member_profile_summary(group_id, user_id)

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
        return self._member_repository.add_member_profile_summary(
            group_id=group_id,
            user_id=user_id,
            profile_summary=profile_summary,
            interests=interests,
            speaking_style=speaking_style,
            representative_texts=representative_texts,
            start_at=start_at,
            end_at=end_at,
            message_count=message_count,
            keep_per_member=keep_per_member,
        )

    def prune_member_profile_summaries(self, *, keep_per_member: int = 3) -> int:
        return self._member_repository.prune_member_profile_summaries(keep_per_member=keep_per_member)

    def recent_member_profile_summaries(
        self,
        group_id: int,
        user_id: int,
        limit: int,
    ) -> list[MemberProfileSummary]:
        return self._member_repository.recent_member_profile_summaries(group_id, user_id, limit)

    def _update_member_impression(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
        *,
        created_at: float,
    ) -> None:
        return self._member_repository._update_member_impression(
            group_id,
            user_id,
            nickname,
            text,
            created_at=created_at,
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
        return self._message_repository.add_bot_sent_message(
            group_id=group_id,
            message_id=message_id,
            bot_reply=bot_reply,
            trigger_user_id=trigger_user_id,
            trigger_nickname=trigger_nickname,
            trigger_text=trigger_text,
            action=action,
            created_at=created_at,
        )

    def bot_sent_message(self, group_id: int, message_id: int) -> BotSentMessage | None:
        return self._message_repository.bot_sent_message(group_id, message_id)

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
        return self._message_repository.add_recalled_reply_feedback(
            group_id=group_id,
            message_id=message_id,
            bot_reply=bot_reply,
            trigger_user_id=trigger_user_id,
            trigger_nickname=trigger_nickname,
            trigger_text=trigger_text,
            action=action,
            owner_reason=owner_reason,
            scene_summary=scene_summary,
            bad_reply_problem=bad_reply_problem,
            avoid_rule=avoid_rule,
            better_direction=better_direction,
            tags=tags,
            operator_id=operator_id,
            reason_user_id=reason_user_id,
            recalled_at=recalled_at,
            reason_at=reason_at,
        )

    def recent_recalled_reply_feedback(
        self,
        group_id: int,
        limit: int,
    ) -> list[RecalledReplyFeedback]:
        return self._message_repository.recent_recalled_reply_feedback(group_id, limit)

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
        return self._message_repository.add_approved_reply_feedback(
            group_id=group_id,
            candidate_text=candidate_text,
            trigger_user_id=trigger_user_id,
            trigger_nickname=trigger_nickname,
            trigger_text=trigger_text,
            action=action,
            style=style,
            tags=tags,
            operator_id=operator_id,
            created_at=created_at,
        )

    def recent_approved_reply_feedback(
        self,
        group_id: int,
        limit: int,
    ) -> list[ApprovedReplyFeedback]:
        return self._message_repository.recent_approved_reply_feedback(group_id, limit)

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


def _custom_jargon_from_row(row: sqlite3.Row) -> CustomJargonEntry:
    return CustomJargonEntry(
        group_id=int(row["group_id"]),
        term=str(row["term"]),
        explanation=str(row["explanation"]),
        created_by=int(row["created_by"]),
        created_at=float(row["created_at"]),
    )
