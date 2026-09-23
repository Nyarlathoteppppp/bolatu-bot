"""Background memory, style, and member-profile maintenance."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from nonebot import logger

from .memory import ChatMessage, MemberProfileSummary, MemoryStore
from .memory_learning import persist_mid_memory_learning, persist_private_mid_memory


@dataclass(frozen=True)
class MemoryMaintenancePolicy:
    group_context_limit: int
    private_context_limit: int
    private_chat_offset: int
    mid_memory_batch_size: int
    mid_memory_min_batch: int
    mid_memory_retry_interval_seconds: int
    mid_memory_empty_skip_streak: int
    style_learn_interval_seconds: int
    style_learn_message_limit: int
    style_learn_candidate_limit: int
    style_learn_per_user_limit: int
    style_learn_min_messages: int
    member_profile_interval_seconds: int
    member_profile_lookback_seconds: int
    member_profile_active_limit: int
    member_profile_min_messages: int
    member_profile_message_limit: int
    member_profile_min_chars: int


class MemoryMaintenanceService:
    """Run one memory-maintenance sweep without owning chat event handling."""

    def __init__(
        self,
        *,
        memory_provider: Callable[[], MemoryStore],
        client_provider: Callable[[], Any | None],
        policy_provider: Callable[[], MemoryMaintenancePolicy],
        record_metric_event: Callable[..., None],
        member_label: Callable[[int, str], str],
        useful_style_rule: Callable[[str, str, str], bool],
    ) -> None:
        self.memory_provider = memory_provider
        self.client_provider = client_provider
        self.policy_provider = policy_provider
        self.record_metric_event = record_metric_event
        self.member_label = member_label
        self.useful_style_rule = useful_style_rule

        # These maps are the single source of truth. plugin.py exposes aliases
        # for compatibility with existing diagnostics and tests.
        self.last_mid_memory_attempt: dict[int, float] = {}
        self.mid_memory_empty_streak: dict[int, int] = {}
        self.last_style_learn_attempt: dict[int, float] = {}

    def note_empty_mid_memory(self, group_id: int, summary_messages: list[ChatMessage]) -> str:
        policy = self.policy_provider()
        memory = self.memory_provider()
        streak = self.mid_memory_empty_streak.get(group_id, 0) + 1
        self.mid_memory_empty_streak[group_id] = streak
        if streak < policy.mid_memory_empty_skip_streak or not summary_messages:
            return "empty_summary"
        memory.advance_memory_summary_cursor(group_id, summary_messages[-1].id)
        self.mid_memory_empty_streak[group_id] = 0
        logger.warning(
            "qq_social_agent mid memory skipped empty window: "
            f"group={group_id} messages={len(summary_messages)} end_id={summary_messages[-1].id}"
        )
        return "skipped_empty_window"

    async def maintain_group(self, group_id: int) -> None:
        client = self.client_provider()
        if client is None:
            return

        memory = self.memory_provider()
        policy = self.policy_provider()
        expired_atoms = memory.expire_due_memory_atoms(group_id=group_id)
        if expired_atoms:
            logger.info(
                f"qq_social_agent expired due memory atoms: group={group_id} count={expired_atoms}"
            )

        mid_messages = memory.messages_for_mid_summary(
            group_id,
            keep_recent=(
                policy.private_context_limit
                if group_id >= policy.private_chat_offset
                else policy.group_context_limit
            ),
            batch_size=policy.mid_memory_batch_size,
            include_bot=False,
        )
        empty_streak = self.mid_memory_empty_streak.get(group_id, 0)
        retry_wait = policy.mid_memory_retry_interval_seconds * (2 ** min(empty_streak, 3))
        if (
            len(mid_messages) >= policy.mid_memory_min_batch
            and time.time() - self.last_mid_memory_attempt.get(group_id, 0.0)
            >= retry_wait
        ):
            self.last_mid_memory_attempt[group_id] = time.time()
            try:
                summary_messages = [msg for msg in mid_messages if not msg.is_bot]
                draft = None
                if len(summary_messages) >= policy.mid_memory_min_batch:
                    draft = await client.summarize_mid_memory(
                        messages=summary_messages,
                        chat_label="QQ 私聊" if group_id >= policy.private_chat_offset else "QQ 群聊",
                    )
                if draft and draft.summary:
                    memory.add_memory_summary(
                        group_id,
                        summary_messages,
                        summary=draft.summary,
                        recall_cues=list(draft.recall_cues),
                    )
                    persist_memory = (
                        persist_private_mid_memory
                        if group_id >= policy.private_chat_offset
                        else persist_mid_memory_learning
                    )
                    learned_atom_ids = persist_memory(
                        memory,
                        group_id=group_id,
                        draft=draft,
                        messages=summary_messages,
                    )
                    logger.info(
                        "qq_social_agent mid memory summarized: "
                        f"group={group_id} messages={len(mid_messages)} cues={len(draft.recall_cues)} "
                        f"atoms={len(learned_atom_ids)}"
                    )
                    self.mid_memory_empty_streak[group_id] = 0
                    self.record_metric_event(
                        "mid_memory_learning",
                        group_id=group_id,
                        stage="memory",
                        action="persisted",
                        atom_count=len(learned_atom_ids),
                        fact_count=len(draft.facts),
                        member_delta_count=len(draft.member_deltas),
                        jargon_count=len(draft.jargon_candidates),
                        open_thread_count=len(draft.open_threads),
                    )
                else:
                    skip_action = self.note_empty_mid_memory(group_id, summary_messages)
                    logger.warning(
                        "qq_social_agent mid memory returned empty summary: "
                        f"group={group_id} messages={len(summary_messages)} action={skip_action}"
                    )
                    self.record_metric_event(
                        "mid_memory_learning",
                        group_id=group_id,
                        stage="memory",
                        action=skip_action,
                        message_count=len(summary_messages),
                    )
            except Exception as exc:
                logger.warning(f"qq_social_agent mid memory skipped: group={group_id} error={exc}")

        if group_id >= policy.private_chat_offset:
            # Direct-chat messages retain rolling summaries and explicit memory,
            # but never teach the shared group style or auto-create a user profile.
            return

        last_attempt = max(
            memory.last_style_rule_at(group_id),
            self.last_style_learn_attempt.get(group_id, 0.0),
        )
        if time.time() - last_attempt >= policy.style_learn_interval_seconds:
            style_messages = memory.messages_for_style_learning(
                group_id,
                limit=policy.style_learn_candidate_limit,
            )
            style_messages = balanced_style_learning_messages(
                style_messages,
                max_messages=policy.style_learn_message_limit,
                max_per_user=policy.style_learn_per_user_limit,
            )
            if len(style_messages) >= policy.style_learn_min_messages:
                self.last_style_learn_attempt[group_id] = time.time()
                try:
                    rules = await client.learn_style_rules(
                        messages=style_messages,
                        chat_label="QQ 群聊",
                    )
                    useful_rules = [
                        rule
                        for rule in rules
                        if self.useful_style_rule(rule.situation, rule.style, rule.source_text)
                    ]
                    style_stats = memory.add_style_rules(
                        group_id,
                        [
                            (
                                rule.situation,
                                rule.style,
                                rule.source_text,
                                rule.source_user_ids,
                                rule.source_message_ids,
                            )
                            for rule in useful_rules
                        ],
                    )
                    if useful_rules:
                        logger.info(
                            "qq_social_agent style rules learned: "
                            f"group={group_id} rules={len(useful_rules)} "
                            f"new={style_stats.get('new', 0)} "
                            f"merged={style_stats.get('merged', 0)} "
                            f"expired={style_stats.get('expired', 0)} "
                            f"skipped={style_stats.get('skipped', 0)}"
                        )
                except Exception as exc:
                    logger.warning(
                        f"qq_social_agent style learning skipped: group={group_id} error={exc}"
                    )

        # Profile learning is intentionally last and capped per sweep. Reply-time
        # work and the mid-memory backlog get priority over many sequential LLM calls.
        await self.maintain_member_profile_summaries(group_id, max_updates=1)

    async def maintain_member_profile_summaries(
        self,
        group_id: int,
        *,
        force: bool = False,
        max_updates: int | None = None,
    ) -> None:
        client = self.client_provider()
        if client is None:
            return
        memory = self.memory_provider()
        policy = self.policy_provider()
        now = time.time()
        start_at = now - policy.member_profile_lookback_seconds
        active_user_ids = memory.active_member_ids_since(
            group_id,
            since_at=start_at,
            limit=policy.member_profile_active_limit,
            min_messages=policy.member_profile_min_messages,
        )
        if not active_user_ids:
            return
        updated = 0
        for user_id in active_user_ids:
            previous = memory.latest_member_profile_summary(group_id, user_id)
            last_summary_at = previous.created_at if previous is not None else 0.0
            if not force and last_summary_at and now - last_summary_at < policy.member_profile_interval_seconds:
                continue
            window_start = start_at
            if previous is not None:
                window_start = max(start_at, float(previous.end_at or previous.created_at))
            messages = memory.member_messages_between(
                group_id,
                user_id,
                start_at=window_start,
                end_at=now + 1,
                limit=policy.member_profile_message_limit,
            )
            messages = member_profile_learning_messages(messages)
            if len(messages) < policy.member_profile_min_messages:
                continue
            if sum(len((item.text or "").strip()) for item in messages) < policy.member_profile_min_chars:
                continue
            label = self.member_label(user_id, messages[-1].nickname)
            previous_text = member_profile_previous_text(previous)
            try:
                draft = await client.summarize_member_profile(
                    messages=messages,
                    member_label=label,
                    chat_label="QQ 群聊",
                    previous_summary=previous_text,
                )
            except Exception as exc:
                logger.warning(
                    "qq_social_agent member profile summary skipped: "
                    f"group={group_id} user={user_id} error={exc}"
                )
                continue
            if not draft.summary:
                continue
            memory.add_member_profile_summary(
                group_id=group_id,
                user_id=user_id,
                profile_summary=draft.summary,
                interests=list(draft.interests),
                speaking_style=draft.speaking_style,
                representative_texts=list(draft.representative_texts),
                start_at=messages[0].created_at,
                end_at=messages[-1].created_at,
                message_count=len(messages),
            )
            logger.info(
                "qq_social_agent member profile summarized: "
                f"group={group_id} user={user_id} messages={len(messages)} "
                f"incremental={previous is not None}"
            )
            updated += 1
            if max_updates is not None and updated >= max(1, max_updates):
                break


def balanced_style_learning_messages(
    messages: list[ChatMessage],
    *,
    max_messages: int = 40,
    max_per_user: int = 5,
) -> list[ChatMessage]:
    selected: list[ChatMessage] = []
    user_counts: dict[int, int] = {}
    seen_texts: set[tuple[int, str]] = set()
    for message in reversed(messages):
        text_key = (message.user_id, re.sub(r"\s+", "", message.text).casefold())
        if text_key in seen_texts or user_counts.get(message.user_id, 0) >= max_per_user:
            continue
        seen_texts.add(text_key)
        user_counts[message.user_id] = user_counts.get(message.user_id, 0) + 1
        selected.append(message)
        if len(selected) >= max_messages:
            break
    balanced = sorted(selected, key=lambda item: (item.created_at, item.id))
    logger.info(
        "qq_social_agent style learning balanced sample: "
        f"candidates={len(messages)} selected={len(balanced)} users={len(user_counts)} "
        f"max_per_user={max_per_user}"
    )
    return balanced


def member_profile_learning_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    selected: list[ChatMessage] = []
    seen: set[str] = set()
    for message in messages:
        text = str(message.text or "").strip()
        if len(text) < 8:
            continue
        substance = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        if len(substance) < 4:
            continue
        key = re.sub(r"\s+", "", text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(message)
    return selected


def member_profile_previous_text(previous: MemberProfileSummary | None) -> str:
    if previous is None:
        return ""
    lines = [previous.profile_summary.strip()[:240]]
    if previous.interests:
        lines.append("兴趣：" + "、".join(previous.interests[:5]))
    if previous.speaking_style.strip():
        lines.append("说话方式：" + previous.speaking_style.strip()[:80])
    return "\n".join(line for line in lines if line)


def is_useful_style_rule(situation: str, style: str, source_text: str = "") -> bool:
    text = f"{situation} {style} {source_text}".strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    low_value_phrases = {
        "是的",
        "是这样的",
        "确实",
        "还好吧",
        "太典了",
        "绷不住了",
        "闹麻了",
        "赢麻了",
        "差不多得了",
        "乐死了",
        "开宰",
        "886",
        "牛逼",
        "看哭了",
        "这么先进",
        "算了",
        "不赖",
        "一般",
        "厚米",
        "啊",
        "有",
        "好",
        "回国",
        "倒",
        "魔了",
        "吓哭了",
    }
    if compact in low_value_phrases:
        return False
    if any(compact == phrase for phrase in low_value_phrases):
        return False
    style_compact = re.sub(r"\s+", "", style)
    source_compact = re.sub(r"\s+", "", source_text)
    if style_compact in low_value_phrases or source_compact in low_value_phrases:
        return False
    if source_compact and len(source_compact) <= 4:
        return False
    if source_compact.startswith("[图片") or "[图片OCR" in source_text:
        return False
    if source_compact.startswith("[长消息") and "摘要" in source_compact[:12]:
        return False
    if "原消息内容未知" in source_text:
        return False
    if len(style_compact) <= 3 and style_compact in {"赞同", "附和", "吐槽"}:
        return False
    if _looks_like_literal_style_rule(style):
        return False
    if source_compact and _has_long_common_substring(style_compact, source_compact, min_len=6):
        return False
    return True


def _looks_like_literal_style_rule(style: str) -> bool:
    stripped = style.strip()
    compact = re.sub(r"\s+", "", stripped)
    if not compact:
        return True
    literal_markers = (
        "说“",
        '说"',
        "用“",
        '用"',
        "短句接“",
        "直接说“",
        "表达“",
        "接“",
    )
    if any(marker in compact for marker in literal_markers):
        return True
    if compact.startswith(("说", "发")) and len(compact) <= 18:
        return True
    if compact in {"重复对方原句", "复读对方原句"}:
        return True
    if re.fullmatch(r"发?[^\w\u4e00-\u9fff]{1,8}", compact):
        return True
    quote_count = compact.count("“") + compact.count("”") + compact.count('"')
    return quote_count > 0 and len(compact) <= 28


def _has_long_common_substring(a: str, b: str, *, min_len: int) -> bool:
    if len(a) < min_len or len(b) < min_len:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    max_size = min(len(shorter), 24)
    for size in range(max_size, min_len - 1, -1):
        for start in range(0, len(shorter) - size + 1):
            if shorter[start : start + size] in longer:
                return True
    return False
