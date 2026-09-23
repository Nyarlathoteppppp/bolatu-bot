"""Daily review generation, delivery, and success state."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .memory import MemoryStore


@dataclass(frozen=True)
class DailyReviewPolicy:
    timezone: ZoneInfo
    hour: int
    minute: int
    message_limit: int
    respect_mute: bool


@dataclass(frozen=True)
class DailyReviewDeliveryServices:
    prepare_political_send_texts: Callable[..., Awaitable[tuple[str, str, list[str]]]]
    send_group_message: Callable[..., Awaitable[int | None]]
    notify_owner_political_gag: Callable[..., Awaitable[None]]
    record_bot_sent_message: Callable[..., None]
    record_metric_event: Callable[..., None]
    action_failed_summary: Callable[[ActionFailed], str]
    short_notice_text: Callable[[str, int], str]
    logger: Any


@dataclass(frozen=True)
class DailyReviewServices:
    memory: MemoryStore
    app_config: Any
    get_deepseek_client: Callable[[], Any]
    get_persona: Callable[[str], Any]
    refresh_self_mute_state_if_stale: Callable[..., Awaitable[float]]
    persist_learning: Callable[..., list[int]]
    sanitize_generated_text: Callable[[str], str]
    split_reply_messages: Callable[..., list[str]]
    delivery: DailyReviewDeliveryServices


class DailyReviewService:
    """Own daily review workflow state shared by scheduled and manual sends."""

    def __init__(self) -> None:
        self.send_locks: dict[tuple[int, str], asyncio.Lock] = {}

    def send_lock(self, group_id: int, review_label: str) -> asyncio.Lock:
        return self.send_locks.setdefault((group_id, review_label), asyncio.Lock())

    @staticmethod
    def local_timestamp_for_today(hour: int, minute: int, *, now: float, policy: DailyReviewPolicy) -> float:
        local = datetime.fromtimestamp(now, policy.timezone)
        return local.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()

    def review_window(self, now: float, *, policy: DailyReviewPolicy) -> tuple[float, float, str]:
        end_at = self.local_timestamp_for_today(policy.hour, policy.minute, now=now, policy=policy)
        if now < end_at:
            end_at -= 24 * 60 * 60
        start_at = end_at - 24 * 60 * 60
        local_end = datetime.fromtimestamp(end_at - 1, policy.timezone)
        label = f"{local_end.year:04d}-{local_end.month:02d}-{local_end.day:02d}"
        return start_at, end_at, label

    @staticmethod
    def today_window(now: float, *, policy: DailyReviewPolicy) -> tuple[float, float, str]:
        local_now = datetime.fromtimestamp(now, policy.timezone)
        start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        label = f"{local_now.year:04d}-{local_now.month:02d}-{local_now.day:02d} 今日到现在"
        return start_local.timestamp(), now, label

    @staticmethod
    def target_groups(services: DailyReviewServices) -> tuple[int, ...]:
        if services.app_config.allowed_groups:
            return tuple(sorted(services.app_config.allowed_groups))
        group_ids = [int(group_id) for group_id in services.app_config.groups if str(group_id).isdigit()]
        return tuple(sorted(set(group_ids)))

    @staticmethod
    def group_enabled(
        group_id: int,
        *,
        now: float,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> bool:
        if not services.app_config.group_allowed(group_id):
            return False
        group_cfg = services.app_config.group_config(group_id)
        state = services.memory.group_state(group_id)
        if not bool(group_cfg.get("enabled", True)) or not bool(state["enabled"]):
            return False
        if policy.respect_mute:
            return float(state["muted_until"]) <= now
        return True

    @staticmethod
    def sent_key(group_id: int, review_label: str) -> str:
        return f"daily_review_sent:{group_id}:{review_label}"

    @staticmethod
    def feedback_context(
        group_id: int,
        *,
        start_at: float,
        end_at: float,
        services: DailyReviewServices,
    ) -> str:
        short_notice = services.delivery.short_notice_text
        lines: list[str] = []
        for item in services.memory.recent_recalled_reply_feedback(group_id, 24):
            if not start_at <= item.reason_at < end_at:
                continue
            lines.append(
                f"- 否决：触发={short_notice(item.trigger_text, 70)}；"
                f"问题={short_notice(item.owner_reason or item.avoid_rule, 100)}"
            )
        for item in services.memory.recent_approved_reply_feedback(group_id, 24):
            if not start_at <= item.created_at < end_at:
                continue
            lines.append(
                f"- 优质：action={item.action}；style={short_notice(item.style, 80)}；"
                f"触发={short_notice(item.trigger_text, 70)}"
            )
        return "\n".join(lines[:20]) or "（当天无审批反馈）"

    async def send_due_reviews(
        self,
        bot: Bot,
        *,
        now: float | None,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> bool:
        delivery = services.delivery
        if services.get_deepseek_client() is None:
            delivery.logger.warning("qq_social_agent daily review skipped: deepseek_client_not_ready")
            delivery.record_metric_event("daily_review", stage="check", action="skipped", reason="deepseek_client_not_ready")
            return True
        target_groups = self.target_groups(services)
        if not target_groups:
            delivery.logger.info("qq_social_agent daily review skipped: no_target_groups")
            delivery.record_metric_event("daily_review", stage="check", action="skipped", reason="no_target_groups")
            return False
        current = time.time() if now is None else now
        start_at, end_at, review_label = self.review_window(current, policy=policy)
        has_pending = False
        for group_id in target_groups:
            if not self.group_enabled(group_id, now=current, policy=policy, services=services):
                delivery.logger.info(f"qq_social_agent daily review skipped: group={group_id} disabled_or_muted")
                delivery.record_metric_event(
                    "daily_review", group_id=group_id, stage="check", action="skipped", reason="disabled_or_muted"
                )
                continue
            sent_key = self.sent_key(group_id, review_label)
            async with self.send_lock(group_id, review_label):
                if services.memory.app_kv_get(sent_key) == "sent":
                    delivery.logger.info(
                        f"qq_social_agent daily review skipped: group={group_id} already_sent date={review_label}"
                    )
                    continue
                success = await self.send_review_for_group(
                    bot,
                    group_id=group_id,
                    start_at=start_at,
                    end_at=end_at,
                    review_label=review_label,
                    sent_key=sent_key,
                    mark_sent=True,
                    source="scheduled",
                    trigger_label="定时复盘",
                    policy=policy,
                    services=services,
                )
            if not success:
                has_pending = True
        return has_pending

    async def send_review_for_group(
        self,
        bot: Bot,
        *,
        group_id: int,
        start_at: float,
        end_at: float,
        review_label: str,
        sent_key: str | None,
        mark_sent: bool,
        source: str,
        trigger_label: str,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> bool:
        memory = services.memory
        delivery = services.delivery
        state = memory.group_state(group_id)
        muted_until = await services.refresh_self_mute_state_if_stale(bot, group_id, float(state["muted_until"] or 0))
        if muted_until > time.time():
            delivery.logger.info(f"qq_social_agent daily review skipped while self muted: group={group_id} until={muted_until}")
            delivery.record_metric_event(
                "daily_review",
                group_id=group_id,
                stage="check",
                action="skipped",
                review_label=review_label,
                source=source,
                reason="self_muted",
                muted_until=muted_until,
            )
            return False
        persona_id = str(
            state["persona"]
            or services.app_config.group_config(group_id).get("persona")
            or services.app_config.default_persona
        )
        persona = services.get_persona(persona_id)
        messages = memory.messages_between(
            group_id,
            start_at=start_at,
            end_at=end_at,
            limit=policy.message_limit,
        )
        review_draft = None
        try:
            client = services.get_deepseek_client()
            review_draft = (
                await client.daily_review_draft(
                    persona=persona,
                    messages=messages,
                    chat_label=f"QQ 群 {group_id}",
                    today_label=review_label,
                    feedback_context=self.feedback_context(
                        group_id,
                        start_at=start_at,
                        end_at=end_at,
                        services=services,
                    ),
                )
                if client is not None
                else None
            )
            review = review_draft.public_reply if review_draft is not None else ""
        except Exception as exc:
            delivery.logger.warning(f"qq_social_agent daily review generation failed: group={group_id} error={exc}")
            delivery.record_metric_event(
                "daily_review",
                group_id=group_id,
                stage="generation",
                action="failed",
                review_label=review_label,
                source=source,
                reason=delivery.short_notice_text(str(exc), 200),
            )
            return False
        if not review:
            review = "今天群里没怎么留给我发挥，我先记一笔：大家还是挺能聊的。"
        review = services.sanitize_generated_text(review)
        parts = services.split_reply_messages(review, max_messages=3)
        if not parts:
            delivery.record_metric_event(
                "daily_review",
                group_id=group_id,
                stage="send",
                action="failed",
                review_label=review_label,
                source=source,
                reason="empty_after_sanitize",
            )
            return False
        for index, part in enumerate(parts):
            try:
                public_text, memory_text, gag = await delivery.prepare_political_send_texts(part, context=review)
                message_id = await delivery.send_group_message(bot, group_id, Message(public_text))
                if gag:
                    await delivery.notify_owner_political_gag(
                        original=part,
                        public=public_text,
                        hits=gag,
                        group_id=group_id,
                        source="daily_review",
                    )
                delivery.record_bot_sent_message(
                    group_id=group_id,
                    message_id=message_id,
                    bot_reply=memory_text,
                    trigger_user_id=0,
                    trigger_nickname="每日复盘",
                    trigger_text=f"{review_label} {trigger_label}",
                    action="daily_review",
                )
                memory.add_message(
                    group_id,
                    int(getattr(bot, "self_id", 0) or 0),
                    persona.name,
                    memory_text,
                    is_bot=True,
                    source_message_id=message_id,
                    source_kind="live",
                )
            except ActionFailed as exc:
                delivery.logger.warning(
                    "qq_social_agent failed sending daily review: "
                    f"group={group_id} {delivery.action_failed_summary(exc)}"
                )
                delivery.record_metric_event(
                    "daily_review",
                    group_id=group_id,
                    stage="send",
                    action="failed",
                    review_label=review_label,
                    source=source,
                    reason=delivery.short_notice_text(delivery.action_failed_summary(exc), 200),
                )
                return False
            if index < len(parts) - 1:
                await asyncio.sleep(0.9)
        if mark_sent and sent_key:
            memory.app_kv_set(sent_key, "sent")
        if review_draft is not None:
            try:
                learned_atom_ids = services.persist_learning(
                    memory,
                    group_id=group_id,
                    review_label=review_label,
                    draft=review_draft,
                    messages=messages,
                )
                delivery.record_metric_event(
                    "daily_review_learning",
                    group_id=group_id,
                    stage="memory",
                    action="persisted",
                    atom_count=len(learned_atom_ids),
                    event_count=len(review_draft.events),
                    member_change_count=len(review_draft.member_changes),
                    jargon_count=len(review_draft.jargon_candidates),
                    feedback_lesson_count=len(review_draft.feedback_lessons),
                    style_observation_count=len(review_draft.style_observations),
                )
            except Exception as exc:
                delivery.logger.warning(
                    "qq_social_agent daily review learning persist failed: "
                    f"group={group_id} date={review_label} error={exc}"
                )
        delivery.record_metric_event(
            "daily_review",
            group_id=group_id,
            stage="send",
            action="sent",
            review_label=review_label,
            source=source,
            message_count=len(messages),
            part_count=len(parts),
            mark_sent=mark_sent,
        )
        delivery.logger.info(
            "qq_social_agent daily review sent: "
            f"group={group_id} date={review_label} messages={len(messages)} parts={len(parts)}"
        )
        return True

    async def send_manual_reviews(
        self,
        bot: Bot,
        *,
        mode: str,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> tuple[int, int]:
        if services.get_deepseek_client() is None:
            services.delivery.logger.warning("qq_social_agent manual daily review skipped: deepseek_client_not_ready")
            return 0, 0
        now = time.time()
        if mode in {"due", "补发", "scheduled", "midnight", "午夜"}:
            start_at, end_at, review_label = self.review_window(now, policy=policy)
            mark_sent = True
            source = "manual_due"
            trigger_label = "补发定时复盘"
        else:
            start_at, end_at, review_label = self.today_window(now, policy=policy)
            mark_sent = False
            source = "manual_today"
            trigger_label = "即时复盘"
        sent_count = 0
        total_count = 0
        for group_id in self.target_groups(services):
            if not self.group_enabled(group_id, now=now, policy=policy, services=services):
                continue
            total_count += 1
            sent_key = self.sent_key(group_id, review_label) if mark_sent else None
            if mark_sent:
                async with self.send_lock(group_id, review_label):
                    if sent_key and services.memory.app_kv_get(sent_key) == "sent":
                        continue
                    if await self.send_review_for_group(
                        bot,
                        group_id=group_id,
                        start_at=start_at,
                        end_at=end_at,
                        review_label=review_label,
                        sent_key=sent_key,
                        mark_sent=mark_sent,
                        source=source,
                        trigger_label=trigger_label,
                        policy=policy,
                        services=services,
                    ):
                        sent_count += 1
            elif await self.send_review_for_group(
                bot,
                group_id=group_id,
                start_at=start_at,
                end_at=end_at,
                review_label=review_label,
                sent_key=sent_key,
                mark_sent=mark_sent,
                source=source,
                trigger_label=trigger_label,
                policy=policy,
                services=services,
            ):
                sent_count += 1
        return sent_count, total_count
