"""Daily review generation, delivery, and success state."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .memory import MemoryStore
from .memory_models import ChatMessage


_PENDING_REVIEW_BATCHES_KEY = "daily_review_batch_pending_keys"


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
    review_batch_enabled: Callable[[], bool] = lambda: False
    review_batch_service: Any | None = None
    review_sync_model: Callable[[], Any | None] = lambda: None


class DailyReviewService:
    """Own daily review workflow state shared by scheduled and manual sends."""

    def __init__(self) -> None:
        self.send_locks: dict[tuple[int, str], asyncio.Lock] = {}

    def send_lock(self, group_id: int, review_label: str) -> asyncio.Lock:
        return self.send_locks.setdefault((group_id, review_label), asyncio.Lock())

    @staticmethod
    def batch_key(group_id: int, review_label: str) -> str:
        return f"daily_review:{group_id}:{review_label}"

    @classmethod
    def batch_state_key(cls, group_id: int, review_label: str) -> str:
        return f"daily_review_batch_state:{cls.batch_key(group_id, review_label)}"

    @staticmethod
    def _pending_batch_keys(services: DailyReviewServices) -> list[str]:
        raw = services.memory.app_kv_get(_PENDING_REVIEW_BATCHES_KEY)
        if not raw:
            return []
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            services.delivery.logger.warning("qq_social_agent daily review batch index is invalid")
            return []
        return list(dict.fromkeys(str(value) for value in values if isinstance(value, str))) if isinstance(values, list) else []

    @classmethod
    def _save_batch_state(
        cls,
        group_id: int,
        review_label: str,
        state: dict[str, Any],
        *,
        services: DailyReviewServices,
    ) -> None:
        state_key = cls.batch_state_key(group_id, review_label)
        services.memory.app_kv_set(state_key, json.dumps(state, ensure_ascii=False))
        keys = cls._pending_batch_keys(services)
        if state_key not in keys:
            keys.append(state_key)
        services.memory.app_kv_set(_PENDING_REVIEW_BATCHES_KEY, json.dumps(keys, ensure_ascii=False))

    @classmethod
    def _delete_batch_state(
        cls,
        state_key: str,
        *,
        services: DailyReviewServices,
    ) -> None:
        services.memory.app_kv_set(state_key, "")
        keys = [key for key in cls._pending_batch_keys(services) if key != state_key]
        services.memory.app_kv_set(_PENDING_REVIEW_BATCHES_KEY, json.dumps(keys, ensure_ascii=False))

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {
                "group_id": item.group_id,
                "user_id": item.user_id,
                "nickname": item.nickname,
                "text": item.text,
                "is_bot": item.is_bot,
                "created_at": item.created_at,
                "id": item.id,
                "source_message_id": item.source_message_id,
                "session_id": item.session_id,
                "message_segments_json": item.message_segments_json,
                "raw_message_json": item.raw_message_json,
                "sender_json": item.sender_json,
            }
            for item in messages
        ]

    @staticmethod
    def _restore_messages(values: list[dict[str, Any]]) -> list[ChatMessage]:
        return [ChatMessage(**value) for value in values]

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

    def _review_request(
        self,
        client: Any,
        *,
        persona: Any,
        messages: list[ChatMessage],
        group_id: int,
        review_label: str,
        start_at: float,
        end_at: float,
        services: DailyReviewServices,
    ) -> dict[str, Any] | None:
        build_request = getattr(client, "build_daily_review_request", None)
        if not callable(build_request):
            return None
        return build_request(
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

    async def _submit_scheduled_batch_review(
        self,
        *,
        group_id: int,
        start_at: float,
        end_at: float,
        review_label: str,
        sent_key: str,
        trigger_label: str,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> bool:
        """Persist intent before submitting so a restart can safely replay the same idempotency key."""
        delivery = services.delivery
        client = services.get_deepseek_client()
        batch_service = services.review_batch_service
        state_key = self.batch_state_key(group_id, review_label)
        existing_raw = services.memory.app_kv_get(state_key)
        existing: dict[str, Any] | None = None
        if existing_raw:
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, dict):
                    existing = parsed
            except (TypeError, ValueError):
                delivery.logger.warning(
                    f"qq_social_agent daily review batch state invalid: group={group_id} date={review_label}"
                )

        if existing and existing.get("status") != "submitting":
            return True
        if batch_service is None or client is None:
            delivery.logger.warning(
                f"qq_social_agent daily review batch unavailable: group={group_id} client_or_transport_not_ready"
            )
            delivery.record_metric_event(
                "daily_review",
                group_id=group_id,
                stage="batch_submit",
                action="failed",
                review_label=review_label,
                reason="client_or_transport_not_ready",
            )
            return False

        if existing is None:
            state = services.memory.group_state(group_id)
            persona_id = str(
                state["persona"]
                or services.app_config.group_config(group_id).get("persona")
                or services.app_config.default_persona
            )
            persona = services.get_persona(persona_id)
            messages = services.memory.messages_between(
                group_id,
                start_at=start_at,
                end_at=end_at,
                limit=policy.message_limit,
            )
            try:
                request = self._review_request(
                    client,
                    persona=persona,
                    messages=messages,
                    group_id=group_id,
                    review_label=review_label,
                    start_at=start_at,
                    end_at=end_at,
                    services=services,
                )
            except Exception as exc:
                delivery.logger.warning(
                    f"qq_social_agent daily review batch request build failed: group={group_id} error={exc}"
                )
                delivery.record_metric_event(
                    "daily_review",
                    group_id=group_id,
                    stage="batch_submit",
                    action="failed",
                    review_label=review_label,
                    reason=delivery.short_notice_text(str(exc), 200),
                )
                return False
            if request is None:
                delivery.logger.warning(
                    f"qq_social_agent daily review batch request builder missing: group={group_id}"
                )
                delivery.record_metric_event(
                    "daily_review",
                    group_id=group_id,
                    stage="batch_submit",
                    action="failed",
                    review_label=review_label,
                    reason="request_builder_missing",
                )
                return False
            state = services.memory.group_state(group_id)
            persona_id = str(
                state["persona"]
                or services.app_config.group_config(group_id).get("persona")
                or services.app_config.default_persona
            )
            existing = {
                "status": "submitting",
                "group_id": group_id,
                "start_at": start_at,
                "end_at": end_at,
                "review_label": review_label,
                "sent_key": sent_key,
                "source": "scheduled",
                "trigger_label": trigger_label,
                "persona_id": persona_id,
                "messages": self._serialize_messages(messages),
                "body": request,
            }
            self._save_batch_state(group_id, review_label, existing, services=services)

        stable_key = self.batch_key(group_id, review_label)
        try:
            batch_id = await batch_service.submit(stable_key, existing["body"], task="daily_review")
        except Exception as exc:
            delivery.logger.warning(
                f"qq_social_agent daily review batch submit failed: group={group_id} error={exc}"
            )
            delivery.record_metric_event(
                "daily_review",
                group_id=group_id,
                stage="batch_submit",
                action="failed",
                review_label=review_label,
                reason=delivery.short_notice_text(str(exc), 200),
            )
            existing["status"] = "fallback"
            existing["failure_reason"] = str(exc)
            self._save_batch_state(group_id, review_label, existing, services=services)
            return True
        existing["status"] = "submitted"
        existing["batch_id"] = str(batch_id)
        self._save_batch_state(group_id, review_label, existing, services=services)
        delivery.record_metric_event(
            "daily_review",
            group_id=group_id,
            stage="batch_submit",
            action="submitted",
            review_label=review_label,
            batch_id=str(batch_id),
        )
        delivery.logger.info(
            f"qq_social_agent daily review batch submitted: group={group_id} date={review_label} batch={batch_id}"
        )
        return True

    async def poll_pending_reviews(
        self,
        bot: Bot,
        *,
        now: float | None,
        policy: DailyReviewPolicy,
        services: DailyReviewServices,
    ) -> bool:
        """Poll and resume persistent review jobs independently of the daily catch-up window."""
        batch_service = services.review_batch_service
        if batch_service is None:
            return False
        delivery = services.delivery
        current = time.time() if now is None else now
        has_pending = False
        for state_key in self._pending_batch_keys(services):
            state_raw = services.memory.app_kv_get(state_key)
            if not state_raw:
                self._delete_batch_state(state_key, services=services)
                continue
            try:
                state = json.loads(state_raw)
                if not isinstance(state, dict):
                    raise ValueError("state is not an object")
            except (TypeError, ValueError):
                delivery.logger.warning(f"qq_social_agent daily review batch state invalid: key={state_key}")
                self._delete_batch_state(state_key, services=services)
                continue

            group_id = int(state["group_id"])
            review_label = str(state["review_label"])
            batch_key = self.batch_key(group_id, review_label)
            async with self.send_lock(group_id, review_label):
                if state.get("status") == "submitting":
                    try:
                        batch_id = await batch_service.submit(batch_key, state["body"], task="daily_review")
                    except Exception as exc:
                        delivery.logger.warning(
                            f"qq_social_agent daily review batch submit replay failed: group={group_id} error={exc}"
                        )
                        state["status"] = "fallback"
                        state["failure_reason"] = str(exc)
                    else:
                        state["status"] = "submitted"
                        state["batch_id"] = str(batch_id)
                    self._save_batch_state(group_id, review_label, state, services=services)

                if state.get("status") == "submitted":
                    try:
                        result = await batch_service.poll(batch_key)
                    except KeyError:
                        try:
                            batch_id = await batch_service.submit(batch_key, state["body"], task="daily_review")
                        except Exception as exc:
                            delivery.logger.warning(
                                f"qq_social_agent daily review batch resubmit failed: group={group_id} error={exc}"
                            )
                            has_pending = True
                            continue
                        state["batch_id"] = str(batch_id)
                        self._save_batch_state(group_id, review_label, state, services=services)
                        has_pending = True
                        continue
                    except Exception as exc:
                        delivery.logger.warning(
                            f"qq_social_agent daily review batch poll failed: group={group_id} error={exc}"
                        )
                        has_pending = True
                        continue
                    status = str(getattr(result, "status", "unknown")).lower()
                    if status in {"pending", "queued", "validating", "in_progress", "processing"}:
                        has_pending = True
                        continue
                    if status == "submission_uncertain":
                        state["status"] = "fallback"
                        state["failure_reason"] = "batch submission uncertain"
                        self._save_batch_state(group_id, review_label, state, services=services)
                    if status not in {"completed", "complete", "succeeded", "success"}:
                        error = str(getattr(result, "error", "") or status)
                        delivery.logger.warning(
                            f"qq_social_agent daily review batch terminated: group={group_id} status={status} error={error}"
                        )
                        delivery.record_metric_event(
                            "daily_review",
                            group_id=group_id,
                            stage="batch_result",
                            action="failed",
                            review_label=review_label,
                            reason=delivery.short_notice_text(error, 200),
                        )
                        state["status"] = "fallback"
                        state["failure_reason"] = error
                        self._save_batch_state(group_id, review_label, state, services=services)
                    else:
                        content = getattr(result, "content", None)
                        if content is None:
                            content = ""
                        state["status"] = "completed"
                        state["content"] = str(content)
                        self._save_batch_state(group_id, review_label, state, services=services)

                if services.memory.app_kv_get(str(state.get("sent_key", ""))) == "sent":
                    batch_service.acknowledge(batch_key)
                    self._delete_batch_state(state_key, services=services)
                    continue

                messages = self._restore_messages(state.get("messages", []))
                if state.get("status") == "fallback":
                    if not self.group_enabled(group_id, now=current, policy=policy, services=services):
                        has_pending = True
                        continue
                    success = await self.send_review_for_group(
                        bot,
                        group_id=group_id,
                        start_at=float(state["start_at"]),
                        end_at=float(state["end_at"]),
                        review_label=review_label,
                        sent_key=str(state["sent_key"]),
                        mark_sent=True,
                        source="scheduled_batch_fallback",
                        trigger_label="定时复盘",
                        policy=policy,
                        services=services,
                        prepared_messages=messages,
                        prepared_persona_id=str(state.get("persona_id", "")) or None,
                    )
                    if success:
                        batch_service.acknowledge(batch_key)
                        self._delete_batch_state(state_key, services=services)
                    else:
                        has_pending = True
                    continue

                if state.get("status") != "completed":
                    continue

                client = services.get_deepseek_client()
                parse_response = getattr(client, "parse_daily_review_response", None) if client is not None else None
                if not callable(parse_response):
                    has_pending = True
                    continue
                try:
                    review_draft = parse_response(
                        state.get("content", ""),
                        messages=messages[-140:],
                        max_chars=520,
                    )
                except Exception as exc:
                    delivery.logger.warning(
                        f"qq_social_agent daily review batch parse failed: group={group_id} error={exc}"
                    )
                    delivery.record_metric_event(
                        "daily_review",
                        group_id=group_id,
                        stage="batch_result",
                        action="parse_failed",
                        review_label=review_label,
                        reason=delivery.short_notice_text(str(exc), 200),
                    )
                    state["status"] = "fallback"
                    state["failure_reason"] = f"parse failed: {exc}"
                    self._save_batch_state(group_id, review_label, state, services=services)
                    has_pending = True
                    continue

                if not self.group_enabled(group_id, now=current, policy=policy, services=services):
                    has_pending = True
                    continue
                success = await self.send_review_for_group(
                    bot,
                    group_id=group_id,
                    start_at=float(state["start_at"]),
                    end_at=float(state["end_at"]),
                    review_label=review_label,
                    sent_key=str(state["sent_key"]),
                    mark_sent=True,
                    source=str(state.get("source", "scheduled")),
                    trigger_label=str(state.get("trigger_label", "定时复盘")),
                    policy=policy,
                    services=services,
                    prepared_draft=review_draft,
                    prepared_messages=messages,
                    prepared_persona_id=str(state.get("persona_id", "")) or None,
                )
                if success:
                    batch_service.acknowledge(batch_key)
                    self._delete_batch_state(state_key, services=services)
                else:
                    has_pending = True
        return has_pending

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
                if services.review_batch_enabled():
                    success = await self._submit_scheduled_batch_review(
                        group_id=group_id,
                        start_at=start_at,
                        end_at=end_at,
                        review_label=review_label,
                        sent_key=sent_key,
                        trigger_label="定时复盘",
                        policy=policy,
                        services=services,
                    )
                else:
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
        prepared_draft: Any | None = None,
        prepared_messages: list[ChatMessage] | None = None,
        prepared_persona_id: str | None = None,
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
        persona_id = prepared_persona_id or str(
            state["persona"]
            or services.app_config.group_config(group_id).get("persona")
            or services.app_config.default_persona
        )
        persona = services.get_persona(persona_id)
        messages = prepared_messages if prepared_messages is not None else memory.messages_between(
            group_id,
            start_at=start_at,
            end_at=end_at,
            limit=policy.message_limit,
        )
        review_draft = prepared_draft
        try:
            client = services.get_deepseek_client()
            if review_draft is None and client is not None:
                selected_model = services.review_sync_model()
                if selected_model is not None and not bool(getattr(selected_model, "batch", False)):
                    request = self._review_request(
                        client,
                        persona=persona,
                        messages=messages,
                        group_id=group_id,
                        review_label=review_label,
                        start_at=start_at,
                        end_at=end_at,
                        services=services,
                    )
                    complete_on_model = getattr(client, "complete_on_model", None)
                    parse_response = getattr(client, "parse_daily_review_response", None)
                    if request is None or not callable(complete_on_model) or not callable(parse_response):
                        raise RuntimeError("background review model client is not fully configured")
                    response = await complete_on_model(
                        task="daily_review",
                        route=selected_model,
                        request=request,
                    )
                    content = response.choices[0].message.content or ""
                    review_draft = parse_response(content, messages=messages[-140:], max_chars=520)
                else:
                    review_draft = await client.daily_review_draft(
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
