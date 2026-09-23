from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from nonebot.adapters.onebot.v11 import Bot, Message, PrivateMessageEvent

from .history_sync import event_message_source_id
from .memory import ChatMessage, MemoryStore
from .private_message_types import BufferedPrivateMessage
from .persona import PersonaRegistry
from .rate_limiter import RateLimiter
from .reply_splitter import split_reply_messages


@dataclass(frozen=True)
class PrivateFollowupServices:
    memory: MemoryStore
    get_deepseek_client: Callable[[], Any]
    private_user_can_chat: Callable[[int], bool]
    private_chat_id: Callable[[int], int]
    rate_limiter: RateLimiter
    personas: PersonaRegistry
    default_persona: str
    format_memory_context: Callable[[list[Any]], str]
    private_priority_context: Callable[[int], str]
    member_label: Callable[[int, str], str]
    first_connected_onebot_bot: Callable[[], Bot | None]
    send_private_message: Callable[..., Awaitable[object]]
    record_metric_event: Callable[..., None]
    short_notice_text: Callable[[str, int], str]
    sanitize_generated_text: Callable[[str], str]
    blocked_backend_fallback_texts: frozenset[str]
    probability_by_user: Mapping[int, float]
    default_probability: float
    private_context_limit: int
    mid_memory_keep_summaries: int
    logger: Any


class PrivateSessionService:
    """Own per-user private buffering, serialization, and follow-up scheduling."""

    def __init__(self) -> None:
        self.processing_locks: dict[int, asyncio.Lock] = {}
        self.message_buffers: dict[int, list[BufferedPrivateMessage]] = {}
        self.buffer_tasks: dict[int, asyncio.Task[None]] = {}
        self.generation_inflight: set[int] = set()
        self.inbound_message_counts: dict[int, int] = {}
        self.followup_tasks: dict[int, asyncio.Task[None]] = {}

    def buffer_message(
        self,
        bot: Bot,
        event: PrivateMessageEvent,
        *,
        text: str,
        correlation_id: str,
        private_nickname: Callable[[PrivateMessageEvent], str],
        delay: float,
        flush: Callable[..., Awaitable[None]],
        logger: Any,
    ) -> None:
        user_id = int(event.user_id)
        item = BufferedPrivateMessage(
            bot=bot,
            event=event,
            text=text,
            user_id=user_id,
            nickname=private_nickname(event),
            created_at=float(getattr(event, "time", 0) or time.time()),
            source_message_id=event_message_source_id(event),
            correlation_id=correlation_id,
        )
        self.message_buffers.setdefault(user_id, []).append(item)
        self.schedule_buffer_flush(user_id, flush=flush, delay=delay)
        logger.info(
            "qq_social_agent buffered private message: "
            f"user={user_id} size={len(self.message_buffers.get(user_id, []))}"
        )

    def schedule_buffer_flush(
        self,
        user_id: int,
        *,
        flush: Callable[..., Awaitable[None]],
        delay: float,
    ) -> None:
        task = self.buffer_tasks.get(user_id)
        if task is None or task.done():
            self.buffer_tasks[user_id] = asyncio.create_task(flush(user_id, delay=delay))

    def processing_lock(self, user_id: int) -> asyncio.Lock:
        lock = self.processing_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self.processing_locks[user_id] = lock
        return lock

    async def flush_after_delay(
        self,
        user_id: int,
        *,
        delay: float,
        retry_delay: float,
        handle_private_message: Callable[..., Awaitable[None]],
        correlation_scope: Callable[[str], Any],
        schedule_followup: Callable[..., None],
        schedule_buffer_flush: Callable[..., None],
        logger: Any,
    ) -> None:
        should_reschedule = False
        reschedule_delay = retry_delay
        try:
            await asyncio.sleep(delay)
            async with self.processing_lock(user_id):
                if user_id in self.generation_inflight:
                    should_reschedule = True
                    logger.info(
                        "qq_social_agent private generation inflight: "
                        f"user={user_id} buffer_deferred size={len(self.message_buffers.get(user_id, []))}"
                    )
                    return
                items = self.message_buffers.pop(user_id, [])
                if not items:
                    return
                self.generation_inflight.add(user_id)
                try:
                    latest = items[-1]
                    with correlation_scope(latest.correlation_id):
                        await handle_private_message(
                            latest.bot,
                            latest.event,
                            correlation_id=latest.correlation_id,
                            buffered_messages=items,
                        )
                finally:
                    self.generation_inflight.discard(user_id)
                schedule_followup(user_id, added_messages=len(items))
                if self.message_buffers.get(user_id):
                    should_reschedule = True
        finally:
            task = asyncio.current_task()
            if self.buffer_tasks.get(user_id) is task:
                self.buffer_tasks.pop(user_id, None)
            if should_reschedule and self.message_buffers.get(user_id):
                schedule_buffer_flush(user_id, delay=reschedule_delay)

    def schedule_followup_if_due(
        self,
        user_id: int,
        *,
        added_messages: int,
        delay: float,
        run_followup: Callable[..., Awaitable[None]],
    ) -> None:
        total = self.inbound_message_counts.get(user_id, 0) + max(0, added_messages)
        self.inbound_message_counts[user_id] = total
        if total < 2 or total % 2:
            return
        previous = self.followup_tasks.get(user_id)
        if previous is not None and not previous.done():
            previous.cancel()
        self.followup_tasks[user_id] = asyncio.create_task(
            run_followup(user_id, expected_message_count=total, delay=delay)
        )

    @staticmethod
    def followup_probability(user_id: int, *, probability_by_user: Mapping[int, float], default: float) -> float:
        return max(0.0, min(1.0, probability_by_user.get(user_id, default)))

    async def run_followup_after_delay(
        self,
        user_id: int,
        *,
        expected_message_count: int,
        delay: float,
        services: PrivateFollowupServices,
    ) -> None:
        try:
            await asyncio.sleep(delay)
            if self.inbound_message_counts.get(user_id, 0) != expected_message_count:
                return
            if self.message_buffers.get(user_id) or user_id in self.generation_inflight:
                return
            roll = random.random()
            probability = self.followup_probability(
                user_id,
                probability_by_user=services.probability_by_user,
                default=services.default_probability,
            )
            chat_id = services.private_chat_id(user_id)
            if roll >= probability:
                services.record_metric_event(
                    "private_followup",
                    group_id=chat_id,
                    user_id=user_id,
                    stage="probability",
                    action="skipped",
                    probability=probability,
                    roll=round(roll, 3),
                )
                return
            deepseek_client = services.get_deepseek_client()
            if deepseek_client is None or not services.private_user_can_chat(user_id):
                return
            state = services.memory.group_state(chat_id)
            if not bool(state["enabled"]):
                return
            rate = services.rate_limiter.allow(chat_id, mentioned=True)
            if not rate.allowed:
                return
            persona = services.personas.get(str(state["persona"] or services.default_persona))
            if persona is None:
                return
            recent = services.memory.recent_messages(chat_id, services.private_context_limit)
            if len(recent) < 2:
                return
            self.generation_inflight.add(user_id)
            try:
                should_continue, continue_reason = await deepseek_client.should_continue_private_chat(
                    persona=persona,
                    recent_messages=recent,
                )
                if not should_continue:
                    services.record_metric_event(
                        "private_followup",
                        group_id=chat_id,
                        user_id=user_id,
                        stage="precheck",
                        action="skipped",
                        probability=probability,
                        roll=round(roll, 3),
                        reason=services.short_notice_text(continue_reason, 80),
                    )
                    return
                reply = await deepseek_client.reply(
                    persona=persona,
                    recent_messages=recent,
                    current_text=(
                        "对方刚连续和你聊了几句，现在停了十秒。"
                        "如果能自然延续刚才的话题、补一个有用观点或轻轻开个新话题，就主动发一句；"
                        "如果没有自然的话，不要回复。"
                    ),
                    current_nickname=services.member_label(
                        user_id,
                        private_nickname_from_recent(recent, user_id),
                    ),
                    mentioned=False,
                    action="reply",
                    chat_label="QQ 私聊",
                    memory_context=services.format_memory_context(
                        services.memory.relevant_memory_summaries(
                            chat_id,
                            " ".join(msg.text for msg in recent[-4:]),
                            limit=services.mid_memory_keep_summaries,
                        )
                    ),
                    priority_context=services.private_priority_context(user_id),
                    speaker_context="当前是一对一私聊。只能自然续聊，不要提群聊、审批或工具流程。",
                )
                audit_send, audit_reason = await deepseek_client.audit_proactive_reply(
                    persona=persona,
                    recent_messages=recent,
                    candidate=reply,
                    chat_label="QQ 私聊",
                )
            finally:
                self.generation_inflight.discard(user_id)
            if not audit_send:
                services.record_metric_event(
                    "private_followup",
                    group_id=chat_id,
                    user_id=user_id,
                    stage="audit",
                    action="rejected",
                    probability=probability,
                    roll=round(roll, 3),
                    reason=services.short_notice_text(audit_reason or "模型判定与近期句子大致重复", 80),
                )
                return
            reply = services.sanitize_generated_text(reply)
            if not reply or reply in services.blocked_backend_fallback_texts:
                return
            bot = services.first_connected_onebot_bot()
            if bot is None:
                return
            for part in split_reply_messages(reply, max_messages=2):
                await services.send_private_message(bot, user_id=user_id, message=Message(part))
                services.memory.add_message(chat_id, int(bot.self_id), persona.name, part, is_bot=True)
            services.record_metric_event(
                "private_followup",
                group_id=chat_id,
                user_id=user_id,
                stage="generation",
                action="sent",
                probability=probability,
                roll=round(roll, 3),
                audit_reason=services.short_notice_text(audit_reason, 80),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            services.logger.warning(f"qq_social_agent private follow-up failed: user={user_id} error={exc}")
        finally:
            task = asyncio.current_task()
            if self.followup_tasks.get(user_id) is task:
                self.followup_tasks.pop(user_id, None)


def private_nickname_from_recent(recent: list[ChatMessage], user_id: int) -> str:
    for item in reversed(recent):
        if not item.is_bot and item.user_id == user_id:
            return item.nickname
    return "对方"
