"""Generation and delivery workflow for proactive group messages."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .memory import MemoryStore


@dataclass(frozen=True)
class ProactiveGroupMessagePolicy:
    context_limit: int
    max_messages: int
    mid_memory_keep_summaries: int
    member_impression_context_limit: int
    memory_atom_context_limit: int
    style_rule_context_limit: int
    raw_corpus_context_limit: int
    raw_corpus_candidate_limit: int
    raw_corpus_context_radius: int
    blocked_backend_fallback_texts: frozenset[str]


@dataclass(frozen=True)
class ProactiveGroupContextServices:
    related_member_user_ids: Callable[..., list[int]]
    format_memory_context: Callable[[list[Any]], str]
    format_member_context: Callable[[list[Any]], str]
    format_memory_atom_context: Callable[[list[Any]], str]
    format_style_context: Callable[[list[Any]], str]
    format_raw_corpus_context: Callable[[list[Any]], str]
    selected_group_jargon_context: Callable[..., Awaitable[str]]
    social_action_service: Any
    assemble_generation_context: Callable[..., Any]


@dataclass(frozen=True)
class ProactiveGroupDeliveryServices:
    prepare_political_send_texts: Callable[..., Awaitable[tuple[str, str, list[str]]]]
    send_group_message: Callable[..., Awaitable[int | None]]
    notify_owner_political_gag: Callable[..., Awaitable[None]]
    record_bot_sent_message: Callable[..., None]
    record_metric_event: Callable[..., None]
    action_failed_summary: Callable[[ActionFailed], str]
    short_notice_text: Callable[[str, int], str]
    logger: Any


@dataclass(frozen=True)
class ProactiveGroupMessageServices:
    memory: MemoryStore
    app_config: Any
    get_deepseek_client: Callable[[], Any]
    get_persona: Callable[[str], Any]
    group_generation_inflight: set[int]
    pending_group_approvals: Mapping[int, Any]
    refresh_self_mute_state_if_stale: Callable[..., Awaitable[float]]
    select_topic: Callable[..., Awaitable[tuple[str, str, int]]]
    record_topic: Callable[..., None]
    context: ProactiveGroupContextServices
    delivery: ProactiveGroupDeliveryServices
    sanitize_generated_text: Callable[[str], str]
    split_reply_messages: Callable[..., list[str]]
    policy: ProactiveGroupMessagePolicy


class ProactiveGroupMessageService:
    """Generate and send proactive group posts while sharing live runtime state."""

    @staticmethod
    def context_query(recent_messages: list[Any]) -> str:
        lines = [message.text.strip() for message in recent_messages[-8:] if message.text and message.text.strip()]
        if not lines:
            return "风雪主动发起轻松群聊话题"
        return "\n".join(lines)[-800:]

    async def send_for_group(
        self,
        bot: Bot,
        *,
        group_id: int,
        probability: int,
        roll: float,
        services: ProactiveGroupMessageServices,
    ) -> bool:
        delivery = services.delivery
        record_metric = delivery.record_metric_event
        if services.get_deepseek_client() is None:
            record_metric(
                "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="deepseek_client_not_ready"
            )
            return False
        if group_id in services.group_generation_inflight:
            record_metric(
                "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="group_generation_inflight"
            )
            return False
        if group_id in services.pending_group_approvals:
            record_metric(
                "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="pending_approval_exists"
            )
            return False
        now = time.time()
        memory = services.memory
        state = memory.group_state(group_id)
        group_cfg = services.app_config.group_config(group_id)
        if (
            not services.app_config.group_allowed(group_id)
            or not bool(group_cfg.get("enabled", True))
            or not bool(state["enabled"])
        ):
            record_metric("proactive_chat", group_id=group_id, stage="check", action="skipped", reason="group_disabled")
            return False
        muted_until = await services.refresh_self_mute_state_if_stale(
            bot, group_id, float(state["muted_until"] or 0)
        )
        if muted_until > now:
            record_metric(
                "proactive_chat",
                group_id=group_id,
                stage="check",
                action="skipped",
                reason="self_muted",
                muted_until=muted_until,
            )
            return False
        persona_id = str(state["persona"] or group_cfg.get("persona") or services.app_config.default_persona)
        persona = services.get_persona(persona_id) or services.get_persona(services.app_config.default_persona)
        if persona is None:
            record_metric(
                "proactive_chat",
                group_id=group_id,
                stage="generation",
                action="skipped",
                reason="persona_not_found",
                persona=persona_id,
            )
            return False
        services.group_generation_inflight.add(group_id)
        try:
            policy = services.policy
            context_services = services.context
            recent_messages = memory.recent_messages(group_id, policy.context_limit)
            topic, topic_bucket, cooled_topic_count = await services.select_topic(
                group_id=group_id,
                now=now,
                recent_messages=recent_messages,
                chat_label="QQ 群聊",
            )
            context_query = self.context_query(recent_messages)
            related_user_ids = context_services.related_member_user_ids(recent_messages, current_user_id=0)
            memory_context = context_services.format_memory_context(
                memory.relevant_memory_summaries(
                    group_id, context_query, limit=policy.mid_memory_keep_summaries
                )
            )
            member_context = context_services.format_member_context(
                memory.member_impressions_for_context(
                    group_id, related_user_ids, limit=policy.member_impression_context_limit
                )
            )
            memory_atoms_context = context_services.format_memory_atom_context(
                memory.relevant_memory_atoms(
                    group_id,
                    context_query,
                    subject_user_ids=related_user_ids,
                    relationship_user_ids=related_user_ids,
                    limit=policy.memory_atom_context_limit,
                )
            )
            style_context = context_services.format_style_context(
                memory.relevant_style_rules(group_id, context_query, limit=policy.style_rule_context_limit)
            )
            raw_corpus_context = context_services.format_raw_corpus_context(
                memory.relevant_raw_corpus_examples(
                    group_id,
                    context_query,
                    limit=policy.raw_corpus_context_limit,
                    candidate_limit=policy.raw_corpus_candidate_limit,
                    context_radius=policy.raw_corpus_context_radius,
                    exclude_user_id=int(bot.self_id),
                    per_user_limit=1,
                )
            )
            jargon_context = await context_services.selected_group_jargon_context(
                group_id,
                recent_messages,
                current_text=context_query,
                current_nickname="风雪主动发起",
            )
            social_action_context = context_services.social_action_service.recent_reaction_context(group_id)
            context_packet = context_services.assemble_generation_context(
                memory_context=memory_context,
                member_context=member_context,
                memory_atoms_context=memory_atoms_context,
                style_context=style_context,
                raw_corpus_context=raw_corpus_context,
                jargon_context=jargon_context,
                social_action_context=social_action_context,
            )
            prompt_text = (
                "这是风雪按固定间隔随机主动发起聊天。这次抽到的话题方向是："
                f"{topic}。必须围绕这个方向自然开口；"
                "如果最近群聊能接上，就把话题方向贴进当前聊天；接不上就轻松开一个新话题。"
                "不要解释自己为什么突然说话，不要像公告，不要总结全场。"
            )
            drafts = await services.get_deepseek_client().reply_candidates(
                persona=persona,
                recent_messages=recent_messages,
                current_text=prompt_text,
                current_nickname="风雪主动发起",
                mentioned=False,
                action="reply",
                chat_label="QQ 群聊",
                context_packet=context_packet,
                include_bot_history=True,
                context_message_limit=policy.context_limit,
                candidate_count=1,
                prompt_flow="reply_direct",
                task_name="proactive_chat",
            )
            if not drafts:
                record_metric(
                    "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="empty_model_reply"
                )
                return False
            reply = services.sanitize_generated_text(drafts[0].text)
            if not reply or reply in policy.blocked_backend_fallback_texts:
                record_metric(
                    "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="empty_after_guard"
                )
                return False
            parts = services.split_reply_messages(reply, max_messages=policy.max_messages)
            if not parts:
                record_metric(
                    "proactive_chat", group_id=group_id, stage="generation", action="skipped", reason="empty_after_split"
                )
                return False
            sent_ids: list[int] = []
            for part in parts:
                public_text, memory_text, gag = await delivery.prepare_political_send_texts(part, context=reply)
                message_id = await delivery.send_group_message(bot, group_id, Message(public_text))
                if gag:
                    await delivery.notify_owner_political_gag(
                        original=part,
                        public=public_text,
                        hits=gag,
                        group_id=group_id,
                        source="proactive_chat",
                    )
                delivery.record_bot_sent_message(
                    group_id=group_id,
                    message_id=message_id,
                    bot_reply=memory_text,
                    trigger_user_id=0,
                    trigger_nickname="风雪主动发起",
                    trigger_text=f"interval_random probability={probability} roll={roll:.2f} topic={topic}",
                    action="proactive_chat",
                )
                memory.add_message(
                    group_id,
                    int(bot.self_id),
                    persona.name,
                    memory_text,
                    is_bot=True,
                    source_message_id=message_id,
                    source_kind="proactive_chat",
                    correlation_id=f"proactive:{group_id}:{int(now)}",
                )
                if message_id is not None:
                    sent_ids.append(message_id)
                await asyncio.sleep(random.uniform(0.8, 1.8))
            delivery.logger.info(
                "qq_social_agent proactive chat sent: "
                f"group={group_id} parts={len(parts)} message_ids={sent_ids} "
                f"probability={probability} roll={roll:.2f} topic={topic!r}"
            )
            record_metric(
                "proactive_chat",
                group_id=group_id,
                stage="send",
                action="sent",
                probability=probability,
                roll=round(roll, 2),
                topic=topic,
                topic_bucket=topic_bucket,
                cooled_topic_count=cooled_topic_count,
                message_count=len(parts),
                message_ids=sent_ids,
            )
            services.record_topic(group_id, topic, now=now)
            return True
        except ActionFailed as exc:
            delivery.logger.warning(
                f"qq_social_agent proactive chat send failed: group={group_id} "
                f"{delivery.action_failed_summary(exc)}"
            )
            record_metric(
                "proactive_chat",
                group_id=group_id,
                stage="send",
                action="failed",
                error=delivery.action_failed_summary(exc),
            )
            return False
        except Exception as exc:
            delivery.logger.warning(f"qq_social_agent proactive chat failed: group={group_id} error={exc}")
            record_metric(
                "proactive_chat",
                group_id=group_id,
                stage="generation",
                action="failed",
                error=delivery.short_notice_text(str(exc), 200),
            )
            return False
        finally:
            services.group_generation_inflight.discard(group_id)
