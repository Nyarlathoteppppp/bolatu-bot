from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .memory import MemoryStore
from .meme_library import PrivateMemeLibrary
from .private_message_types import PrivateGenerationContext, PrivateTurn
from .reply_splitter import split_reply_messages


@dataclass(frozen=True)
class PrivateReplyServices:
    memory: MemoryStore
    private_meme_library: PrivateMemeLibrary
    get_deepseek_client: Callable[[], Any]
    send_private_message: Callable[..., Awaitable[object]]
    message_with_reply_quote: Callable[[Message, object], Message]
    sanitize_generated_text: Callable[[str], str]
    private_meme_context_eligible: Callable[..., bool]
    action_failed_summary: Callable[[Exception], str]
    record_metric_event: Callable[..., None]
    logger: Any


async def generate_and_send_private_reply(
    turn: PrivateTurn,
    context: PrivateGenerationContext,
    *,
    services: PrivateReplyServices,
) -> None:
    deepseek_client = services.get_deepseek_client()
    try:
        reply = await deepseek_client.reply(
            persona=context.persona,
            recent_messages=list(context.recent_messages),
            current_text=context.current_text,
            current_nickname=context.current_nickname,
            mentioned=True,
            chat_label="QQ 私聊",
            action=context.decision.action,
            market_context=context.market_context,
            fresh_context=context.fresh_context,
            memory_context=context.memory_context,
            member_context=context.member_context,
            memory_atoms_context=context.memory_atoms_context,
            style_context=context.style_context,
            raw_corpus_context=context.raw_corpus_context,
            jargon_context=context.jargon_context,
            recall_feedback_context=context.recall_feedback_context,
            speaker_context=context.speaker_context,
            priority_context=context.priority_context,
        )
    except Exception as exc:
        services.logger.warning(
            f"qq_social_agent private reply generation failed: user={turn.user_id} error={exc}"
        )
        return
    if not reply:
        services.logger.info(f"qq_social_agent skipped private reply: user={turn.user_id} reason=empty_model_reply")
        return
    reply = services.sanitize_generated_text(reply)

    selected_meme_id: int | None = None
    meme_gate = services.private_meme_library.turn_gate(
        turn.user_id,
        received_messages=turn.received_message_count,
    )
    if meme_gate.allowed and services.private_meme_context_eligible(
        decision=context.decision,
        reply=reply,
        market_context=context.market_context,
        fresh_context=context.fresh_context,
    ):
        meme_candidates = services.private_meme_library.candidates(
            turn.user_id,
            query=f"{context.current_text}\n{reply}",
        )
        if meme_candidates:
            try:
                meme_choice = await deepseek_client.select_private_meme(
                    current_text=context.current_text,
                    reply_text=reply,
                    candidates=services.private_meme_library.candidate_text(meme_candidates),
                )
            except Exception as exc:
                services.logger.warning(
                    f"qq_social_agent private meme selector failed: user={turn.user_id} error={exc}"
                )
                meme_choice = None
            candidate_ids = {asset.id for asset in meme_candidates}
            if meme_choice is not None and meme_choice.send and meme_choice.meme_id in candidate_ids:
                selected_meme_id = meme_choice.meme_id
            services.record_metric_event(
                "private_meme_selector",
                group_id=turn.chat_id,
                user_id=turn.user_id,
                stage="selection",
                action="selected" if selected_meme_id else "skipped",
                gate_reason=meme_gate.reason,
                turn_count=meme_gate.messages_since_last_meme,
                selected_meme_id=selected_meme_id,
                reason=meme_choice.reason if meme_choice is not None else "selector_error",
            )
    else:
        services.record_metric_event(
            "private_meme_selector",
            group_id=turn.chat_id,
            user_id=turn.user_id,
            stage="eligibility",
            action="skipped",
            gate_reason=meme_gate.reason,
            turn_count=meme_gate.messages_since_last_meme,
        )

    reply_parts = split_reply_messages(reply, max_messages=3)
    services.logger.info(
        "qq_social_agent sending private reply: "
        f"user={turn.user_id} chars={len(reply)} parts={len(reply_parts)}"
    )
    for index, part in enumerate(reply_parts):
        try:
            await services.send_private_message(
                turn.bot,
                user_id=turn.user_id,
                message=services.message_with_reply_quote(Message(part), turn.source_message_id if index == 0 else ""),
            )
            services.memory.add_message(turn.chat_id, turn.self_id, context.persona.name, part, is_bot=True)
        except ActionFailed as exc:
            services.logger.warning(
                "qq_social_agent failed sending private reply: "
                f"user={turn.user_id} {services.action_failed_summary(exc)}"
            )
            return
        if index < len(reply_parts) - 1:
            await asyncio.sleep(0.9)
    if selected_meme_id is not None:
        image_ref = services.private_meme_library.image_base64_ref(selected_meme_id)
        if image_ref:
            try:
                await services.send_private_message(
                    turn.bot,
                    user_id=turn.user_id,
                    message=Message(MessageSegment.image(file=image_ref)),
                )
            except ActionFailed as exc:
                services.logger.warning(
                    "qq_social_agent failed sending private meme: "
                    f"user={turn.user_id} meme={selected_meme_id} {services.action_failed_summary(exc)}"
                )
            else:
                services.private_meme_library.mark_sent(turn.user_id, selected_meme_id)
                asset = services.memory.meme_asset(selected_meme_id)
                services.memory.add_message(
                    turn.chat_id,
                    turn.self_id,
                    context.persona.name,
                    f"[风雪附了一张私人表情包：{asset.description if asset else selected_meme_id}]",
                    is_bot=True,
                )
                services.record_metric_event(
                    "private_meme_selector",
                    group_id=turn.chat_id,
                    user_id=turn.user_id,
                    stage="delivery",
                    action="sent",
                    meme_id=selected_meme_id,
                )
