from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot, Message, PrivateMessageEvent

from .content_ingestion import ContentIngestionService
from .media_context import ImageOcrContext, collect_ocr_image_segments
from .memory import MemoryStore
from .rag_retriever import RAGService
from .tools.voice_transcript import VoiceTranscriptContext
from .private_message_types import BufferedPrivateMessage, PrivateTurn


@dataclass(frozen=True)
class PrivateTurnServices:
    memory: MemoryStore
    rag_service: RAGService
    content_ingestion: ContentIngestionService
    approval_handler: Callable[[Bot, int, str], Awaitable[bool]]
    private_user_can_chat: Callable[[int], bool]
    private_chat_id: Callable[[int], int]
    source_message_id: Callable[[PrivateMessageEvent], str]
    private_nickname: Callable[[PrivateMessageEvent], str]
    file_metadata_context: Callable[..., Awaitable[str]]
    media_worth_reading: Callable[..., Awaitable[bool]]
    image_ocr_context_for_event: Callable[..., Awaitable[ImageOcrContext]]
    format_image_ocr_context: Callable[[ImageOcrContext], str]
    message_has_forward_context: Callable[[PrivateMessageEvent], bool]
    forward_context_text: Callable[..., Awaitable[str]]
    force_obey_command_response: Callable[[int, str], str | None]
    extract_force_obey_once_text: Callable[[int, str], str | None]
    force_obey_context: Callable[..., str]
    message_text_for_context: Callable[..., Awaitable[str]]
    event_message_storage_kwargs: Callable[..., dict[str, str]]
    schedule_private_memory_maintenance: Callable[[int], None]
    send_private_message: Callable[..., Awaitable[object]]
    record_metric_event: Callable[..., None]
    message_factory: Callable[[str], Message]
    short_notice_text: Callable[[str, int], str]
    private_context_reset_commands: frozenset[str]
    command_only_private_user_ids: frozenset[int]
    long_message_summary_threshold: int
    logger: Any


def _join_context_parts(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _join_context_blocks(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


async def prepare_private_turn(
    bot: Bot,
    event: PrivateMessageEvent,
    *,
    correlation_id: str,
    received_text: str,
    buffered_messages: list[BufferedPrivateMessage] | None,
    services: PrivateTurnServices,
) -> PrivateTurn | None:
    """Claim, authorize, enrich, normalize, and persist one private turn."""

    buffered_items = list(buffered_messages or ())
    if buffered_items:
        latest = buffered_items[-1]
        bot, event, correlation_id, text = latest.bot, latest.event, latest.correlation_id, latest.text
    else:
        text = received_text
    if not text:
        services.logger.info("qq_social_agent ignored private: empty_text")
        return None

    user_id = int(event.user_id)
    chat_id = services.private_chat_id(user_id)
    source_message_id = services.source_message_id(event)
    claim_items = buffered_items or [
        BufferedPrivateMessage(
            bot=bot,
            event=event,
            text=text,
            user_id=user_id,
            nickname=services.private_nickname(event),
            created_at=float(getattr(event, "time", 0) or time.time()),
            source_message_id=source_message_id,
            correlation_id=correlation_id,
        )
    ]
    accepted_items: list[BufferedPrivateMessage] = []
    for item in claim_items:
        if services.memory.claim_inbound_message(
            chat_id,
            item.source_message_id,
            correlation_id=item.correlation_id,
            created_at=item.created_at,
        ):
            accepted_items.append(item)
    if not accepted_items:
        services.logger.info(
            "qq_social_agent ignored duplicate private message: "
            f"user={user_id} source_message_id={source_message_id}"
        )
        services.record_metric_event(
            "message_duplicate",
            group_id=chat_id,
            user_id=user_id,
            stage="private",
            action="duplicate",
            source_message_id=source_message_id,
            correlation_id=correlation_id,
        )
        return None

    latest = accepted_items[-1]
    if buffered_items:
        bot, event = latest.bot, latest.event
        correlation_id, source_message_id, text = latest.correlation_id, latest.source_message_id, latest.text
    if await services.approval_handler(bot, user_id, text):
        return None
    if user_id in services.command_only_private_user_ids:
        services.logger.info(f"qq_social_agent ignored private: user={user_id} command_only")
        return None
    if not services.private_user_can_chat(user_id):
        services.logger.info(f"qq_social_agent ignored private: user={user_id} not_allowed")
        return None

    file_context = await services.file_metadata_context(bot, event)
    if file_context and file_context not in text:
        text = _join_context_parts(text, file_context)
    content_context = await services.content_ingestion.context_for_event(
        bot,
        event,
        allow_file_content=True,
        voice_context=VoiceTranscriptContext(mentioned=True),
    )
    if content_context.text:
        text = _join_context_parts(text, content_context.text)
    if content_context.file_count or content_context.voice_count:
        services.record_metric_event(
            "content_ingestion",
            group_id=chat_id,
            user_id=user_id,
            stage="private_media_context",
            action="recognized" if content_context.text else "skipped",
            file_count=content_context.file_count,
            voice_count=content_context.voice_count,
            file_status=content_context.file_status,
            voice_status=content_context.voice_status,
        )
    image_segments = collect_ocr_image_segments(getattr(event, "message", []) or [])
    if image_segments and await services.media_worth_reading(
        kind="ocr",
        caption=text,
        addressed=True,
        item_count=len(image_segments),
        group_id=chat_id,
        user_id=user_id,
    ):
        ocr_context = await services.image_ocr_context_for_event(
            bot,
            event,
            group_allowed=True,
            group_id=chat_id,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        if ocr_context.text:
            text = _join_context_parts(text, services.format_image_ocr_context(ocr_context))
    elif image_segments:
        ocr_context = ImageOcrContext("", len(image_segments), 0, "jev_skip")

    forward_context = ""
    if services.message_has_forward_context(event) and await services.media_worth_reading(
        kind="forward",
        caption=text,
        addressed=True,
        item_count=1,
        group_id=chat_id,
        user_id=user_id,
    ):
        forward_context = await services.forward_context_text(bot, event, nickname=services.private_nickname(event))
        if not forward_context:
            services.record_metric_event(
                "content_ingestion",
                group_id=chat_id,
                user_id=user_id,
                stage="private_forward_context",
                action="unavailable",
                source_message_id=source_message_id,
            )

    force_obey_response = services.force_obey_command_response(user_id, text)
    if force_obey_response is not None:
        await services.send_private_message(bot, user_id=user_id, message=services.message_factory(force_obey_response))
        services.logger.info(f"qq_social_agent private force obey command: user={user_id} text={text!r}")
        return None
    if text in services.private_context_reset_commands:
        services.memory.reset_group_messages(chat_id)
        await services.send_private_message(
            bot,
            user_id=user_id,
            message=services.message_factory("私聊上下文已清空，重新开始。"),
        )
        services.logger.info(f"qq_social_agent private context reset: user={user_id}")
        return None

    forced_once_context = ""
    forced_once_text = services.extract_force_obey_once_text(user_id, text)
    if forced_once_text is not None:
        text = forced_once_text
        forced_once_context = services.force_obey_context(user_id, one_shot=True)
    nickname = services.private_nickname(event)
    if not forward_context or len((text or "").strip()) > services.long_message_summary_threshold:
        text = await services.message_text_for_context(text, nickname=nickname, chat_label="QQ 私聊")
    if forward_context:
        text = _join_context_blocks(text, forward_context)

    prompt_text = text
    if len(accepted_items) > 1:
        earlier = [services.short_notice_text(item.text, 240) for item in accepted_items[:-1] if item.text.strip()]
        if earlier:
            prompt_text = "[对方连续发了多条私聊]\n" + "\n".join(earlier + [text])
    services.logger.info(f"qq_social_agent private start: user={user_id} text={text!r}")
    for item in accepted_items[:-1]:
        services.memory.add_message(
            chat_id,
            user_id,
            item.nickname,
            item.text,
            is_bot=False,
            source_message_id=item.source_message_id,
            correlation_id=item.correlation_id,
            **services.event_message_storage_kwargs(item.event, bot=item.bot),
        )
    services.memory.add_message(
        chat_id,
        user_id,
        nickname,
        text,
        is_bot=False,
        source_message_id=source_message_id,
        correlation_id=correlation_id,
        **services.event_message_storage_kwargs(event, bot=bot),
    )
    private_state = services.memory.private_conversation_state(chat_id)
    if private_state is None or "display_name" not in private_state.frozen_fields:
        services.memory.update_private_conversation_state(chat_id=chat_id, user_id=user_id, display_name=nickname)
    services.rag_service.request_source_sync()
    services.schedule_private_memory_maintenance(chat_id)
    return PrivateTurn(
        bot=bot,
        user_id=user_id,
        chat_id=chat_id,
        self_id=int(event.self_id),
        nickname=nickname,
        source_message_id=source_message_id,
        correlation_id=correlation_id,
        text=prompt_text,
        forced_once_context=forced_once_context,
        received_message_count=len(accepted_items),
    )
