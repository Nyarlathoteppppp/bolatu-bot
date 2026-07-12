from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import onebot_gateway
from .media_context import coerce_int, message_text_from_payload, sender_nickname
from .memory import MemoryStore


@dataclass(frozen=True)
class HistoricalMessage:
    group_id: int
    message_id: str
    user_id: int
    nickname: str
    text: str
    created_at: float


@dataclass(frozen=True)
class ReplyReference:
    message_id: str
    user_id: int | None
    nickname: str
    text: str


async def backfill_group_history(
    bot: onebot_gateway.OneBotGateway,
    memory: MemoryStore,
    group_id: int,
    *,
    count: int,
    self_id: int,
) -> int:
    raw_messages = await onebot_gateway.get_group_msg_history(bot, group_id, count=count)
    normalized = normalize_history_messages(raw_messages, fallback_group_id=group_id)
    inserted = 0
    for item in sorted(normalized, key=lambda message: (message.created_at, message.message_id)):
        if not item.text:
            continue
        is_bot = item.user_id == self_id
        if memory.add_message(
            item.group_id,
            item.user_id,
            item.nickname,
            item.text,
            is_bot=is_bot,
            created_at=item.created_at,
            source_message_id=item.message_id,
            source_kind="history",
            correlation_id=f"history:{item.group_id}:{item.message_id}",
        ):
            inserted += 1
    return inserted


def normalize_history_messages(payload: Any, *, fallback_group_id: int) -> list[HistoricalMessage]:
    items = payload if isinstance(payload, list) else []
    return [
        message
        for item in items
        if isinstance(item, dict)
        for message in [normalize_message_payload(item, fallback_group_id=fallback_group_id)]
        if message is not None
    ]


def normalize_message_payload(payload: dict[str, Any], *, fallback_group_id: int) -> HistoricalMessage | None:
    message_id = _message_id_from_payload(payload)
    if not message_id:
        return None
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    user_id = coerce_int(payload.get("user_id") or sender.get("user_id") or sender.get("uin"), 0)
    if not user_id:
        return None
    group_id = coerce_int(payload.get("group_id"), fallback_group_id)
    nickname = sender_nickname(sender, fallback_user_id=user_id)
    text = message_text_from_payload(payload)
    created_at = float(coerce_int(payload.get("time") or payload.get("timestamp"), 0) or time.time())
    return HistoricalMessage(
        group_id=group_id,
        message_id=message_id,
        user_id=user_id,
        nickname=nickname or str(user_id),
        text=text,
        created_at=created_at,
    )


async def resolve_reply_reference(
    bot: onebot_gateway.OneBotGateway,
    event: Any,
) -> ReplyReference | None:
    if _event_reply_has_text(event):
        return None
    message_id = reply_message_id(event)
    if not message_id:
        return None
    payload = await onebot_gateway.get_msg(bot, message_id)
    if not payload:
        return None
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    user_id = coerce_int(payload.get("user_id") or sender.get("user_id") or sender.get("uin"), 0) or None
    nickname = sender_nickname(sender, fallback_user_id=user_id)
    text = message_text_from_payload(payload)
    return ReplyReference(message_id=str(message_id), user_id=user_id, nickname=nickname, text=text)


def reply_message_id(event: Any) -> str:
    reply = getattr(event, "reply", None)
    message_id = getattr(reply, "message_id", None) if reply is not None else None
    if message_id:
        return str(message_id)
    for segment in getattr(event, "message", []) or []:
        segment_type = str(getattr(segment, "type", "") or "")
        data = getattr(segment, "data", {}) or {}
        if isinstance(segment, dict):
            segment_type = str(segment.get("type", segment_type) or "")
            data = segment.get("data", data) or {}
        if segment_type == "reply":
            for key in ("id", "message_id"):
                value = str(data.get(key, "") or "").strip() if isinstance(data, dict) else ""
                if value:
                    return value
    return ""


def event_message_source_id(event: Any) -> str:
    return str(getattr(event, "message_id", "") or "").strip()


def _event_reply_has_text(event: Any) -> bool:
    reply = getattr(event, "reply", None)
    if reply is None:
        return False
    raw_message = getattr(reply, "message", None)
    return bool(message_text_from_payload(raw_message))


def _message_id_from_payload(payload: dict[str, Any]) -> str:
    for key in ("message_id", "message_seq", "msg_id", "msgId", "id"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    return ""
