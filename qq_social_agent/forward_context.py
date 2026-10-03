"""Read forwarded chat records while preserving original speakers and image ownership."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Protocol

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent
from nonebot.adapters.onebot.v11.exception import ActionFailed

from . import onebot_gateway
from .media_context import ImageOcrResult, collect_ocr_image_segments
from .message_segments import message_text_from_payload, segment_type_and_data
from .message_summary import MessageSummaryService
from .member_context import member_label
from .speaker_context import _short_notice_text


class ForwardImageReader(Protocol):
    async def ocr_image_segment(self, bot: Bot, data: dict[str, object]) -> ImageOcrResult | None: ...


@dataclass(frozen=True)
class ForwardContextPolicy:
    timezone: tzinfo
    max_records: int = 20
    max_images: int = 4
    line_limit: int = 320


def forward_message_ids(event: GroupMessageEvent | PrivateMessageEvent) -> list[str]:
    ids: list[str] = []
    for segment in event.message:
        segment_type, data = segment_type_and_data(segment)
        if segment_type != "forward":
            continue
        for key in ("id", "forward_id", "resid"):
            value = str(data.get(key, "") or "").strip()
            if value:
                ids.append(value)
                break
    return ids


def inline_forward_payloads(event: GroupMessageEvent | PrivateMessageEvent) -> list[object]:
    payloads: list[object] = []
    for segment in event.message:
        segment_type, data = segment_type_and_data(segment)
        if segment_type != "forward":
            continue
        for key in ("content", "messages", "message"):
            value = data.get(key)
            if isinstance(value, (list, dict)) and value:
                payloads.append(value)
                break
    return payloads


def normalized_forward_item(item: object) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None
    node = item.get("data") if str(item.get("type", "") or "").casefold() == "node" else None
    normalized = node if isinstance(node, dict) else item
    return normalized if isinstance(normalized, dict) else None


def forward_messages_from_payload(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if str(payload.get("type", "") or "").casefold() == "node":
        return [payload]
    for key in ("messages", "message", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("messages", "message", "content"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def forward_sender_label(sender: object, item: dict[str, object]) -> str:
    if isinstance(sender, dict):
        name = str(sender.get("card") or sender.get("nickname") or sender.get("name") or "").strip()
        user_id = str(sender.get("user_id") or sender.get("uin") or "").strip()
        if name and user_id:
            return member_label(int(user_id), name) if user_id.isdigit() else name
        if name:
            return name
        if user_id:
            return f"QQ{user_id}"
    fallback = str(item.get("sender_name") or item.get("nickname") or item.get("user_id") or "某人").strip()
    return fallback or "某人"


def forward_content_plain_text(content: object) -> str:
    return message_text_from_payload(content, language="zh")


def forward_record_time_label(item: dict[str, object], *, timezone: tzinfo) -> str:
    raw_time = item.get("time", item.get("timestamp", item.get("msg_time", 0)))
    try:
        timestamp = float(raw_time or 0)
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    if timestamp > 10_000_000_000:
        timestamp /= 1000.0
    try:
        return datetime.fromtimestamp(timestamp, timezone).strftime("%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def format_forward_record_line(item: object, *, policy: ForwardContextPolicy, ocr_text: str = "") -> str:
    normalized = normalized_forward_item(item)
    if normalized is None:
        return ""
    sender = normalized.get("sender") if isinstance(normalized.get("sender"), dict) else {}
    sender_name = forward_sender_label(sender, normalized)
    content = normalized.get("content", normalized.get("message", ""))
    text = forward_content_plain_text(content)
    extra = _short_notice_text(ocr_text, 360)
    if extra and extra not in text:
        text = f"{text} {extra}".strip() if text else extra
    if not text:
        return ""
    timestamp = forward_record_time_label(normalized, timezone=policy.timezone)
    prefix = f"[{timestamp}] " if timestamp else ""
    return f"{prefix}{sender_name}: {_short_notice_text(text, policy.line_limit)}"


def extract_forward_record_lines(payload: object, *, limit: int, policy: ForwardContextPolicy) -> list[str]:
    if limit <= 0:
        return []
    lines: list[str] = []
    for item in forward_messages_from_payload(payload):
        if len(lines) >= limit:
            break
        line = format_forward_record_line(item, policy=policy)
        if line:
            lines.append(line)
    return lines


class ForwardContextService:
    def __init__(
        self, *, summaries: MessageSummaryService, images: ForwardImageReader | None,
        policy: ForwardContextPolicy,
    ) -> None:
        self.summaries = summaries
        self.images = images
        self.policy = policy

    async def context_text(
        self,
        bot: Bot,
        event: GroupMessageEvent | PrivateMessageEvent,
        *,
        nickname: str,
    ) -> str:
        payloads: list[object] = list(inline_forward_payloads(event))
        if not payloads:
            for forward_id in forward_message_ids(event)[:2]:
                try:
                    payload = await onebot_gateway.get_forward_msg(bot, forward_id)
                except ActionFailed as exc:
                    logger.warning(
                        "qq_social_agent forward context fetch failed: "
                        f"forward_id={forward_id} {onebot_gateway.action_failed_summary(exc)}"
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "qq_social_agent forward context fetch failed: "
                        f"forward_id={forward_id} error={exc}"
                    )
                    continue
                if payload:
                    payloads.append(payload)
                if payloads:
                    break
        records = await self.records_from_payloads(bot, payloads)
        if not records:
            return ""
        raw = "\n".join(records)
        summary = await self.summaries.forward_records(raw, nickname=nickname)
        if not summary:
            return ""
        return f"{nickname}传了聊天记录，内容如下：\n{summary}"

    async def records_from_payloads(self, bot: Bot, payloads: list[object]) -> list[str]:
        records: list[str] = []
        ocr_remaining = self.policy.max_images
        for payload in payloads:
            for item in forward_messages_from_payload(payload):
                if len(records) >= self.policy.max_records:
                    return records
                ocr_text = ""
                used = 0
                if ocr_remaining > 0:
                    ocr_text, used = await self.ocr_record_images(bot, item, remaining=ocr_remaining)
                    ocr_remaining = max(0, ocr_remaining - used)
                line = format_forward_record_line(item, policy=self.policy, ocr_text=ocr_text)
                if line:
                    records.append(line)
        return records

    async def ocr_record_images(
        self,
        bot: Bot,
        item: object,
        *,
        remaining: int,
    ) -> tuple[str, int]:
        if remaining <= 0 or self.images is None:
            return "", 0
        normalized = normalized_forward_item(item)
        if normalized is None:
            return "", 0
        content = normalized.get("content", normalized.get("message", ""))
        images = collect_ocr_image_segments(content, limit=remaining)
        if not images:
            return "", 0
        texts: list[str] = []
        used = 0
        for data in images:
            if used >= remaining:
                break
            used += 1
            try:
                result = await self.images.ocr_image_segment(bot, data)
            except Exception as exc:
                logger.warning(f"qq_social_agent forward image ocr failed: error={exc}")
                continue
            if result is None or not result.text:
                continue
            texts.append(_short_notice_text(result.text, 280))
        if not texts:
            return "", used
        rendered = "；".join(f"[图:{item}]" for item in texts)
        return rendered, used
