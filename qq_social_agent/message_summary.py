"""Compress incoming message content independently of the NoneBot runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from nonebot import logger

from .speaker_context import _short_notice_text


class MessageSummarizer(Protocol):
    async def summarize_long_message(
        self, *, text: str, speaker_label: str, chat_label: str,
        original_chars: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class MessageSummaryPolicy:
    threshold: int = 100
    source_limit: int = 1800
    fallback_head: int = 72
    fallback_tail: int = 28
    forward_threshold: int = 1400


def compact_long_message(text: str, policy: MessageSummaryPolicy) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= policy.threshold:
        return clean
    head = clean[:policy.fallback_head].rstrip()
    tail = clean[-policy.fallback_tail:].lstrip()
    if tail and tail not in head:
        return f"{head} ... [长消息{len(clean)}字，已省略] ... {tail}"
    return f"{head} ... [长消息{len(clean)}字，已省略]"


def compact_forward_records(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= 8 and sum(len(line) for line in lines) <= 900:
        return "\n".join(lines)
    head = lines[:5]
    tail = lines[-2:] if len(lines) > 7 else []
    omitted = max(0, len(lines) - len(head) - len(tail))
    parts = list(head)
    if omitted:
        parts.append(f"...[另有{omitted}条转发记录]")
    parts.extend(tail)
    return "\n".join(parts)


class MessageSummaryService:
    def __init__(self, client: MessageSummarizer | None, policy: MessageSummaryPolicy) -> None:
        self.client = client
        self.policy = policy

    async def message_text(self, text: str, *, nickname: str, chat_label: str) -> str:
        clean = text.strip()
        if len(clean) <= self.policy.threshold:
            return text
        fallback = compact_long_message(clean, self.policy)
        if self.client is None:
            return fallback
        try:
            summary = await self.client.summarize_long_message(
                text=clean[:self.policy.source_limit], speaker_label=nickname,
                chat_label=chat_label, original_chars=len(clean),
            )
        except Exception as exc:
            logger.warning(
                "qq_social_agent long message summary failed: "
                f"chat={chat_label} nickname={nickname!r} chars={len(clean)} error={exc}"
            )
            return fallback
        summary = re.sub(r"\s+", " ", summary).strip()
        if not summary:
            return fallback
        if len(summary) > 160:
            summary = summary[:157].rstrip() + "..."
        logger.info(
            "qq_social_agent compacted long message: "
            f"chat={chat_label} nickname={nickname!r} raw_chars={len(clean)} summary_chars={len(summary)}"
        )
        return f"[长消息{len(clean)}字摘要] {summary}"

    async def forward_records(self, raw: str, *, nickname: str) -> str:
        clean = raw.strip()
        if not clean:
            return ""
        if len(clean) <= self.policy.forward_threshold:
            return clean
        fallback = compact_forward_records(clean)
        if self.client is None:
            return fallback
        try:
            summary = await self.client.summarize_long_message(
                text=raw[:self.policy.source_limit],
                speaker_label=f"多位原发言人（由{nickname}转发）",
                chat_label="QQ 转发聊天记录，每行已标明原发言人，不要把内容算成转发者说的",
                original_chars=len(raw),
            )
        except Exception as exc:
            logger.warning(
                "qq_social_agent forward context summary failed: "
                f"nickname={nickname!r} chars={len(raw)} error={exc}"
            )
            return fallback
        summary = re.sub(r"\s+", " ", summary).strip()
        return _short_notice_text(summary, 360) if summary else fallback
