from __future__ import annotations

import time
from dataclasses import dataclass

from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent

from .member_context import member_label as _member_label
from .pipeline_types import PipelineState


@dataclass(frozen=True)
class BufferedGroupMessage:
    bot: Bot
    event: GroupMessageEvent
    text: str
    user_id: int
    nickname: str
    created_at: float
    source_message_id: str = ""
    correlation_id: str = ""
    inbound_sequence: int = 0
    pipeline_state: PipelineState | None = None
    addressed: bool = False
    direct_addressed: bool = False
    followup_soft: bool = False
    session_id: str = ""
    message_segments_json: str = ""
    raw_message_json: str = ""
    sender_json: str = ""


def buffered_current_text(items: list[BufferedGroupMessage] | None) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0].text
    recent_items = items[-6:]
    last_item = items[-1]
    last_label = _member_label(last_item.user_id, last_item.nickname)
    speaker_count = len({item.user_id for item in recent_items})
    lines = [
        f"【连续消息，按时间顺序；最后触发者：{last_label}】",
        f"当前发言人只有 {last_label}。",
    ]
    if speaker_count > 1:
        lines.append(
            f"上面编号里还有其他人，他们不是 {last_label}；不要把旁人的话当成 {last_label} 说的，也不要把两个人认成同一个。"
        )
    if len(items) > len(recent_items):
        lines.append(f"（前面还有 {len(items) - len(recent_items)} 条普通群消息）")
    for index, item in enumerate(recent_items, start=1):
        line = buffered_message_context_line(index, item, last_user_id=last_item.user_id)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def buffered_message_context_line(
    index: int,
    item: BufferedGroupMessage,
    *,
    last_user_id: int | None = None,
) -> str:
    text = (item.text or "").strip()
    if not text:
        return ""
    label = _member_label(item.user_id, item.nickname)
    if text.startswith(f"{label}回复") or text.startswith(f"{label}说"):
        body = text
    else:
        body = f"{label}说：{text}"
    role = "当前发言" if last_user_id is not None and item.user_id == last_user_id else "旁人"
    return f"{index}. [{role}] {body}"


def buffered_current_user_id(items: list[BufferedGroupMessage] | None) -> int:
    if not items:
        return 0
    return items[-1].user_id


def buffered_current_nickname(items: list[BufferedGroupMessage] | None) -> str:
    if not items:
        return "群友"
    return items[-1].nickname


def buffered_last_created_at(items: list[BufferedGroupMessage] | None) -> float:
    if not items:
        return time.time()
    return items[-1].created_at

