from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import onebot_gateway
from .media_context import coerce_int
from .memory import MemoryStore


@dataclass(frozen=True)
class GroupDirectorySyncResult:
    group_id: int
    member_count: int
    group_name: str
    synced_at: float


async def sync_group_directory(
    bot: onebot_gateway.OneBotGateway,
    memory: MemoryStore,
    group_id: int,
) -> GroupDirectorySyncResult:
    synced_at = time.time()
    group_name = ""
    member_count = 0
    group_info = await onebot_gateway.get_group_info(bot, group_id)
    if group_info:
        group_name = str(group_info.get("group_name") or group_info.get("group_memo") or "").strip()
        member_count = coerce_int(group_info.get("member_count"), 0)
        memory.upsert_group_info(
            group_id=group_id,
            group_name=group_name,
            member_count=member_count,
            max_member_count=coerce_int(group_info.get("max_member_count"), 0),
            last_synced_at=synced_at,
        )

    raw_members = await onebot_gateway.get_group_member_list(bot, group_id)
    members = [_normalize_member(group_id, item, synced_at=synced_at) for item in raw_members]
    members = [member for member in members if member["user_id"]]
    memory.replace_group_members(group_id, members, synced_at=synced_at)
    if members:
        member_count = len(members)
    return GroupDirectorySyncResult(
        group_id=group_id,
        member_count=member_count,
        group_name=group_name,
        synced_at=synced_at,
    )


def _normalize_member(group_id: int, payload: dict[str, Any], *, synced_at: float) -> dict[str, Any]:
    user_id = coerce_int(payload.get("user_id") or payload.get("uin") or payload.get("uid"), 0)
    card = str(payload.get("card") or payload.get("card_name") or "").strip()
    nickname = str(payload.get("nickname") or payload.get("nick") or payload.get("name") or "").strip()
    title = str(payload.get("title") or payload.get("special_title") or "").strip()
    return {
        "group_id": group_id,
        "user_id": user_id,
        "nickname": nickname or str(user_id),
        "card": card,
        "role": str(payload.get("role") or "").strip(),
        "title": title,
        "joined_at": float(coerce_int(payload.get("join_time") or payload.get("joined_at"), 0)),
        "last_sent_at": float(coerce_int(payload.get("last_sent_time") or payload.get("last_sent_at"), 0)),
        "last_synced_at": synced_at,
        "active": True,
    }
