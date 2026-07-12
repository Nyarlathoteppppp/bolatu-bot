from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from . import onebot_gateway


DEFAULT_REACTION_EMOJI_IDS = {
    "agree": "76",
    "like": "76",
    "care": "49",
    "hug": "49",
    "laugh": "28",
    "tease": "101",
    "surprise": "32",
    "question": "32",
    "applause": "99",
    "heart": "66",
}


@dataclass(frozen=True)
class ReactionResult:
    sent: bool
    reason: str
    reaction: str
    emoji_id: str


class SocialActionService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        emoji_ids: dict[str, str] | None = None,
        per_user_cooldown_seconds: int = 120,
        per_group_cooldown_seconds: int = 18,
        max_per_group_hour: int = 24,
    ) -> None:
        self.enabled = enabled
        self.emoji_ids = dict(DEFAULT_REACTION_EMOJI_IDS)
        if emoji_ids:
            self.emoji_ids.update({str(key): str(value) for key, value in emoji_ids.items() if value})
        self.per_user_cooldown_seconds = max(0, int(per_user_cooldown_seconds))
        self.per_group_cooldown_seconds = max(0, int(per_group_cooldown_seconds))
        self.max_per_group_hour = max(0, int(max_per_group_hour))
        self._last_user_reaction_at: dict[tuple[int, int], float] = {}
        self._last_group_reaction_at: dict[int, float] = {}
        self._group_reaction_times: dict[int, deque[float]] = {}
        self._reacted_messages: set[tuple[int, str]] = set()

    @classmethod
    def from_config(cls, raw: object) -> "SocialActionService":
        cfg = raw if isinstance(raw, dict) else {}
        emoji_ids = cfg.get("emoji_ids") if isinstance(cfg.get("emoji_ids"), dict) else None
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            emoji_ids=emoji_ids,
            per_user_cooldown_seconds=int(cfg.get("per_user_cooldown_seconds", 120)),
            per_group_cooldown_seconds=int(cfg.get("per_group_cooldown_seconds", 18)),
            max_per_group_hour=int(cfg.get("max_per_group_hour", 24)),
        )

    async def react_to_message(
        self,
        bot: onebot_gateway.OneBotGateway,
        *,
        group_id: int,
        user_id: int,
        message_id: int | str,
        reaction: str,
        now: float | None = None,
    ) -> ReactionResult:
        current = time.time() if now is None else now
        normalized = normalize_reaction(reaction)
        emoji_id = self.emoji_ids.get(normalized, self.emoji_ids["agree"])
        message_key = str(message_id or "").strip()
        if not self.enabled:
            return ReactionResult(False, "disabled", normalized, emoji_id)
        if not message_key:
            return ReactionResult(False, "missing_message_id", normalized, emoji_id)
        reason = self._cooldown_reason(group_id, user_id, message_key, now=current)
        if reason:
            return ReactionResult(False, reason, normalized, emoji_id)
        await onebot_gateway.set_msg_emoji_like(bot, message_key, emoji_id)
        self._remember_reaction(group_id, user_id, message_key, now=current)
        return ReactionResult(True, "sent", normalized, emoji_id)

    def _cooldown_reason(self, group_id: int, user_id: int, message_id: str, *, now: float) -> str:
        if (group_id, message_id) in self._reacted_messages:
            return "message_already_reacted"
        last_user_at = self._last_user_reaction_at.get((group_id, user_id), 0.0)
        if self.per_user_cooldown_seconds and now - last_user_at < self.per_user_cooldown_seconds:
            return "user_cooldown"
        last_group_at = self._last_group_reaction_at.get(group_id, 0.0)
        if self.per_group_cooldown_seconds and now - last_group_at < self.per_group_cooldown_seconds:
            return "group_cooldown"
        bucket = self._group_reaction_times.setdefault(group_id, deque())
        while bucket and now - bucket[0] > 3600:
            bucket.popleft()
        if self.max_per_group_hour and len(bucket) >= self.max_per_group_hour:
            return "group_hourly_limit"
        return ""

    def _remember_reaction(self, group_id: int, user_id: int, message_id: str, *, now: float) -> None:
        self._reacted_messages.add((group_id, message_id))
        self._last_user_reaction_at[(group_id, user_id)] = now
        self._last_group_reaction_at[group_id] = now
        self._group_reaction_times.setdefault(group_id, deque()).append(now)


def reaction_from_action(action: str, requested_reaction: str = "") -> str:
    normalized = normalize_reaction(requested_reaction)
    if requested_reaction and normalized in DEFAULT_REACTION_EMOJI_IDS:
        return normalized
    mapping = {
        "agree": "agree",
        "care": "care",
        "tease": "tease",
        "mock_repeated_question": "tease",
        "echo_mood": "laugh",
        "observe": "like",
        "react": "agree",
    }
    return mapping.get(str(action or "").strip().lower(), "agree")


def normalize_reaction(value: str) -> str:
    key = str(value or "").strip().lower()
    aliases = {
        "": "agree",
        "thumb": "agree",
        "thumbs_up": "agree",
        "thumbsup": "agree",
        "赞": "agree",
        "like": "agree",
        "ok": "agree",
        "hug": "care",
        "comfort": "care",
        "抱抱": "care",
        "heart": "heart",
        "爱心": "heart",
        "哈哈": "laugh",
        "笑": "laugh",
        "笑死": "laugh",
        "bad_laugh": "tease",
        "坏笑": "tease",
        "surprised": "surprise",
        "问号": "question",
        "clap": "applause",
        "鼓掌": "applause",
    }
    normalized = aliases.get(key, key)
    return normalized if normalized in DEFAULT_REACTION_EMOJI_IDS else "agree"
