from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from . import onebot_gateway


DEFAULT_REACTION_EMOJI_IDS = {
    "agree": ("76",),
    "like": ("76",),
    "care": ("49",),
    "hug": ("49",),
    "laugh": ("182",),
    "tease": ("101",),
    "surprise": ("32",),
    "question": ("32",),
    "applause": ("99",),
    "heart": ("66",),
}


@dataclass(frozen=True)
class ReactionResult:
    sent: bool
    reason: str
    reaction: str
    emoji_id: str


@dataclass(frozen=True)
class SocialReactionRecord:
    group_id: int
    user_id: int
    target_label: str
    message_id: str
    reaction: str
    emoji_id: str
    created_at: float


@dataclass(frozen=True)
class PokeContext:
    was_poked: bool = False
    ai_selected: bool = False
    directly_cued: bool = False
    familiar_user: bool = False
    playful_banter: bool = False


@dataclass(frozen=True)
class PokeResult:
    sent: bool
    reason: str
    policy_reason: str = ""


class SocialActionService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        emoji_ids: dict[str, object] | None = None,
        per_user_cooldown_seconds: int = 120,
        per_group_cooldown_seconds: int = 18,
        max_per_group_hour: int = 24,
        poke_enabled: bool = False,
        poke_familiar_user_ids: set[int] | None = None,
        poke_per_user_cooldown_seconds: int = 7200,
        poke_per_group_cooldown_seconds: int = 1800,
        poke_global_cooldown_seconds: int = 300,
        poke_max_per_group_day: int = 4,
    ) -> None:
        self.enabled = enabled
        self.emoji_ids = {
            key: tuple(values)
            for key, values in DEFAULT_REACTION_EMOJI_IDS.items()
        }
        if emoji_ids:
            self.emoji_ids.update(_normalize_emoji_id_config(emoji_ids))
        self.per_user_cooldown_seconds = max(0, int(per_user_cooldown_seconds))
        self.per_group_cooldown_seconds = max(0, int(per_group_cooldown_seconds))
        self.max_per_group_hour = max(0, int(max_per_group_hour))
        self._last_user_reaction_at: dict[tuple[int, int], float] = {}
        self._last_group_reaction_at: dict[int, float] = {}
        self._group_reaction_times: dict[int, deque[float]] = {}
        self._reacted_messages: set[tuple[int, str]] = set()
        self._recent_reactions: dict[int, deque[SocialReactionRecord]] = {}
        self.poke_enabled = bool(poke_enabled)
        self.poke_familiar_user_ids = set(poke_familiar_user_ids or set())
        self.poke_per_user_cooldown_seconds = max(0, int(poke_per_user_cooldown_seconds))
        self.poke_per_group_cooldown_seconds = max(0, int(poke_per_group_cooldown_seconds))
        self.poke_global_cooldown_seconds = max(0, int(poke_global_cooldown_seconds))
        self.poke_max_per_group_day = max(0, int(poke_max_per_group_day))
        self._last_user_poke_at: dict[tuple[int, int], float] = {}
        self._last_group_poke_at: dict[int, float] = {}
        self._last_global_poke_at = 0.0
        self._group_poke_times: dict[int, deque[float]] = {}

    @classmethod
    def from_config(cls, raw: object) -> "SocialActionService":
        cfg = raw if isinstance(raw, dict) else {}
        emoji_ids = cfg.get("emoji_ids") if isinstance(cfg.get("emoji_ids"), dict) else None
        poke = cfg.get("poke") if isinstance(cfg.get("poke"), dict) else {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            emoji_ids=emoji_ids,
            per_user_cooldown_seconds=int(cfg.get("per_user_cooldown_seconds", 120)),
            per_group_cooldown_seconds=int(cfg.get("per_group_cooldown_seconds", 18)),
            max_per_group_hour=int(cfg.get("max_per_group_hour", 24)),
            poke_enabled=bool(poke.get("enabled", False)),
            poke_familiar_user_ids=_int_set(poke.get("familiar_user_ids", [])),
            poke_per_user_cooldown_seconds=int(poke.get("per_user_cooldown_seconds", 7200)),
            poke_per_group_cooldown_seconds=int(poke.get("per_group_cooldown_seconds", 1800)),
            poke_global_cooldown_seconds=int(poke.get("global_cooldown_seconds", 300)),
            poke_max_per_group_day=int(poke.get("max_per_group_day", 4)),
        )

    async def poke_user(
        self,
        bot: onebot_gateway.OneBotGateway,
        *,
        group_id: int,
        user_id: int,
        context: PokeContext,
        now: float | None = None,
    ) -> PokeResult:
        current = time.time() if now is None else now
        policy_reason = self._poke_policy_reason(user_id, context)
        if policy_reason.startswith("deny_"):
            return PokeResult(False, policy_reason, policy_reason)
        cooldown_reason = self._poke_cooldown_reason(group_id, user_id, now=current)
        if cooldown_reason:
            return PokeResult(False, cooldown_reason, policy_reason)
        await onebot_gateway.send_poke(bot, user_id, group_id=group_id)
        self._remember_poke(group_id, user_id, now=current)
        return PokeResult(True, "sent", policy_reason)

    def _poke_policy_reason(self, user_id: int, context: PokeContext) -> str:
        if not self.poke_enabled:
            return "deny_disabled"
        familiar = bool(context.familiar_user or user_id in self.poke_familiar_user_ids)
        if context.was_poked:
            return "reciprocal_poke"
        if context.ai_selected:
            return "ai_selected_poke"
        if familiar and context.directly_cued:
            return "familiar_direct_cue"
        if familiar and context.playful_banter:
            return "familiar_banter"
        if not familiar:
            return "deny_unfamiliar_user"
        return "deny_missing_social_signal"

    def _poke_cooldown_reason(self, group_id: int, user_id: int, *, now: float) -> str:
        if self.poke_global_cooldown_seconds and now - self._last_global_poke_at < self.poke_global_cooldown_seconds:
            return "poke_global_cooldown"
        last_group = self._last_group_poke_at.get(group_id, 0.0)
        if self.poke_per_group_cooldown_seconds and now - last_group < self.poke_per_group_cooldown_seconds:
            return "poke_group_cooldown"
        last_user = self._last_user_poke_at.get((group_id, user_id), 0.0)
        if self.poke_per_user_cooldown_seconds and now - last_user < self.poke_per_user_cooldown_seconds:
            return "poke_user_cooldown"
        bucket = self._group_poke_times.setdefault(group_id, deque())
        while bucket and now - bucket[0] > 24 * 60 * 60:
            bucket.popleft()
        if self.poke_max_per_group_day and len(bucket) >= self.poke_max_per_group_day:
            return "poke_group_daily_limit"
        return ""

    def _remember_poke(self, group_id: int, user_id: int, *, now: float) -> None:
        self._last_global_poke_at = now
        self._last_group_poke_at[group_id] = now
        self._last_user_poke_at[(group_id, user_id)] = now
        self._group_poke_times.setdefault(group_id, deque()).append(now)

    def status_snapshot(self) -> dict[str, object]:
        return {
            "reactions_enabled": self.enabled,
            "poke_enabled": self.poke_enabled,
            "poke_familiar_user_count": len(self.poke_familiar_user_ids),
            "poke_global_cooldown_seconds": self.poke_global_cooldown_seconds,
            "poke_per_group_cooldown_seconds": self.poke_per_group_cooldown_seconds,
            "poke_per_user_cooldown_seconds": self.poke_per_user_cooldown_seconds,
            "poke_max_per_group_day": self.poke_max_per_group_day,
        }

    def recent_reaction_context(self, group_id: int, *, limit: int = 4) -> str:
        records = self._recent_reactions.get(group_id)
        if not records:
            return ""
        lines: list[str] = []
        for record in list(records)[-max(1, limit) :]:
            label = record.target_label or f"QQ {record.user_id}"
            lines.append(f"- 风雪刚给 {label} 的消息点了 {record.reaction} 表情")
        return "\n".join(lines)

    async def react_to_message(
        self,
        bot: onebot_gateway.OneBotGateway,
        *,
        group_id: int,
        user_id: int,
        message_id: int | str,
        reaction: str,
        target_label: str = "",
        now: float | None = None,
    ) -> ReactionResult:
        current = time.time() if now is None else now
        normalized = normalize_reaction(reaction)
        emoji_id = self._emoji_id_for(group_id, normalized)
        message_key = str(message_id or "").strip()
        if not self.enabled:
            return ReactionResult(False, "disabled", normalized, emoji_id)
        if not message_key:
            return ReactionResult(False, "missing_message_id", normalized, emoji_id)
        reason = self._cooldown_reason(group_id, user_id, message_key, now=current)
        if reason:
            return ReactionResult(False, reason, normalized, emoji_id)
        await onebot_gateway.set_msg_emoji_like(bot, message_key, emoji_id)
        self._remember_reaction(
            group_id,
            user_id,
            message_key,
            reaction=normalized,
            emoji_id=emoji_id,
            target_label=target_label,
            now=current,
        )
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

    def _emoji_id_for(self, group_id: int, reaction: str) -> str:
        candidates = self.emoji_ids.get(reaction) or self.emoji_ids["agree"]
        if len(candidates) <= 1:
            return candidates[0]
        recent = list(self._recent_reactions.get(group_id, ()))
        last_emoji = recent[-1].emoji_id if recent else ""
        usage = {emoji_id: 0 for emoji_id in candidates}
        for record in recent:
            if record.emoji_id in usage:
                usage[record.emoji_id] += 1
        ranked = sorted(
            candidates,
            key=lambda emoji_id: (
                emoji_id == last_emoji,
                usage.get(emoji_id, 0),
                candidates.index(emoji_id),
            ),
        )
        return ranked[0]

    def _remember_reaction(
        self,
        group_id: int,
        user_id: int,
        message_id: str,
        *,
        reaction: str,
        emoji_id: str,
        target_label: str,
        now: float,
    ) -> None:
        self._reacted_messages.add((group_id, message_id))
        self._last_user_reaction_at[(group_id, user_id)] = now
        self._last_group_reaction_at[group_id] = now
        self._group_reaction_times.setdefault(group_id, deque()).append(now)
        bucket = self._recent_reactions.setdefault(group_id, deque(maxlen=20))
        bucket.append(
            SocialReactionRecord(
                group_id=group_id,
                user_id=user_id,
                target_label=target_label,
                message_id=message_id,
                reaction=reaction,
                emoji_id=emoji_id,
                created_at=now,
            )
        )


def reaction_from_action(action: str, requested_reaction: str = "") -> str:
    normalized = normalize_reaction(requested_reaction)
    if requested_reaction and normalized in DEFAULT_REACTION_EMOJI_IDS:
        return normalized
    mapping = {
        "agree": "agree",
        "care": "care",
        "tease": "tease",
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


def _normalize_emoji_id_config(raw: dict[str, object]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_value in raw.items():
        key = normalize_reaction(str(raw_key))
        values: list[str] = []
        if isinstance(raw_value, (list, tuple, set)):
            source = raw_value
        else:
            source = (raw_value,)
        for item in source:
            value = str(item or "").strip()
            if value and value not in values:
                values.append(value)
        if values:
            result[key] = tuple(values)
    return result


def _int_set(value: object) -> set[int]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    result: set[int] = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            result.add(parsed)
    return result
