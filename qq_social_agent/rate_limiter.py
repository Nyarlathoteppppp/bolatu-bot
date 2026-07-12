from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from .config import RateConfig
from .memory import ChatMessage, MemoryStore


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    reason: str


class RateLimiter:
    def __init__(self, memory: MemoryStore, config: RateConfig):
        self.memory = memory
        self.config = config

    def allow(
        self,
        group_id: int,
        *,
        mentioned: bool,
        now: float | None = None,
        event_at: float | None = None,
    ) -> RateDecision:
        now = now or time.time()
        state = self.memory.group_state(group_id)
        if not state["enabled"]:
            return RateDecision(False, "group_paused")
        if float(state["muted_until"]) > now:
            return RateDecision(False, "group_muted")
        if self.config.quiet_hours_enabled and self._in_quiet_hours():
            if not mentioned:
                return RateDecision(False, "quiet_hours")

        replies_hour = self.memory.recent_bot_replies(group_id, 3600)
        replies_10min = [msg for msg in replies_hour if now - msg.created_at <= 600]
        if len(replies_hour) >= self.config.max_replies_per_hour:
            return RateDecision(False, "hour_limit")
        if len(replies_10min) >= self.config.max_replies_per_10min:
            return RateDecision(False, "ten_min_limit")

        cooldown_reference = now if event_at is None else float(event_at)
        last = next(
            (reply for reply in replies_hour if reply.created_at <= cooldown_reference),
            None,
        )
        if last:
            min_interval = (
                self.config.hard_mention_interval_seconds
                if mentioned
                else self.config.min_interval_seconds
            )
            if cooldown_reference - last.created_at < min_interval:
                return RateDecision(False, "cooldown")

        if self._consecutive_bot_replies(group_id) >= self.config.max_consecutive_replies:
            if not mentioned:
                return RateDecision(False, "consecutive_limit")

        return RateDecision(True, "ok")

    def _consecutive_bot_replies(self, group_id: int) -> int:
        recent = self.memory.recent_messages(group_id, 8)
        count = 0
        for msg in reversed(recent):
            if not msg.is_bot:
                break
            count += 1
        return count

    def _in_quiet_hours(self) -> bool:
        now = datetime.now().strftime("%H:%M")
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        if start <= end:
            return start <= now < end
        return now >= start or now < end
