"""Lifecycle and timing policy for scheduled proactive group chat."""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo


class ProactiveChatSchedulerService:
    def __init__(
        self,
        *,
        timezone: ZoneInfo,
        interval_seconds: float,
        poll_jitter_seconds: float,
        daytime_percent: int,
        quiet_percent: int,
        quiet_start_hour: int,
        quiet_end_hour: int,
        target_groups: Callable[[], tuple[int, ...]],
        send_for_group: Callable[[object, int, int, float], Awaitable[bool]],
        record_metric_event: Callable[..., None],
        logger: Any,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.timezone = timezone
        self.interval_seconds = interval_seconds
        self.poll_jitter_seconds = poll_jitter_seconds
        self.daytime_percent = daytime_percent
        self.quiet_percent = quiet_percent
        self.quiet_start_hour = quiet_start_hour
        self.quiet_end_hour = quiet_end_hour
        self.target_groups = target_groups
        self.send_for_group = send_for_group
        self.record_metric_event = record_metric_event
        self.logger = logger
        self.uniform = uniform
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def ensure_task(self, bot: object) -> None:
        bot_key = str(getattr(bot, "self_id", "default"))
        task = self.tasks.get(bot_key)
        if task is not None and not task.done():
            return
        self.tasks[bot_key] = asyncio.create_task(self._run(bot, bot_key))
        self.logger.info(
            "qq_social_agent proactive chat scheduler started: "
            f"bot={bot_key} interval={int(self.interval_seconds)}s "
            f"daytime={self.daytime_percent}% quiet={self.quiet_percent}%"
        )

    def seconds_until_next_tick(
        self,
        now: float | None = None,
        *,
        interval_seconds: float | None = None,
        poll_jitter_seconds: float | None = None,
    ) -> float:
        interval = self.interval_seconds if interval_seconds is None else interval_seconds
        jitter_limit = self.poll_jitter_seconds if poll_jitter_seconds is None else poll_jitter_seconds
        jitter = self.uniform(0.0, jitter_limit) if jitter_limit > 0 else 0.0
        return max(1.0, interval + jitter)

    def probability_percent(self, now: float | None = None) -> int:
        current = time.time() if now is None else now
        hour = datetime.fromtimestamp(current, self.timezone).hour
        if self._hour_in_range(hour, self.quiet_start_hour, self.quiet_end_hour):
            return self.quiet_percent
        return self.daytime_percent

    @staticmethod
    def _hour_in_range(hour: int, start: int, end: int) -> bool:
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    async def _run(self, bot: object, bot_key: str) -> None:
        try:
            if self.poll_jitter_seconds > 0:
                await asyncio.sleep(self.uniform(1.0, self.poll_jitter_seconds))
            while True:
                await asyncio.sleep(self.seconds_until_next_tick())
                now = time.time()
                probability = self.probability_percent(now)
                for group_id in self.target_groups():
                    roll = self.uniform(0.0, 100.0)
                    if roll >= probability:
                        self.logger.info(
                            "qq_social_agent proactive chat skipped by probability: "
                            f"group={group_id} probability={probability} roll={roll:.2f}"
                        )
                        self.record_metric_event(
                            "proactive_chat",
                            group_id=group_id,
                            stage="probability",
                            action="skipped",
                            probability=probability,
                            roll=round(roll, 2),
                        )
                        continue
                    await self.send_for_group(bot, group_id, probability, roll)
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(f"qq_social_agent proactive chat scheduler stopped: bot={bot_key} error={exc}")
        finally:
            if self.tasks.get(bot_key) is asyncio.current_task():
                self.tasks.pop(bot_key, None)
