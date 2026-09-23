"""Lifecycle and timing policy for the weekly usage report."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo


class WeeklyUsageReportSchedulerService:
    def __init__(
        self,
        *,
        timezone: ZoneInfo,
        weekday: int,
        hour: int,
        minute: int,
        poll_seconds: int,
        send_report: Callable[[object, float], Awaitable[bool]],
        logger: Any,
    ) -> None:
        self.timezone = timezone
        self.weekday = weekday
        self.hour = hour
        self.minute = minute
        self.poll_seconds = poll_seconds
        self.send_report = send_report
        self.logger = logger
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def ensure_task(self, bot: object) -> None:
        bot_key = str(getattr(bot, "self_id", "default"))
        task = self.tasks.get(bot_key)
        if task is not None and not task.done():
            return
        self.tasks[bot_key] = asyncio.create_task(self._run(bot, bot_key))
        self.logger.info(f"qq_social_agent weekly usage report scheduler started: bot={bot_key}")

    def seconds_until_next_report(self, now: float | None = None) -> float:
        current = time.time() if now is None else now
        local = datetime.fromtimestamp(current, self.timezone)
        days_ahead = (self.weekday - local.weekday()) % 7
        target_date = (local + timedelta(days=days_ahead)).replace(
            hour=self.hour,
            minute=self.minute,
            second=0,
            microsecond=0,
        )
        if target_date.timestamp() <= current:
            target_date = target_date + timedelta(days=7)
        return max(1.0, target_date.timestamp() - current)

    def report_due(self, now: float | None = None) -> bool:
        current = datetime.fromtimestamp(time.time() if now is None else now, self.timezone)
        if current.weekday() != self.weekday:
            return False
        return current.hour > self.hour or (current.hour == self.hour and current.minute >= self.minute)

    async def _run(self, bot: object, bot_key: str) -> None:
        try:
            while True:
                delay = self.poll_seconds
                try:
                    now = time.time()
                    if self.report_due(now):
                        await self.send_report(bot, now)
                    delay = min(self.seconds_until_next_report(time.time()), self.poll_seconds)
                except Exception as exc:
                    self.logger.warning(
                        f"qq_social_agent weekly usage report tick failed: bot={bot_key} error={exc}"
                    )
                await asyncio.sleep(max(1.0, delay))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                f"qq_social_agent weekly usage report scheduler stopped: bot={bot_key} error={exc}"
            )
        finally:
            if self.tasks.get(bot_key) is asyncio.current_task():
                self.tasks.pop(bot_key, None)
