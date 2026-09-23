"""Lifecycle and timing policy for scheduled daily reviews."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo


class DailyReviewSchedulerService:
    def __init__(
        self,
        *,
        timezone: ZoneInfo,
        hour: int,
        minute: int,
        catch_up_seconds: int,
        retry_seconds: int,
        poll_seconds: int,
        send_due_reviews: Callable[[object, float], Awaitable[bool]],
        record_metric_event: Callable[..., None],
        logger: Any,
        summarize_error: Callable[[str, int], str],
    ) -> None:
        self.timezone = timezone
        self.hour = hour
        self.minute = minute
        self.catch_up_seconds = catch_up_seconds
        self.retry_seconds = retry_seconds
        self.poll_seconds = poll_seconds
        self.send_due_reviews = send_due_reviews
        self.record_metric_event = record_metric_event
        self.logger = logger
        self.summarize_error = summarize_error
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def ensure_task(self, bot: object) -> None:
        bot_key = str(getattr(bot, "self_id", "default"))
        task = self.tasks.get(bot_key)
        if task is not None and not task.done():
            return
        self.tasks[bot_key] = asyncio.create_task(self._run(bot, bot_key))
        self.logger.info(f"qq_social_agent daily review scheduler started: bot={bot_key}")

    def seconds_until_next_review(self, now: float | None = None) -> float:
        current = time.time() if now is None else now
        target = self._today_target(current)
        if current <= target:
            return max(1.0, target - current)
        if current - target <= 90:
            return 1.0
        return max(1.0, target + 24 * 60 * 60 - current)

    def within_catch_up_window(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        target = self._today_target(current)
        return 0 <= current - target <= self.catch_up_seconds

    def _today_target(self, now: float) -> float:
        local = datetime.fromtimestamp(now, self.timezone)
        return local.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0).timestamp()

    async def _run(self, bot: object, bot_key: str) -> None:
        try:
            while True:
                delay = self.retry_seconds
                try:
                    now = time.time()
                    within_catch_up = self.within_catch_up_window(now)
                    has_pending = False
                    if within_catch_up:
                        has_pending = await self.send_due_reviews(bot, now)
                    delay = (
                        self.retry_seconds
                        if within_catch_up and has_pending
                        else min(self.seconds_until_next_review(time.time()), self.poll_seconds)
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"qq_social_agent daily review scheduler tick failed: bot={bot_key} error={exc}"
                    )
                    self.record_metric_event(
                        "daily_review",
                        stage="scheduler",
                        action="tick_failed",
                        reason=self.summarize_error(str(exc), 200),
                    )
                await asyncio.sleep(max(1.0, delay))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(f"qq_social_agent daily review scheduler stopped: bot={bot_key} error={exc}")
        finally:
            if self.tasks.get(bot_key) is asyncio.current_task():
                self.tasks.pop(bot_key, None)
