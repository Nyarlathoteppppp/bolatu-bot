from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable


class BackgroundLearningCoordinator:
    """Single-worker scheduler so learning never fans out from message handlers."""

    def __init__(
        self,
        maintain: Callable[[int], Awaitable[None]],
        *,
        target_groups: Callable[[], Iterable[int]],
        is_busy: Callable[[int], bool] | None = None,
        sweep_seconds: float = 60.0,
        busy_retry_seconds: float = 10.0,
    ) -> None:
        self.maintain = maintain
        self.target_groups = target_groups
        self.is_busy = is_busy or (lambda _group_id: False)
        self.sweep_seconds = max(5.0, float(sweep_seconds))
        self.busy_retry_seconds = max(0.01, float(busy_retry_seconds))
        self._pending: set[int] = set()
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._current_group_id: int | None = None
        self._last_error = ""

    def start(self) -> None:
        if self._closed or (self._task is not None and not self._task.done()):
            return
        self._pending.update(int(value) for value in self.target_groups() if int(value) > 0)
        self._event.set()
        self._task = asyncio.create_task(self._run())

    def notify(self, group_id: int) -> None:
        if group_id <= 0 or self._closed:
            return
        self._pending.add(int(group_id))
        self._event.set()

    async def close(self) -> None:
        self._closed = True
        self._event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def status_snapshot(self) -> dict[str, object]:
        return {
            "running": bool(self._task is not None and not self._task.done()),
            "pending_groups": sorted(self._pending),
            "current_group_id": self._current_group_id,
            "last_error": self._last_error,
            "sweep_seconds": self.sweep_seconds,
        }

    async def _run(self) -> None:
        while not self._closed:
            if not self._pending:
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=self.sweep_seconds)
                except asyncio.TimeoutError:
                    self._pending.update(
                        int(value) for value in self.target_groups() if int(value) > 0
                    )
                self._event.clear()
            if not self._pending:
                continue
            group_id = min(self._pending)
            self._pending.discard(group_id)
            if self.is_busy(group_id):
                self._pending.add(group_id)
                await asyncio.sleep(self.busy_retry_seconds)
                continue
            self._current_group_id = group_id
            try:
                await self.maintain(group_id)
                self._last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:240]
            finally:
                self._current_group_id = None
