from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .group_message_types import BufferedGroupMessage
from .observability import correlation_scope


@dataclass
class GroupSessionState:
    """The shared registries for ordinary, addressed, and proactive group turns."""

    processing_locks: dict[int, asyncio.Lock] = field(default_factory=dict)
    message_buffers: dict[int, list[BufferedGroupMessage]] = field(default_factory=dict)
    buffer_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    generation_inflight: set[int] = field(default_factory=set)
    addressed_waiters: dict[int, int] = field(default_factory=dict)
    inbound_sequences: dict[int, int] = field(default_factory=dict)


class GroupSessionService:
    """Serialize group turns and preserve addressed speaker/trigger ordering.

    State is supplied by the composition root; creating an adapter does not copy
    queues or locks. The same registries are shared with shutdown and proactive
    generation.
    """

    def __init__(self, state: GroupSessionState) -> None:
        self.state = state

    def processing_lock(self, group_id: int) -> asyncio.Lock:
        lock = self.state.processing_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self.state.processing_locks[group_id] = lock
        return lock

    def next_inbound_sequence(self, group_id: int) -> int:
        sequence = self.state.inbound_sequences.get(group_id, 0) + 1
        self.state.inbound_sequences[group_id] = sequence
        return sequence

    def begin_addressed(self, group_id: int) -> None:
        self.state.addressed_waiters[group_id] = self.state.addressed_waiters.get(group_id, 0) + 1

    def end_addressed(self, group_id: int) -> None:
        remaining = self.state.addressed_waiters.get(group_id, 1) - 1
        if remaining > 0:
            self.state.addressed_waiters[group_id] = remaining
        else:
            self.state.addressed_waiters.pop(group_id, None)

    def should_defer_reply(self, group_id: int) -> bool:
        return (
            group_id in self.state.generation_inflight
            or self.state.addressed_waiters.get(group_id, 0) > 0
            or any(item.addressed or item.direct_addressed for item in self.state.message_buffers.get(group_id, ()))
        )

    def drain_for_contextual_search(self, group_id: int) -> list[BufferedGroupMessage]:
        task = self.state.buffer_tasks.pop(group_id, None)
        if task is not None and not task.done():
            task.cancel()
        return self.state.message_buffers.pop(group_id, [])

    def buffer_message(
        self,
        group_id: int,
        item: BufferedGroupMessage,
        *,
        schedule_buffer_flush: Callable[..., None],
        logger: Any,
    ) -> None:
        self.state.message_buffers.setdefault(group_id, []).append(item)
        schedule_buffer_flush(group_id)
        logger.info(
            "qq_social_agent buffered group message: "
            f"group={group_id} size={len(self.state.message_buffers.get(group_id, []))}"
        )

    def schedule_buffer_flush(
        self,
        group_id: int,
        *,
        delay: float,
        flush: Callable[..., Awaitable[None]],
    ) -> None:
        task = self.state.buffer_tasks.get(group_id)
        if task is None or task.done():
            self.state.buffer_tasks[group_id] = asyncio.create_task(flush(group_id, delay=delay))

    async def flush_after_delay(
        self,
        group_id: int,
        *,
        delay: float,
        retry_delay: float,
        handle_group_message: Callable[..., Awaitable[None]],
        schedule_buffer_flush: Callable[..., None],
        record_metric_event: Callable[..., None],
        logger: Any,
    ) -> None:
        should_reschedule = False
        reschedule_delay = retry_delay
        try:
            await asyncio.sleep(delay)
            async with self.processing_lock(group_id):
                if self.state.addressed_waiters.get(group_id, 0) > 0:
                    should_reschedule = True
                    return
                if group_id in self.state.generation_inflight:
                    logger.info(
                        "qq_social_agent group generation inflight: "
                        f"group={group_id} buffer_deferred size={len(self.state.message_buffers.get(group_id, []))}"
                    )
                    should_reschedule = True
                    return
                items = self.state.message_buffers.pop(group_id, [])
                if not items:
                    return
                first_addressed = next(
                    (index for index, item in enumerate(items) if item.addressed or item.direct_addressed),
                    None,
                )
                if first_addressed is not None:
                    first_user = items[first_addressed].user_id
                    batch_start = first_addressed
                    while batch_start > 0 and items[batch_start - 1].user_id == first_user:
                        batch_start -= 1
                    batch_end = first_addressed + 1
                    while batch_end < len(items) and items[batch_end].user_id == first_user:
                        batch_end += 1
                    batch = items[batch_start:batch_end]
                    rest = items[:batch_start] + items[batch_end:]
                    if rest:
                        self.state.message_buffers[group_id] = rest
                        should_reschedule = True
                    items = batch
                    logger.info(
                        "qq_social_agent split addressed group buffer: "
                        f"group={group_id} keep_user={first_user} "
                        f"batch={len(items)} remaining={len(rest)}"
                    )
                logger.info(
                    "qq_social_agent flushing group buffer: "
                    f"group={group_id} size={len(items)}"
                )
                latest = items[-1]
                if latest.pipeline_state is not None:
                    record_metric_event(
                        "group_flow_timing", group_id=group_id, user_id=latest.user_id,
                        stage="buffer_wait", action="completed",
                        elapsed_ms=int((time.monotonic() - latest.pipeline_state.received_monotonic) * 1000),
                        correlation_id=latest.correlation_id,
                    )
                self.state.generation_inflight.add(group_id)
                try:
                    with correlation_scope(latest.correlation_id):
                        await handle_group_message(latest.bot, latest.event, buffered_messages=items)
                finally:
                    self.state.generation_inflight.discard(group_id)
                    pending_size = len(self.state.message_buffers.get(group_id, []))
                    logger.info(
                        "qq_social_agent group generation finished: "
                        f"group={group_id} pending_buffer={pending_size}"
                    )
                    if pending_size:
                        should_reschedule = True
        finally:
            task = asyncio.current_task()
            if self.state.buffer_tasks.get(group_id) is task:
                self.state.buffer_tasks.pop(group_id, None)
            if should_reschedule and self.state.message_buffers.get(group_id):
                schedule_buffer_flush(group_id, delay=reschedule_delay)
