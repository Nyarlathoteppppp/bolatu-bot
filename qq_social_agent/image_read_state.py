"""Source-linked image observations and shared recognition jobs.

The message exists before recognition starts. Results enrich that same message;
they never become a new user turn or change its arrival time.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable

from .media_context import ImageOcrContext, ImageOcrService, collect_ocr_image_segments
from .resolver_result import RESOLVED

if TYPE_CHECKING:
    from .ellipsis_resolver import EllipsisResolution
    from .memory import ChatMessage


def stored_image_segments(segments: str | None) -> list[dict[str, Any]]:
    return collect_ocr_image_segments(json.loads(segments or "[]"))


def resolved_image_message(ellipsis: EllipsisResolution, messages: Iterable[ChatMessage]) -> ChatMessage | None:
    """Consume the resolver's binding, without selecting another candidate."""
    if ellipsis.status != RESOLVED:
        return None
    for message in messages:
        source = str(getattr(message, "source_message_id", "") or "")
        key = f"s{source}" if source else f"m{getattr(message, 'id', 0)}"
        if ((source and source == ellipsis.source_message_id)
                or key == ellipsis.source_key):
            if stored_image_segments(getattr(message, "message_segments_json", "")):
                return message
    return None


class ImageReadStateStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._tasks: dict[int, asyncio.Task[ImageOcrContext]] = {}
        conn.execute("""
            create table if not exists image_read_states (
                message_id integer primary key,
                status text not null,
                text text not null default '',
                image_count integer not null,
                ocr_count integer not null default 0,
                reason text not null default ''
            )
        """)
        conn.commit()

    def observe(self, message_id: int, segments: str) -> None:
        images = stored_image_segments(segments)
        if images:
            self.conn.execute("""
                insert or ignore into image_read_states(message_id, status, image_count)
                values (?, 'received', ?)
            """, (message_id, len(images)))

    def skipped(self, message_id: int, reason: str) -> None:
        self.conn.execute("""
            update image_read_states set status = 'skipped', reason = ?
            where message_id = ? and status in ('received', 'skipped')
        """, (reason, message_id))
        self.conn.commit()

    async def read(self, bot: Any, service: ImageOcrService, row: sqlite3.Row) -> ImageOcrContext:
        message_id = int(row["id"])
        images = stored_image_segments(row["message_segments_json"])
        if not images:
            return ImageOcrContext("", 0, 0)
        task = self._tasks.get(message_id)
        if task is None:
            state = self.conn.execute(
                "select * from image_read_states where message_id = ?", (message_id,)
            ).fetchone()
            if (state is not None and state["status"] == "ready"
                    and state["ocr_count"] >= min(state["image_count"], service.max_images_per_message)):
                return ImageOcrContext(state["text"], state["image_count"], state["ocr_count"])
            self.observe(message_id, row["message_segments_json"])
            task = asyncio.create_task(self._read(bot, service, message_id, images))
            self._tasks[message_id] = task
        # Cancelling one incoming message must not cancel recognition used by
        # the original image or another follow-up.
        return await asyncio.shield(task)

    async def _read(self, bot: Any, service: ImageOcrService, message_id: int, images: list[dict[str, Any]]) -> ImageOcrContext:
        self.conn.commit()
        try:
            event = SimpleNamespace(message=[{"type": "image", "data": data} for data in images])
            result = await service.context_for_event(bot, event)
            result = replace(result, image_count=len(images))
            self.conn.execute("""
                update image_read_states set status = ?, text = ?, image_count = ?,
                    ocr_count = ?, reason = ? where message_id = ?
            """, ("ready" if result.text else "unavailable", result.text,
                  result.image_count, result.ocr_count, result.skipped_reason, message_id))
            return result
        except Exception:
            self.conn.execute("update image_read_states set status = 'unavailable', reason = 'error' where message_id = ?", (message_id,))
            raise
        finally:
            self.conn.commit()
            self._tasks.pop(message_id, None)

    def enrich(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            return messages
        ids = [message.id for message in messages]
        placeholders = ",".join("?" for _ in ids)
        states = {int(row["message_id"]): row for row in self.conn.execute(
            f"""select states.*, messages.text as message_text
                from image_read_states states join messages on messages.id = states.message_id
                where states.message_id in ({placeholders})""", ids
        )}
        enriched = []
        for message in messages:
            state = states.get(message.id)
            if state is None:
                enriched.append(message)
                continue
            base_text = str(state["message_text"])
            content_in_message = bool(state["text"] and state["text"] in base_text)
            status = "reading" if message.id in self._tasks else state["status"]
            data = {
                "source_message_id": message.source_message_id,
                "image_count": state["image_count"],
                "recognized_image_count": state["ocr_count"],
                "status": status,
                "observation": {
                    "received": "图片已收到，尚无识别结果",
                    "reading": "图片已收到，正在识别，内容尚未获得",
                    "ready": "图片已收到，以下为识别结果",
                    "unavailable": "图片已收到，但本次识别没有获得内容",
                    "skipped": "图片已收到，本轮未识别",
                }[status],
                "recognized_content": "" if content_in_message else state["text"],
                "recognized_content_in_message": content_in_message,
                "reason": state["reason"],
            }
            context = "[图片接收与识别状态] " + json.dumps(data, ensure_ascii=False)
            enriched.append(replace(message, text=f"{base_text}\n{context}"))
        return enriched

    def clear_group(self, group_id: int) -> None:
        ids = [int(row[0]) for row in self.conn.execute("select id from messages where group_id = ?", (group_id,))]
        for message_id in ids:
            task = self._tasks.pop(message_id, None)
            if task is not None:
                task.cancel()
        self.conn.execute("delete from image_read_states where message_id in (select id from messages where group_id = ?)", (group_id,))

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
