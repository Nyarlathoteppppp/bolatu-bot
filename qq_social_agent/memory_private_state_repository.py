"""PrivateStateRepository over the caller-owned SQLite connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time

from .memory_models import PrivateConversationState
from .memory_text import _json_text_list


class PrivateStateRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def private_conversation_state(self, chat_id: int) -> PrivateConversationState | None:
        row = self.conn.execute(
            "select * from private_conversation_states where chat_id = ?",
            (chat_id,),
        ).fetchone()
        return _private_conversation_state_from_row(row) if row is not None else None

    def recent_private_conversation_states(self, limit: int = 80) -> list[PrivateConversationState]:
        rows = self.conn.execute(
            "select * from private_conversation_states order by updated_at desc, chat_id desc limit ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [_private_conversation_state_from_row(row) for row in rows]

    def update_private_conversation_state(
        self,
        *,
        chat_id: int,
        user_id: int,
        display_name: str | None = None,
        relationship_note: str | None = None,
        interaction_tone: str | None = None,
        current_topic: str | None = None,
        open_threads: list[str] | tuple[str, ...] | None = None,
        frozen_fields: list[str] | tuple[str, ...] | None = None,
    ) -> PrivateConversationState:
        """Create or edit direct-chat state without changing factual memory atoms.

        ``None`` preserves a field; an empty string/list deliberately clears it.
        This makes correction in the WebUI predictable and auditable facts remain
        in ``memory_atoms`` instead of becoming mutable prompt text.
        """

        old = self.private_conversation_state(chat_id)
        now = time.time()
        clean_name = _clean_private_state_text(
            display_name if display_name is not None else (old.display_name if old else ""), 80
        )
        clean_relation = _clean_private_state_text(
            relationship_note if relationship_note is not None else (old.relationship_note if old else ""), 420
        )
        clean_tone = _clean_private_state_text(
            interaction_tone if interaction_tone is not None else (old.interaction_tone if old else ""), 240
        )
        clean_topic = _clean_private_state_text(
            current_topic if current_topic is not None else (old.current_topic if old else ""), 280
        )
        clean_threads = _clean_private_threads(
            open_threads if open_threads is not None else (old.open_threads if old else ())
        )
        clean_frozen = _clean_private_state_fields(
            frozen_fields if frozen_fields is not None else (old.frozen_fields if old else ())
        )
        self.conn.execute(
            """
            insert into private_conversation_states(
              chat_id, user_id, display_name, relationship_note, interaction_tone,
              current_topic, open_threads_json, frozen_fields_json, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(chat_id) do update set
              user_id = excluded.user_id,
              display_name = excluded.display_name,
              relationship_note = excluded.relationship_note,
              interaction_tone = excluded.interaction_tone,
              current_topic = excluded.current_topic,
              open_threads_json = excluded.open_threads_json,
              frozen_fields_json = excluded.frozen_fields_json,
              updated_at = excluded.updated_at
            """,
            (
                int(chat_id), int(user_id), clean_name, clean_relation, clean_tone,
                clean_topic, json.dumps(clean_threads, ensure_ascii=False),
                json.dumps(clean_frozen, ensure_ascii=False), now,
            ),
        )
        self.conn.commit()
        state = self.private_conversation_state(chat_id)
        if state is None:
            raise RuntimeError("private conversation state was not saved")
        return state

    def refresh_private_conversation_learning(
        self,
        *,
        chat_id: int,
        user_id: int,
        display_name: str,
        open_threads: list[str] | tuple[str, ...],
    ) -> PrivateConversationState:
        """Refresh short-term private state while honouring manual field locks."""

        old = self.private_conversation_state(chat_id)
        frozen = set(old.frozen_fields if old else ())
        learned_threads = _clean_private_threads(open_threads)
        return self.update_private_conversation_state(
            chat_id=chat_id,
            user_id=user_id,
            display_name=None if "display_name" in frozen else display_name,
            current_topic=None if "current_topic" in frozen else (learned_threads[0] if learned_threads else ""),
            open_threads=None if "open_threads" in frozen else learned_threads,
        )


def _clean_private_state_text(value: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _clean_private_threads(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_private_state_text(value, 280)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= 5:
            break
    return cleaned


def _clean_private_state_fields(values: object) -> list[str]:
    allowed = {"display_name", "relationship_note", "interaction_tone", "current_topic", "open_threads"}
    if not isinstance(values, (list, tuple)):
        return []
    return [field for field in values if isinstance(field, str) and field in allowed]


def _private_conversation_state_from_row(row: sqlite3.Row) -> PrivateConversationState:
    return PrivateConversationState(
        chat_id=int(row["chat_id"]),
        user_id=int(row["user_id"]),
        display_name=str(row["display_name"] or ""),
        relationship_note=str(row["relationship_note"] or ""),
        interaction_tone=str(row["interaction_tone"] or ""),
        current_topic=str(row["current_topic"] or ""),
        open_threads=tuple(_clean_private_threads(_json_text_list(row["open_threads_json"], limit=5))),
        frozen_fields=tuple(_clean_private_state_fields(_json_text_list(row["frozen_fields_json"], limit=5))),
        updated_at=float(row["updated_at"] or 0.0),
    )
