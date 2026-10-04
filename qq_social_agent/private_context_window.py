"""Keep direct-chat turn context within the current conversation session."""

from __future__ import annotations

from .memory_models import ChatMessage


PRIVATE_SESSION_GAP_SECONDS = 12 * 60 * 60


def current_private_session_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    if not messages:
        return []
    for index in range(len(messages) - 1, 0, -1):
        if messages[index].created_at - messages[index - 1].created_at >= PRIVATE_SESSION_GAP_SECONDS:
            return messages[index:]
    return messages
