"""Choose reply evidence from a larger history without changing speaking decisions."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .memory import ChatMessage


_TERMS = re.compile(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]+", re.IGNORECASE)
_FILLER = {"这个", "那个", "什么", "怎么", "你们", "我们", "就是", "还是"}


def _key(message: ChatMessage) -> tuple[object, ...]:
    if message.id:
        return ("db", message.group_id, message.id)
    if message.source_message_id:
        return ("source", message.group_id, message.source_message_id)
    return ("text", message.group_id, message.user_id, message.created_at, message.text)


def _pinned(message: ChatMessage, source_ids: set[str], db_ids: set[int]) -> bool:
    return message.source_message_id in source_ids or message.id in db_ids


def omitted_message_counts(
    original: Sequence[ChatMessage], selected: Sequence[ChatMessage],
) -> list[int]:
    """Count actual unselected rows between retained messages, never DB ID gaps."""
    positions = {_key(message): index for index, message in enumerate(original)}
    counts: list[int] = []
    previous: int | None = None
    for message in selected:
        position = positions.get(_key(message))
        counts.append(max(0, position - previous - 1) if position is not None and previous is not None else 0)
        previous = position
    return counts


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _TERMS.findall(text.casefold()):
        if len(match) < 2:
            continue
        if "\u4e00" <= match[0] <= "\u9fff":
            terms.update(match[index:index + 2] for index in range(len(match) - 1))
        else:
            terms.add(match)
    return terms - _FILLER


def shortlist_older_messages(
    older: Sequence[ChatMessage],
    *,
    current_text: str,
    pinned_source_ids: set[str],
    pinned_db_ids: set[int],
    limit: int = 32,
) -> list[ChatMessage]:
    """Keep exact evidence, nearby turns, then topic overlap within JEV's 32 questions."""
    if len(older) <= limit:
        return list(older)
    pinned = [index for index, message in enumerate(older) if _pinned(message, pinned_source_ids, pinned_db_ids)]
    chosen = set(pinned[-limit:])
    nearby_slots = min(limit // 2, limit - len(chosen))
    for index in range(len(older) - 1, -1, -1):
        if nearby_slots <= 0:
            break
        if index not in chosen:
            chosen.add(index)
            nearby_slots -= 1
    query_terms = _terms(current_text)
    ranked = sorted(
        (index for index in range(len(older)) if index not in chosen),
        key=lambda index: (
            len(query_terms & _terms(older[index].text)),
            index,
        ),
        reverse=True,
    )
    chosen.update(ranked[: limit - len(chosen)])
    return [older[index] for index in sorted(chosen)]


def select_generation_messages(
    messages: Sequence[ChatMessage],
    *,
    candidates: Sequence[ChatMessage],
    scores: Sequence[float] | None,
    pinned_source_ids: set[str],
    pinned_db_ids: set[int],
    keep_recent: int = 6,
    max_total: int = 12,
) -> list[ChatMessage]:
    """Keep continuous turns and exact evidence; add only positively ranked history."""
    recent = list(messages[-keep_recent:]) if keep_recent else []
    fixed = [message for message in messages if _pinned(message, pinned_source_ids, pinned_db_ids)]
    selected: dict[tuple[object, ...], ChatMessage] = {
        _key(message): message for message in [*recent, *fixed]
    }
    if scores is None or len(scores) != len(candidates):
        for message in reversed(messages[:-keep_recent] if keep_recent else messages):
            if len(selected) >= max_total:
                break
            selected.setdefault(_key(message), message)
        return sorted(selected.values(), key=lambda message: (message.created_at, message.id))
    slots = max(0, max_total - len(selected))
    if slots:
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (float(scores[item[0]]), item[0]),
            reverse=True,
        )
        for index, message in ranked:
            if float(scores[index]) < 0.28:
                break
            selected.setdefault(_key(message), message)
            if len(selected) >= max_total:
                break
    return sorted(selected.values(), key=lambda message: (message.created_at, message.id))
