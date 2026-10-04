"""Shared row conversion, relevance and evidence-key helpers for repositories."""
from __future__ import annotations

import json
import re
import sqlite3

from .memory_models import ChatMessage


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    keys = set(row.keys())
    return ChatMessage(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        nickname=str(row["nickname"]),
        text=str(row["text"]),
        is_bot=bool(row["is_bot"]),
        created_at=float(row["created_at"]),
        id=int(row["id"]),
        source_message_id=str(row["source_message_id"] or "") if "source_message_id" in keys else "",
        session_id=str(row["session_id"] or "") if "session_id" in keys else "",
        message_segments_json=str(row["message_segments_json"] or "") if "message_segments_json" in keys else "",
        raw_message_json=str(row["raw_message_json"] or "") if "raw_message_json" in keys else "",
        sender_json=str(row["sender_json"] or "") if "sender_json" in keys else "",
    )


def _clamp_float(value: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _unique_recent_ints(values: tuple[int, ...] | list[int], *, limit: int | None = None) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed <= 0:
            continue
        if parsed in result:
            result.remove(parsed)
        result.append(parsed)
    if limit is not None and limit > 0:
        result = result[-limit:]
    return result


def _text_relevance_score(query: str, haystack: str) -> int:
    query_terms = _relevance_terms(query)
    if not query_terms:
        return 0
    haystack_lower = haystack.casefold()
    score = 0
    for term in query_terms:
        term_lower = term.casefold()
        if term_lower not in haystack_lower:
            continue
        score += 3 if len(term_lower) >= 4 else 1
    return score


def _relevance_terms(text: str) -> set[str]:
    lowered = text.casefold()
    terms = {
        match.group(0)
        for match in re.finditer(r"[a-z0-9_]{2,}", lowered)
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 8:
            terms.add(chunk)
        for size in (2, 3, 4):
            if len(chunk) < size:
                continue
            for index in range(0, len(chunk) - size + 1):
                terms.add(chunk[index : index + size])
    stop_terms = {
        "这个",
        "那个",
        "什么",
        "怎么",
        "就是",
        "然后",
        "可以",
        "不是",
        "没有",
        "一下",
        "感觉",
        "时候",
    }
    return {term for term in terms if term not in stop_terms}


def _loads_int_list(value: object) -> list[int]:
    try:
        raw = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def _source_message_key(source_message_id: int | str | None) -> str | None:
    if source_message_id is None:
        return None
    key = str(source_message_id).strip()
    return key or None
