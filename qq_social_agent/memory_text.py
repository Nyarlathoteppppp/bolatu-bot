"""Shared memory text and JSON list conversion helpers."""
from __future__ import annotations

import json
import re


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _json_text_list(value: object, *, limit: int) -> list[str]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = []
    if not isinstance(raw, list):
        return []
    return _clean_text_list([str(item) for item in raw], limit=limit, item_limit=140)


def _clean_text_list(items: list[str], *, limit: int, item_limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text[:item_limit])
        if len(cleaned) >= limit:
            break
    return cleaned
