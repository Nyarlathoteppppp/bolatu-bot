"""Existing linked-account identity mappings and expansion."""
from __future__ import annotations

from collections.abc import Iterable


PRIVATE_CHAT_ID_OFFSET = 10_000_000_000_000

KEDAI_PRIMARY_USER_ID = 3066256514

KEDAI_ALT_USER_ID = 2947279300

LINKED_ACCOUNT_GROUPS: tuple[frozenset[int], ...] = (
    frozenset({KEDAI_PRIMARY_USER_ID, KEDAI_ALT_USER_ID}),
)

def linked_account_ids(user_id: int | None) -> frozenset[int]:
    if user_id is None:
        return frozenset()
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return frozenset()
    if uid <= 0:
        return frozenset()
    for group in LINKED_ACCOUNT_GROUPS:
        if uid in group:
            return group
    return frozenset({uid})

def expand_linked_account_ids(user_ids: Iterable[int | None]) -> set[int]:
    expanded: set[int] = set()
    for user_id in user_ids:
        expanded.update(linked_account_ids(user_id))
    return {uid for uid in expanded if uid > 0}

def linked_account_note(user_id: int | None) -> str:
    uid = int(user_id or 0)
    if uid == KEDAI_PRIMARY_USER_ID:
        return "与 2947279300（纯真代代/科无代）是同一人的主号"
    if uid == KEDAI_ALT_USER_ID:
        return "与 3066256514（邪恶代代/科有代）是同一人的小号"
    return ""
