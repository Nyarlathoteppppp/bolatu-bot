from __future__ import annotations

import math
from typing import Any

RESOLVED = "RESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"
NONE = "NONE"
AMBIGUOUS = "AMBIGUOUS"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"

UNRESOLVED_STATUSES = frozenset({AMBIGUOUS, UNAVAILABLE, ERROR})
SOURCE_RULE = "rule"
SOURCE_JEV = "jev"
SOURCE_REPAIR = "repair"


def choice_confidence(answers: Any, key: str) -> float | None:
    """Return Jev's native Choice confidence, rejecting malformed values."""
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    if not isinstance(answer, dict):
        return None
    value = answer.get("confidence")
    if isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def joint_choice_confidence(answers: Any, *keys: str) -> float | None:
    """Use the weakest native confidence when a decision needs multiple Choices."""
    values = [choice_confidence(answers, key) for key in keys]
    if not values or any(value is None for value in values):
        return None
    return min(value for value in values if value is not None)


def has_usable_value(obj) -> bool:
    status = str(getattr(obj, "status", "") or "")
    return status == RESOLVED


def is_unresolved(obj) -> bool:
    return str(getattr(obj, "status", "") or "") in UNRESOLVED_STATUSES


def _has_field(obj, name: str) -> bool:
    return name in getattr(obj, "__dataclass_fields__", {}) or hasattr(obj, name)


def finalize_result(obj, *, has_value: bool) -> None:
    status = str(getattr(obj, "status", "") or "")
    unresolved = bool(getattr(obj, "unresolved", False))
    if not status:
        if unresolved:
            status = AMBIGUOUS
        elif has_value:
            status = RESOLVED
        else:
            status = NOT_APPLICABLE
    object.__setattr__(obj, "status", status)
    if _has_field(obj, "unresolved"):
        object.__setattr__(obj, "unresolved", status in UNRESOLVED_STATUSES)
    source = str(getattr(obj, "source", "") or "")
    if not source and _has_field(obj, "source"):
        object.__setattr__(obj, "source", SOURCE_RULE)
    value = str(getattr(obj, "value", "") or "")
    if not value and _has_field(obj, "value"):
        inferred = _infer_value(obj)
        if inferred:
            object.__setattr__(obj, "value", inferred)


def _infer_value(obj) -> str:
    user_ids = getattr(obj, "user_ids", ())
    if user_ids:
        return ",".join(str(uid) for uid in user_ids)
    source_key = str(getattr(obj, "source_key", "") or "")
    if source_key:
        return source_key
    target_key = str(getattr(obj, "target_key", "") or "")
    if target_key:
        return target_key
    action = str(getattr(obj, "action", "") or "")
    if action and action not in {"", "IGNORE", "none"}:
        target_atom_id = getattr(obj, "target_atom_id", None)
        return f"{action}:{target_atom_id}" if target_atom_id else action
    kind = str(getattr(obj, "kind", "") or "")
    if kind and kind not in {"", "NONE"}:
        return kind
    return ""
