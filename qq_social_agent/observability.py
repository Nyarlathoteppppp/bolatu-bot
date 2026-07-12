from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator


_correlation_id: ContextVar[str] = ContextVar("qq_social_agent_correlation_id", default="")
_connected_bots: set[str] = set()
_bot_connections: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class Stopwatch:
    started_at: float

    @classmethod
    def start(cls) -> "Stopwatch":
        return cls(time.monotonic())

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)


def current_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> Token[str]:
    return _correlation_id.set(correlation_id)


def reset_correlation_id(token: Token[str]) -> None:
    _correlation_id.reset(token)


@contextmanager
def correlation_scope(correlation_id: str) -> Iterator[str]:
    """Temporarily bind a correlation id and always restore the previous value."""

    token = set_correlation_id(correlation_id)
    try:
        yield correlation_id
    finally:
        reset_correlation_id(token)


def event_correlation_id(event: Any, *, scope: str) -> str:
    message_id = getattr(event, "message_id", None) or ""
    group_id = getattr(event, "group_id", None) or ""
    user_id = getattr(event, "user_id", None) or ""
    timestamp = getattr(event, "time", None) or ""
    if message_id:
        return f"{scope}:{group_id}:{message_id}"
    seed = ":".join(str(part) for part in (scope, group_id, user_id, timestamp) if part != "")
    suffix = uuid.uuid4().hex[:8]
    return f"{seed}:{suffix}" if seed else f"{scope}:{suffix}"


def mark_bot_connected(bot_id: int | str) -> None:
    bot_key = str(bot_id)
    now = time.time()
    _connected_bots.add(bot_key)
    state = dict(_bot_connections.get(bot_key, {}))
    state.update(
        {
            "bot_id": bot_key,
            "connected": True,
            "last_connected_at": now,
            "last_seen_at": now,
            "connection_count": int(state.get("connection_count") or 0) + 1,
            "consecutive_api_errors": 0,
        }
    )
    _bot_connections[bot_key] = state


def mark_bot_seen(bot_id: int | str) -> None:
    """Record activity observed from a bot without changing connection ownership."""

    bot_key = str(bot_id)
    state = dict(_bot_connections.get(bot_key, {"bot_id": bot_key}))
    state.update(
        {
            "bot_id": bot_key,
            "connected": bot_key in _connected_bots,
            "last_seen_at": time.time(),
        }
    )
    _bot_connections[bot_key] = state


def mark_onebot_api_success(
    bot_id: int | str,
    api: str,
    *,
    elapsed_ms: int | float | None = None,
) -> None:
    """Record a successful OneBot API call and refresh bot activity."""

    bot_key = str(bot_id)
    now = time.time()
    state = dict(_bot_connections.get(bot_key, {"bot_id": bot_key}))
    state.update(
        {
            "bot_id": bot_key,
            "connected": bot_key in _connected_bots,
            "last_seen_at": now,
            "last_api_at": now,
            "last_api_success_at": now,
            "last_api_name": str(api).strip(),
            "last_api_outcome": "success",
            "last_api_elapsed_ms": _normalized_elapsed_ms(elapsed_ms),
            "api_call_count": int(state.get("api_call_count") or 0) + 1,
            "api_success_count": int(state.get("api_success_count") or 0) + 1,
            "consecutive_api_errors": 0,
        }
    )
    _bot_connections[bot_key] = state


def mark_onebot_api_error(
    bot_id: int | str,
    api: str,
    *,
    elapsed_ms: int | float | None = None,
    error: object = "",
    timeout: bool = False,
) -> None:
    """Record a failed OneBot API call without treating the attempt as bot activity."""

    bot_key = str(bot_id)
    now = time.time()
    state = dict(_bot_connections.get(bot_key, {"bot_id": bot_key}))
    state.update(
        {
            "bot_id": bot_key,
            "connected": bot_key in _connected_bots,
            "last_api_at": now,
            "last_api_error_at": now,
            "last_api_name": str(api).strip(),
            "last_api_outcome": "timeout" if timeout else "error",
            "last_api_elapsed_ms": _normalized_elapsed_ms(elapsed_ms),
            "last_api_error": _error_summary(error),
            "last_api_error_type": type(error).__name__ if not isinstance(error, str) else "",
            "api_call_count": int(state.get("api_call_count") or 0) + 1,
            "api_error_count": int(state.get("api_error_count") or 0) + 1,
            "api_timeout_count": int(state.get("api_timeout_count") or 0) + (1 if timeout else 0),
            "consecutive_api_errors": int(state.get("consecutive_api_errors") or 0) + 1,
        }
    )
    _bot_connections[bot_key] = state


def mark_bot_disconnected(bot_id: int | str) -> None:
    bot_key = str(bot_id)
    now = time.time()
    _connected_bots.discard(bot_key)
    state = dict(_bot_connections.get(bot_key, {"bot_id": bot_key}))
    state.update(
        {
            "connected": False,
            "last_disconnected_at": now,
            "last_seen_at": now,
            "disconnect_count": int(state.get("disconnect_count") or 0) + 1,
        }
    )
    _bot_connections[bot_key] = state


def onebot_status_snapshot() -> dict[str, Any]:
    return {
        "connected_bots": sorted(_connected_bots),
        "bots": [dict(_bot_connections[key]) for key in sorted(_bot_connections)],
    }


def readiness_snapshot(*, deepseek_ready: bool, db_ready: bool = True) -> dict[str, Any]:
    return {
        "ok": bool(deepseek_ready and db_ready and _connected_bots),
        "deepseek_ready": bool(deepseek_ready),
        "db_ready": bool(db_ready),
        "connected_bots": sorted(_connected_bots),
    }


def _normalized_elapsed_ms(value: int | float | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _error_summary(error: object) -> str:
    text = str(error or "").strip()
    return text[:200]
