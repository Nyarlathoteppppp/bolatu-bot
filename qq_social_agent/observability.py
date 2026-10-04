from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator

# Preserve existing imports while implementations live in their owning modules.
from .trace_render import (
    render_trace_html,
    _render_trace_article,
    _render_trace_phase,
    _trace_status_css,
    _format_trace_time,
    _html_text,
)
from .trace_snapshot import (
    TRACE_STAGE_ORDER,
    DEFAULT_TRACE_LIMIT,
    MAX_TRACE_LIMIT,
    DEFAULT_TRACE_EVENTS_PER_STAGE,
    MAX_TRACE_EVENTS_PER_STAGE,
    DEFAULT_TRACE_INPUT_LIMIT,
    MAX_TRACE_INPUT_LIMIT,
    MAX_TRACE_ERRORS,
    MAX_TRACE_METADATA_ITEMS,
    MAX_TRACE_METADATA_LIST_ITEMS,
    MAX_TRACE_METADATA_DEPTH,
    MAX_TRACE_METADATA_STRING,
    _EVENT_TYPE_STAGES,
    _STAGE_ALIASES,
    _TRACE_METADATA_KEYS,
    _TOKEN_COUNT_KEYS,
    _SENSITIVE_KEY_PARTS,
    _BEARER_RE,
    _API_KEY_RE,
    _SECRET_ASSIGNMENT_RE,
    _LONG_IDENTIFIER_RE,
    _NormalizedTraceEvent,
    _normalized_elapsed_ms,
    build_trace_snapshot,
    _group_trace_events,
    trace_json_snapshot,
    sanitize_trace_metadata,
    _normalize_trace_event,
    _trace_from_events,
    _trace_phase,
    _trace_event_dict,
    _canonical_trace_stage,
    _onebot_api_stage,
    _trace_event_condition,
    _metadata_has_error,
    _phase_status,
    _trace_status,
    _trace_error_message,
    _event_elapsed_ms,
    _event_field,
    _metadata_mapping,
    _sanitize_trace_value,
    _sensitive_metadata_key,
    _sanitize_trace_string,
    _sanitize_url,
    _trace_identifier,
    _json_scalar,
    _compact_label,
    _coerce_float,
    _bounded_int,
)


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
    now = time.time()
    state = dict(_bot_connections.get(bot_key, {"bot_id": bot_key}))
    state.update(
        {
            "bot_id": bot_key,
            "connected": bot_key in _connected_bots,
            "last_seen_at": now,
            "last_inbound_event_at": now,
        }
    )
    _bot_connections[bot_key] = state


SEND_APIS = frozenset({"send_group_msg", "send_private_msg", "send_msg", "send_group_forward_msg", "send_private_forward_msg"})
SEND_HEALTH_TTL_SECONDS = 600.0


def _record_send_outcome(state: dict[str, Any], api: str, outcome: str, now: float, error: object = "") -> None:
    if api not in SEND_APIS:
        return
    channels = dict(state.get("send_channels") or {})
    channel = dict(channels.get(api) or {})
    channel.update(last_attempt_at=now, outcome=outcome)
    if outcome == "success":
        channel.update(last_success_at=now, consecutive_failures=0, last_error="")
    else:
        channel.update(last_failure_at=now, last_error=_error_summary(error),
                       consecutive_failures=int(channel.get("consecutive_failures") or 0) + 1)
    channels[api] = channel
    state["send_channels"] = channels


def delivery_health_snapshot(onebot: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    channels = {}
    for bot in onebot.get("bots", []):
        if not bot.get("connected"):
            continue
        observed = {"send_group_msg": {}, "send_private_msg": {}, **(bot.get("send_channels") or {})}
        for api, value in observed.items():
            entry = dict(value)
            outcome = value.get("outcome")
            if outcome is None:
                status = "unverified"
            elif outcome == "success":
                status = "verified" if now - value.get("last_success_at", 0) <= SEND_HEALTH_TTL_SECONDS else "unverified"
            else:
                status = "unknown" if outcome == "unknown" else "failed"
            entry["status"] = status
            channels[f"{bot.get('bot_id')}:{api}"] = entry
    failed = any(v["status"] == "failed" for v in channels.values())
    unknown = any(v["status"] == "unknown" for v in channels.values())
    return {"ok": not (failed or unknown), "status": "failed" if failed else "unknown" if unknown else
            "verified" if channels and all(v["status"] == "verified" for v in channels.values()) else "unverified",
            "channels": channels}


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
    _record_send_outcome(state, api, "success", now)
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
    unknown = timeout or isinstance(error, (TimeoutError, ConnectionError)) or "timeout" in str(error).lower()
    # Transport failures lack a QQ rejection; their delivery result is unknown.
    if not isinstance(error, str) and type(error).__name__ in {"NetworkError", "CancelledError"}:
        unknown = True
    _record_send_outcome(state, api, "unknown" if unknown else "failed", now, error)
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


def _error_summary(error: object) -> str:
    text = str(error or "").strip()
    return text[:200]
