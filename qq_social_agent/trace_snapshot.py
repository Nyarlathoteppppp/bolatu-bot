"""Trace event normalization, aggregation and metadata redaction."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, replace
from itertools import islice
from typing import Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACE_STAGE_ORDER = (
    "receive",
    "ocr",
    "history",
    "buffer",
    "rag",
    "decision",
    "search/tool",
    "generation",
    "approval",
    "send",
)
DEFAULT_TRACE_LIMIT = 50
MAX_TRACE_LIMIT = 200
DEFAULT_TRACE_EVENTS_PER_STAGE = 6
MAX_TRACE_EVENTS_PER_STAGE = 20
DEFAULT_TRACE_INPUT_LIMIT = 5_000
MAX_TRACE_INPUT_LIMIT = 20_000
MAX_TRACE_ERRORS = 8
MAX_TRACE_METADATA_ITEMS = 24
MAX_TRACE_METADATA_LIST_ITEMS = 12
MAX_TRACE_METADATA_DEPTH = 3
MAX_TRACE_METADATA_STRING = 240
_EVENT_TYPE_STAGES = {
    "message_received": "receive",
    "group_gate": "decision",
    "message_duplicate": "receive",
    "image_ocr": "ocr",
    "history_backfill": "history",
    "reply_reference": "history",
    "message_buffered": "buffer",
    "rag_retrieval": "rag",
    "rag_knowledge_ingested": "rag",
    "decision_start": "decision",
    "decision_result": "decision",
    "suppression": "decision",
    "tool_call": "search/tool",
    "search": "search/tool",
    "candidate_generated": "generation",
    "reply_generated": "generation",
    "message_sent": "send",
    "social_action": "send",
}
_STAGE_ALIASES = {
    "receive": "receive",
    "received": "receive",
    "inbound": "receive",
    "ingress": "receive",
    "ocr": "ocr",
    "image_ocr": "ocr",
    "media_context": "ocr",
    "media": "ocr",
    "history": "history",
    "history_backfill": "history",
    "reply_reference": "history",
    "buffer": "buffer",
    "buffered": "buffer",
    "buffer_wait": "buffer",
    "lock_wait": "buffer",
    "locked": "buffer",
    "rag": "rag",
    "rag_retrieval": "rag",
    "retrieval": "rag",
    "decision": "decision",
    "llm_decision": "decision",
    "decision_gate": "decision",
    "search": "search/tool",
    "tool": "search/tool",
    "search/tool": "search/tool",
    "fresh_context": "search/tool",
    "market": "search/tool",
    "generation": "generation",
    "generate": "generation",
    "reply_direct": "generation",
    "reply_candidates": "generation",
    "approval": "approval",
    "review": "approval",
    "review_disabled": "approval",
    "probability": "approval",
    "send": "send",
    "sent": "send",
}
_TRACE_METADATA_KEYS = (
    "correlation_id",
    "message_id",
    "source_message_id",
    "elapsed_ms",
    "latency_ms",
    "duration_ms",
    "flow_elapsed_ms",
    "receive_elapsed_ms",
    "approval_wait_ms",
    "error",
    "error_type",
    "status",
    "outcome",
    "success",
    "api",
    "api_name",
)
_TOKEN_COUNT_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "max_tokens",
    "token_count",
}
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "authorization",
    "password",
    "passwd",
    "credential",
    "private_key",
    "secret",
    "cookie",
    "signature",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+\-/]+=*")
_API_KEY_RE = re.compile(r"(?<![a-z0-9])sk-[a-z0-9_-]{8,}", re.IGNORECASE)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization|cookie)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)
_LONG_IDENTIFIER_RE = re.compile(r"(?<![\d.])\d{7,18}(?![\d.])")


@dataclass(frozen=True)
class _NormalizedTraceEvent:
    trace_id: str
    correlation_id: str
    message_id: str
    event_type: str
    raw_stage: str
    stage: str
    action: str
    group_id: object
    user_id: object
    metadata: dict[str, object]
    created_at: float
    elapsed_ms: int | None
    flow_elapsed_ms: int | None
    input_index: int


def _normalized_elapsed_ms(value: int | float | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def build_trace_snapshot(
    events: Iterable[object],
    *,
    limit: int = DEFAULT_TRACE_LIMIT,
    events_per_stage: int = DEFAULT_TRACE_EVENTS_PER_STAGE,
    input_limit: int = DEFAULT_TRACE_INPUT_LIMIT,
    generated_at: float | None = None,
) -> dict[str, object]:
    """Build a bounded, JSON-serializable trace snapshot from metric-like events.

    Inputs may be ``BotMetricEvent`` objects, sqlite rows, or dictionaries using
    the common ``event_type/stage/action/metadata/created_at`` fields. Events
    without a correlation id or message id, and events that cannot be mapped to
    the fixed trace stages, are counted and omitted.
    """

    trace_limit = _bounded_int(limit, default=DEFAULT_TRACE_LIMIT, minimum=1, maximum=MAX_TRACE_LIMIT)
    stage_event_limit = _bounded_int(
        events_per_stage,
        default=DEFAULT_TRACE_EVENTS_PER_STAGE,
        minimum=1,
        maximum=MAX_TRACE_EVENTS_PER_STAGE,
    )
    source_limit = _bounded_int(
        input_limit,
        default=DEFAULT_TRACE_INPUT_LIMIT,
        minimum=1,
        maximum=MAX_TRACE_INPUT_LIMIT,
    )
    sampled = list(islice(iter(events), source_limit + 1))
    input_truncated = len(sampled) > source_limit
    if input_truncated:
        sampled = sampled[:source_limit]

    normalized_events: list[_NormalizedTraceEvent] = []
    dropped_untraceable = 0
    dropped_unmapped = 0
    for index, raw_event in enumerate(sampled):
        normalized, drop_reason = _normalize_trace_event(raw_event, input_index=index)
        if normalized is None:
            if drop_reason == "unmapped_stage":
                dropped_unmapped += 1
            else:
                dropped_untraceable += 1
            continue
        normalized_events.append(normalized)

    grouped = _group_trace_events(normalized_events)

    traces = [
        _trace_from_events(trace_events, events_per_stage=stage_event_limit)
        for trace_events in grouped.values()
    ]
    traces.sort(
        key=lambda trace: (
            _coerce_float(trace.get("last_event_at"), 0.0),
            str(trace.get("trace_id") or ""),
        ),
        reverse=True,
    )
    available_trace_count = len(traces)
    traces = traces[:trace_limit]
    omitted_trace_count = max(0, available_trace_count - len(traces))
    omitted_stage_event_count = sum(int(trace.get("omitted_event_count") or 0) for trace in traces)
    snapshot_time = _coerce_float(generated_at, time.time()) if generated_at is not None else time.time()
    return {
        "schema_version": 2,
        "generated_at": snapshot_time,
        "stage_order": list(TRACE_STAGE_ORDER),
        "source_event_count": len(sampled),
        "available_trace_count": available_trace_count,
        "trace_count": len(traces),
        "omitted_trace_count": omitted_trace_count,
        "dropped_untraceable_event_count": dropped_untraceable,
        "dropped_unmapped_event_count": dropped_unmapped,
        "omitted_stage_event_count": omitted_stage_event_count,
        "input_truncated": input_truncated,
        "truncated": bool(input_truncated or omitted_trace_count or omitted_stage_event_count),
        "limits": {
            "traces": trace_limit,
            "events_per_stage": stage_event_limit,
            "input_events": source_limit,
        },
        "traces": traces,
    }


def _group_trace_events(events: list[_NormalizedTraceEvent]) -> dict[str, list[_NormalizedTraceEvent]]:
    """Merge message-id-only events into an unambiguous correlation trace."""

    precise_candidates: dict[tuple[str, str], set[str]] = {}
    global_candidates: dict[str, set[str]] = {}
    for event in events:
        if not event.correlation_id or not event.message_id:
            continue
        group_key = _trace_identifier(event.group_id) or "-"
        precise_candidates.setdefault((group_key, event.message_id), set()).add(event.correlation_id)
        global_candidates.setdefault(event.message_id, set()).add(event.correlation_id)

    precise_aliases = {
        key: next(iter(values)) for key, values in precise_candidates.items() if len(values) == 1
    }
    global_aliases = {
        key: next(iter(values)) for key, values in global_candidates.items() if len(values) == 1
    }
    grouped: dict[str, list[_NormalizedTraceEvent]] = {}
    for event in events:
        trace_id = event.trace_id
        if not event.correlation_id and event.message_id:
            group_key = _trace_identifier(event.group_id) or "-"
            trace_id = precise_aliases.get(
                (group_key, event.message_id),
                global_aliases.get(event.message_id, trace_id),
            )
        normalized = event if trace_id == event.trace_id else replace(event, trace_id=trace_id)
        grouped.setdefault(trace_id, []).append(normalized)
    return grouped


def trace_json_snapshot(
    events: Iterable[object],
    *,
    limit: int = DEFAULT_TRACE_LIMIT,
    events_per_stage: int = DEFAULT_TRACE_EVENTS_PER_STAGE,
    input_limit: int = DEFAULT_TRACE_INPUT_LIMIT,
    generated_at: float | None = None,
) -> dict[str, object]:
    """Compatibility-friendly name for the JSON-ready trace snapshot builder."""

    return build_trace_snapshot(
        events,
        limit=limit,
        events_per_stage=events_per_stage,
        input_limit=input_limit,
        generated_at=generated_at,
    )


def sanitize_trace_metadata(metadata: object) -> dict[str, object]:
    """Bound and redact untrusted event metadata for JSON and HTML output."""

    raw = _metadata_mapping(metadata)
    sanitized = _sanitize_trace_value(raw, depth=0)
    return sanitized if isinstance(sanitized, dict) else {}


def _normalize_trace_event(
    raw_event: object,
    *,
    input_index: int,
) -> tuple[_NormalizedTraceEvent | None, str]:
    metadata = _metadata_mapping(_event_field(raw_event, "metadata", None))
    if not metadata:
        metadata = _metadata_mapping(_event_field(raw_event, "metadata_json", None))
    for key in _TRACE_METADATA_KEYS:
        value = _event_field(raw_event, key, None)
        if value is not None and key not in metadata:
            metadata[key] = value

    correlation_id = _trace_identifier(
        _event_field(raw_event, "correlation_id", None) or metadata.get("correlation_id")
    )
    message_id = _trace_identifier(
        _event_field(raw_event, "message_id", None)
        or _event_field(raw_event, "source_message_id", None)
        or metadata.get("message_id")
        or metadata.get("source_message_id")
    )
    group_id = _event_field(raw_event, "group_id", None)
    user_id = _event_field(raw_event, "user_id", None)
    if correlation_id:
        trace_id = correlation_id
    elif message_id:
        group_key = _trace_identifier(group_id) or "-"
        trace_id = f"message:{group_key}:{message_id}"
    else:
        return None, "untraceable"

    event_type = _compact_label(_event_field(raw_event, "event_type", "unknown"), limit=80) or "unknown"
    raw_stage = _compact_label(_event_field(raw_event, "stage", ""), limit=80)
    action = _compact_label(_event_field(raw_event, "action", ""), limit=80)
    stage = _canonical_trace_stage(event_type, raw_stage, action, metadata)
    if stage is None:
        return None, "unmapped_stage"
    created_at = _coerce_float(
        _event_field(raw_event, "created_at", None)
        or _event_field(raw_event, "timestamp", None)
        or _event_field(raw_event, "time", None),
        0.0,
    )
    elapsed_ms = _event_elapsed_ms(metadata, stage=stage)
    flow_elapsed_ms = _normalized_elapsed_ms(metadata.get("flow_elapsed_ms"))
    return (
        _NormalizedTraceEvent(
            trace_id=trace_id,
            correlation_id=correlation_id,
            message_id=message_id,
            event_type=event_type,
            raw_stage=raw_stage,
            stage=stage,
            action=action,
            group_id=_json_scalar(group_id),
            user_id=_json_scalar(user_id),
            metadata=sanitize_trace_metadata(metadata),
            created_at=max(0.0, created_at),
            elapsed_ms=elapsed_ms,
            flow_elapsed_ms=flow_elapsed_ms,
            input_index=input_index,
        ),
        "",
    )


def _trace_from_events(
    events: list[_NormalizedTraceEvent],
    *,
    events_per_stage: int,
) -> dict[str, object]:
    ordered = sorted(events, key=lambda event: (event.created_at, event.input_index))
    phases: list[dict[str, object]] = []
    displayed_event_count = 0
    errors: list[dict[str, object]] = []
    for stage in TRACE_STAGE_ORDER:
        stage_events = [event for event in ordered if event.stage == stage]
        phase = _trace_phase(stage, stage_events, events_per_stage=events_per_stage)
        displayed_event_count += int(phase["displayed_event_count"])
        phases.append(phase)
        for event in stage_events:
            condition = _trace_event_condition(event)
            if condition not in {"error", "timeout"} or len(errors) >= MAX_TRACE_ERRORS:
                continue
            errors.append(
                {
                    "stage": stage,
                    "event_type": event.event_type,
                    "action": event.action,
                    "status": condition,
                    "message": _trace_error_message(event),
                    "created_at": event.created_at,
                }
            )

    first_at = min((event.created_at for event in ordered), default=0.0)
    last_at = max((event.created_at for event in ordered), default=0.0)
    wall_total_ms = max(0, int(round((last_at - first_at) * 1000))) if last_at >= first_at else 0
    explicit_total_ms = max((event.flow_elapsed_ms or 0 for event in ordered), default=0)
    total_duration_ms = max(wall_total_ms, explicit_total_ms)
    phase_by_stage = {str(phase["stage"]): phase for phase in phases}
    trace_status = _trace_status(ordered, phase_by_stage)
    correlation_id = next((event.correlation_id for event in ordered if event.correlation_id), "")
    message_id = next((event.message_id for event in ordered if event.message_id), "")
    group_id = next((event.group_id for event in ordered if event.group_id is not None), None)
    user_id = next((event.user_id for event in ordered if event.user_id is not None), None)
    return {
        "trace_id": ordered[0].trace_id,
        "correlation_id": correlation_id,
        "message_id": message_id,
        "group_id": group_id,
        "user_id": user_id,
        "status": trace_status,
        "first_event_at": first_at,
        "last_event_at": last_at,
        "total_duration_ms": total_duration_ms,
        "event_count": len(ordered),
        "displayed_event_count": displayed_event_count,
        "omitted_event_count": max(0, len(ordered) - displayed_event_count),
        "error_count": sum(
            1 for event in ordered if _trace_event_condition(event) in {"error", "timeout"}
        ),
        "errors": errors,
        "phases": phases,
    }


def _trace_phase(
    stage: str,
    events: list[_NormalizedTraceEvent],
    *,
    events_per_stage: int,
) -> dict[str, object]:
    if not events:
        return {
            "stage": stage,
            "status": "not_reached",
            "started_at": None,
            "ended_at": None,
            "duration_ms": None,
            "event_count": 0,
            "displayed_event_count": 0,
            "omitted_event_count": 0,
            "events": [],
        }
    ordered = sorted(events, key=lambda event: (event.created_at, event.input_index))
    first_at = ordered[0].created_at
    last_at = ordered[-1].created_at
    wall_ms = max(0, int(round((last_at - first_at) * 1000)))
    explicit_ms = max((event.elapsed_ms or 0 for event in ordered), default=0)
    duration_ms = max(wall_ms, explicit_ms)
    displayed = ordered[-events_per_stage:]
    return {
        "stage": stage,
        "status": _phase_status(ordered),
        "started_at": first_at,
        "ended_at": last_at,
        "duration_ms": duration_ms,
        "event_count": len(ordered),
        "displayed_event_count": len(displayed),
        "omitted_event_count": max(0, len(ordered) - len(displayed)),
        "events": [_trace_event_dict(event) for event in displayed],
    }


def _trace_event_dict(event: _NormalizedTraceEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type,
        "raw_stage": event.raw_stage,
        "action": event.action,
        "status": _trace_event_condition(event),
        "created_at": event.created_at,
        "elapsed_ms": event.elapsed_ms,
        "metadata": event.metadata,
    }


def _canonical_trace_stage(
    event_type: str,
    raw_stage: str,
    action: str,
    metadata: Mapping[str, object],
) -> str | None:
    event_key = event_type.casefold().strip()
    if event_key.startswith("approval_"):
        return "approval"
    mapped = _EVENT_TYPE_STAGES.get(event_key)
    if mapped:
        return mapped
    if event_key in {"onebot_api", "onebot_call"}:
        api = str(metadata.get("api") or metadata.get("api_name") or action).casefold()
        return _onebot_api_stage(api)
    stage_key = raw_stage.casefold().strip().replace("-", "_").replace(" ", "_")
    if stage_key in _STAGE_ALIASES:
        return _STAGE_ALIASES[stage_key]
    if "approval" in event_key:
        return "approval"
    if "candidate" in event_key or "generation" in event_key:
        return "generation"
    if event_key.startswith("send_") or event_key.endswith("_sent"):
        return "send"
    if "search" in event_key or "tool" in event_key:
        return "search/tool"
    if "decision" in event_key:
        return "decision"
    if "buffer" in event_key:
        return "buffer"
    if "history" in event_key or "reference" in event_key:
        return "history"
    if "ocr" in event_key:
        return "ocr"
    if "receive" in event_key or "inbound" in event_key:
        return "receive"
    return None


def _onebot_api_stage(api: str) -> str | None:
    if not api:
        return None
    if "ocr" in api or api == "get_image":
        return "ocr"
    if api in {"get_msg", "get_forward_msg", "get_group_msg_history", "get_friend_msg_history", "get_file"}:
        return "history"
    if api.startswith("send_") or api in {"set_msg_emoji_like", "send_poke", "group_poke"}:
        return "send"
    return None


def _trace_event_condition(event: _NormalizedTraceEvent) -> str:
    tokens = " ".join(
        (
            event.event_type,
            event.raw_stage,
            event.action,
            str(event.metadata.get("status") or ""),
            str(event.metadata.get("outcome") or ""),
        )
    ).casefold()
    if "timeout" in tokens or "timed_out" in tokens:
        return "timeout"
    if (
        _metadata_has_error(event.metadata)
        or re.search(r"(?:^|[_\s-])(failed|failure|error)(?:$|[_\s-])", tokens)
        or event.event_type.casefold().endswith(("_failed", "_error"))
    ):
        return "error"
    if event.event_type.casefold() == "decision_result" and event.metadata.get("should_reply") is False:
        return "skipped"
    if (
        event.event_type.casefold() in {"suppression", "message_duplicate", "approval_canceled"}
        or any(word in tokens for word in ("blocked", " reject", "rejected", "canceled", "cancelled", "skipped", "ignore", "duplicate"))
    ):
        return "skipped"
    if any(word in tokens for word in ("pending", "waiting", "queued")):
        return "pending"
    return "ok"


def _metadata_has_error(metadata: Mapping[str, object]) -> bool:
    value = metadata.get("error")
    if value not in (None, "", False, 0, [], {}):
        return True
    status = str(metadata.get("status") or metadata.get("outcome") or "").strip().casefold()
    return status in {"error", "failed", "failure", "timeout", "timed_out"}


def _phase_status(events: list[_NormalizedTraceEvent]) -> str:
    conditions = [_trace_event_condition(event) for event in events]
    if "timeout" in conditions:
        return "timeout"
    if "error" in conditions:
        return "error"
    return conditions[-1] if conditions[-1] in {"pending", "skipped"} else "ok"


def _trace_status(
    events: list[_NormalizedTraceEvent],
    phases: Mapping[str, Mapping[str, object]],
) -> str:
    phase_statuses = [str(phases[stage].get("status") or "") for stage in TRACE_STAGE_ORDER]
    if "timeout" in phase_statuses:
        return "timeout"
    if "error" in phase_statuses:
        return "error"
    send_status = str(phases["send"].get("status") or "")
    if send_status == "ok":
        return "complete"
    approval_status = str(phases["approval"].get("status") or "")
    if approval_status == "pending":
        return "pending_approval"
    last_condition = _trace_event_condition(events[-1]) if events else "ok"
    if last_condition == "skipped" or "skipped" in phase_statuses:
        return "skipped"
    return "in_progress"


def _trace_error_message(event: _NormalizedTraceEvent) -> str:
    for key in ("error", "error_message", "reason", "detail", "status"):
        value = event.metadata.get(key)
        if value not in (None, "", False):
            return _compact_label(value, limit=240)
    return event.action or event.event_type


def _event_elapsed_ms(metadata: Mapping[str, object], *, stage: str) -> int | None:
    keys = ["elapsed_ms", "latency_ms", "duration_ms"]
    if stage == "approval":
        keys.append("approval_wait_ms")
    values = [_normalized_elapsed_ms(metadata.get(key)) for key in keys]
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _event_field(event: object, key: str, default: object = None) -> object:
    if isinstance(event, Mapping):
        return event.get(key, default)
    try:
        keys = event.keys()  # type: ignore[attr-defined]
        if key in keys:
            return event[key]  # type: ignore[index]
    except (AttributeError, KeyError, TypeError):
        pass
    return getattr(event, key, default)


def _metadata_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (str, bytes, bytearray)):
        try:
            decoded = value.decode("utf-8", errors="replace") if not isinstance(value, str) else value
            parsed = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeError, TypeError):
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _sanitize_trace_value(value: object, *, depth: int, key: str = "") -> object:
    if _sensitive_metadata_key(key):
        return "[REDACTED]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        return f"[binary {len(value)} bytes]"
    if isinstance(value, str):
        text = _sanitize_url(value) if "url" in key.casefold() else value
        return _sanitize_trace_string(text)
    if depth >= MAX_TRACE_METADATA_DEPTH:
        return "[nested data omitted]"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        items = list(value.items())
        for raw_key, item in items[:MAX_TRACE_METADATA_ITEMS]:
            clean_key = _compact_label(raw_key, limit=80) or "unknown"
            result[clean_key] = _sanitize_trace_value(item, depth=depth + 1, key=clean_key)
        if len(items) > MAX_TRACE_METADATA_ITEMS:
            result["__truncated_items__"] = len(items) - MAX_TRACE_METADATA_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [
            _sanitize_trace_value(item, depth=depth + 1, key=key)
            for item in items[:MAX_TRACE_METADATA_LIST_ITEMS]
        ]
        if len(items) > MAX_TRACE_METADATA_LIST_ITEMS:
            result.append(f"[{len(items) - MAX_TRACE_METADATA_LIST_ITEMS} more items omitted]")
        return result
    return _sanitize_trace_string(str(value))


def _sensitive_metadata_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(key).casefold()).strip("_")
    if normalized in _TOKEN_COUNT_KEYS:
        return False
    if normalized in {"token", "session", "session_id", "auth"}:
        return True
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _sanitize_trace_string(value: str) -> str:
    text = " ".join(str(value).split())
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _API_KEY_RE.sub("[REDACTED_API_KEY]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    text = _LONG_IDENTIFIER_RE.sub(lambda match: f"***{match.group(0)[-4:]}", text)
    if len(text) > MAX_TRACE_METADATA_STRING:
        text = text[: MAX_TRACE_METADATA_STRING - 1] + "…"
    return text


def _sanitize_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    query: list[tuple[str, str]] = []
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "[REDACTED]" if _sensitive_metadata_key(key) else item))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _trace_identifier(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]+", "", str(value)).strip()
    if not text or text == "0":
        return ""
    return text[:180]


def _json_scalar(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _compact_label(value, limit=120)


def _compact_label(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _coerce_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))
