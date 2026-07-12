from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .observability import mark_onebot_api_error, mark_onebot_api_success


DEFAULT_API_TIMEOUT_SECONDS = 10.0
_ERROR_TEXT_LIMIT = 240


class OneBotGateway(Protocol):
    async def call_api(self, api: str, **data: Any) -> Any:
        ...


@dataclass
class _ApiCallStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    cancellations: int = 0
    in_flight: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_call_at: float | None = None
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error: str = ""


_gateway_stats = _ApiCallStats()
_api_stats: dict[str, _ApiCallStats] = {}
_last_api = ""


async def call_api(
    bot: OneBotGateway,
    api: str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
    **data: Any,
) -> Any:
    api_name = str(api or "").strip()
    if not api_name:
        raise ValueError("OneBot API name is required")
    started_at = time.time()
    started = time.perf_counter()
    _record_call_started(api_name, started_at=started_at)
    outcome = "success"
    error_text = ""
    caught_error: object = ""
    try:
        operation = bot.call_api(api_name, **data)
        timeout = float(timeout_seconds) if timeout_seconds is not None else None
        if timeout is None or timeout <= 0:
            return await operation
        return await asyncio.wait_for(operation, timeout=timeout)
    except asyncio.TimeoutError as exc:
        outcome = "timeout"
        error_text = f"timeout after {float(timeout_seconds or 0):g}s"
        caught_error = exc
        raise
    except asyncio.CancelledError as exc:
        outcome = "cancelled"
        error_text = "cancelled"
        caught_error = exc
        raise
    except Exception as exc:
        outcome = "failure"
        error_text = _compact_error(exc)
        caught_error = exc
        raise
    finally:
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
        _record_call_finished(
            api_name,
            outcome=outcome,
            latency_ms=latency_ms,
            finished_at=time.time(),
            error_text=error_text,
        )
        bot_id = str(getattr(bot, "self_id", "") or "").strip()
        if bot_id:
            if outcome == "success":
                mark_onebot_api_success(bot_id, api_name, elapsed_ms=latency_ms)
            elif outcome != "cancelled":
                mark_onebot_api_error(
                    bot_id,
                    api_name,
                    elapsed_ms=latency_ms,
                    error=caught_error or error_text,
                    timeout=outcome == "timeout",
                )


def unwrap_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


async def get_group_info(
    bot: OneBotGateway,
    group_id: int,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = await call_api(bot, "get_group_info", timeout_seconds=timeout_seconds, group_id=group_id)
    data = unwrap_data(payload)
    return data if isinstance(data, dict) else {}


async def get_group_member_list(
    bot: OneBotGateway,
    group_id: int,
    *,
    no_cache: bool = True,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    payload = await call_api(
        bot,
        "get_group_member_list",
        timeout_seconds=timeout_seconds,
        group_id=group_id,
        no_cache=no_cache,
    )
    return _list_payload(payload, "members", "member_list")


async def get_group_msg_history(
    bot: OneBotGateway,
    group_id: int,
    *,
    count: int = 50,
    message_seq: int | str | None = None,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    request: dict[str, Any] = {"group_id": group_id, "count": count}
    if message_seq is not None:
        request["message_seq"] = str(message_seq)
    payload = await call_api(bot, "get_group_msg_history", timeout_seconds=timeout_seconds, **request)
    return _list_payload(payload, "messages", "message")


async def get_msg(
    bot: OneBotGateway,
    message_id: int | str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = await call_api(
        bot,
        "get_msg",
        timeout_seconds=timeout_seconds,
        message_id=int(message_id),
    )
    data = unwrap_data(payload)
    return data if isinstance(data, dict) else {}


async def get_forward_msg(
    bot: OneBotGateway,
    forward_id: str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    return await call_api(
        bot,
        "get_forward_msg",
        timeout_seconds=timeout_seconds,
        id=forward_id,
    )


async def get_image(
    bot: OneBotGateway,
    file: str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = await call_api(bot, "get_image", timeout_seconds=timeout_seconds, file=file)
    data = unwrap_data(payload)
    return data if isinstance(data, dict) else {}


async def get_file(
    bot: OneBotGateway,
    file_id: str | None = None,
    *,
    file: str | None = None,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request: dict[str, Any] = {}
    if file_id:
        request["file_id"] = str(file_id)
    if file:
        request["file"] = str(file)
    if not request:
        return {}
    payload = await call_api(bot, "get_file", timeout_seconds=timeout_seconds, **request)
    data = unwrap_data(payload)
    return data if isinstance(data, dict) else {}


async def get_record(
    bot: OneBotGateway,
    *,
    file_id: str | None = None,
    file: str | None = None,
    out_format: str = "mp3",
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request: dict[str, Any] = {"out_format": str(out_format or "mp3")}
    if file_id:
        request["file_id"] = str(file_id)
    if file:
        request["file"] = str(file)
    if len(request) <= 1:
        return {}
    payload = await call_api(bot, "get_record", timeout_seconds=timeout_seconds, **request)
    data = unwrap_data(payload)
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return {"file": data}
    return {}


async def ocr_image(
    bot: OneBotGateway,
    image: str,
    *,
    enhanced: bool = False,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    api = ".ocr_image" if enhanced else "ocr_image"
    return await call_api(bot, api, timeout_seconds=timeout_seconds, image=image)


async def set_msg_emoji_like(
    bot: OneBotGateway,
    message_id: int | str,
    emoji_id: str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    return await call_api(
        bot,
        "set_msg_emoji_like",
        timeout_seconds=timeout_seconds,
        message_id=int(message_id),
        emoji_id=str(emoji_id),
    )


async def send_poke(
    bot: OneBotGateway,
    user_id: int | str,
    *,
    group_id: int | str | None = None,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    request: dict[str, Any] = {"user_id": int(user_id)}
    if group_id is not None:
        request["group_id"] = int(group_id)
    return await call_api(bot, "send_poke", timeout_seconds=timeout_seconds, **request)


async def mark_group_msg_as_read(
    bot: OneBotGateway,
    group_id: int | str,
    *,
    timeout_seconds: float | None = DEFAULT_API_TIMEOUT_SECONDS,
) -> Any:
    return await call_api(
        bot,
        "mark_group_msg_as_read",
        timeout_seconds=timeout_seconds,
        group_id=int(group_id),
    )


def status_snapshot() -> dict[str, Any]:
    return {
        **_stats_payload(_gateway_stats),
        "default_timeout_seconds": DEFAULT_API_TIMEOUT_SECONDS,
        "last_api": _last_api,
        "apis": {name: _stats_payload(stats) for name, stats in sorted(_api_stats.items())},
    }


def _record_call_started(api: str, *, started_at: float) -> None:
    global _last_api
    _last_api = api
    for stats in (_gateway_stats, _api_stats.setdefault(api, _ApiCallStats())):
        stats.calls += 1
        stats.in_flight += 1
        stats.last_call_at = started_at


def _record_call_finished(
    api: str,
    *,
    outcome: str,
    latency_ms: float,
    finished_at: float,
    error_text: str,
) -> None:
    for stats in (_gateway_stats, _api_stats.setdefault(api, _ApiCallStats())):
        stats.in_flight = max(0, stats.in_flight - 1)
        stats.total_latency_ms += latency_ms
        stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
        if outcome == "success":
            stats.successes += 1
            stats.last_success_at = finished_at
            continue
        stats.failures += 1
        stats.last_failure_at = finished_at
        stats.last_error = error_text[:_ERROR_TEXT_LIMIT]
        if outcome == "timeout":
            stats.timeouts += 1
        elif outcome == "cancelled":
            stats.cancellations += 1


def _stats_payload(stats: _ApiCallStats) -> dict[str, Any]:
    completed = stats.successes + stats.failures
    average_latency_ms = stats.total_latency_ms / completed if completed else 0.0
    return {
        "calls": stats.calls,
        "successes": stats.successes,
        "failures": stats.failures,
        "timeouts": stats.timeouts,
        "cancellations": stats.cancellations,
        "in_flight": stats.in_flight,
        "average_latency_ms": round(average_latency_ms, 2),
        "max_latency_ms": round(stats.max_latency_ms, 2),
        "last_call_at": stats.last_call_at,
        "last_success_at": stats.last_success_at,
        "last_failure_at": stats.last_failure_at,
        "last_error": stats.last_error,
    }


def _compact_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()).strip()
    label = type(exc).__name__
    rendered = f"{label}: {text}" if text else label
    return rendered[:_ERROR_TEXT_LIMIT]


def _list_payload(payload: Any, *keys: str) -> list[dict[str, Any]]:
    data = unwrap_data(payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []
