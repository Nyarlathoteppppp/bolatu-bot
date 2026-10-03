from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Callable, Optional
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from nonebot import logger
from openai import AsyncOpenAI

from .config import LLMConfig, LLMModelRoute, LLMProviderConfig, parse_llm_model_route
from .observability import current_correlation_id

LLMUsageRecorder = Callable[[str, str, Optional[int], Optional[int], Optional[int]], None]
_usage_recorder: LLMUsageRecorder | None = None
_PROVIDER_FAILURE_WINDOW_SECONDS = 120.0
_PROVIDER_FAILURE_THRESHOLD = 3
_PROVIDER_CIRCUIT_SECONDS = 300.0
# Off the reply path; no user is waiting on these.
_BACKGROUND_TASKS = frozenset({"mid_memory", "daily_review", "member_profile", "style_learning"})
_BACKGROUND_ATTEMPT_SECONDS = 90.0
_BACKGROUND_TOTAL_SECONDS = 150.0


class LLMGateway:
    """Single transport and routing entry for chat-model tasks."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.clients: dict[str, AsyncOpenAI] = {}
        for name, provider in config.providers.items():
            api_key = os.getenv(provider.api_key_env)
            if not api_key:
                logger.warning(
                    "qq_social_agent llm provider key missing: "
                    f"provider={name} env={provider.api_key_env}"
                )
                continue
            self.clients[name] = AsyncOpenAI(
                api_key=api_key,
                base_url=provider.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        if not self.clients:
            raise RuntimeError("No LLM API key is configured. Put provider keys in .env.")
        self.route_overrides: dict[str, LLMModelRoute] = {}
        self._provider_failures: dict[str, list[float]] = {}
        self._provider_circuit_until: dict[str, float] = {}
        self._provider_exhausted: set[str] = set()
        self._reply_peak_combo_enabled = False

    async def aclose(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients.values()), return_exceptions=True)

    async def probe_model(self, route: LLMModelRoute, *, timeout_seconds: float = 8.0) -> tuple[bool, str]:
        """Check one configured model directly, without a fallback masking its result."""
        client = self.clients.get(route.provider)
        if client is None:
            return False, "未配置密钥"
        provider = self.config.providers[route.provider]
        request: dict[str, object] = {
            "model": route.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 260 if route.provider == "openrouter" else 64,
        }
        extra_body = _extra_body_for_route(provider, route)
        if extra_body:
            request["extra_body"] = extra_body
        try:
            timed_client = client.with_options(timeout=timeout_seconds, max_retries=0)
            operation = (
                timed_client.responses.create(**_responses_request(request, provider))
                if provider.api == "responses"
                else timed_client.chat.completions.create(**request)
            )
            response = await asyncio.wait_for(
                operation,
                timeout=timeout_seconds + 0.25,
            )
            if provider.api == "responses":
                response = _responses_as_chat(response)
        except Exception as exc:
            if _is_quota_exhaustion(exc):
                self._provider_exhausted.add(route.provider)
                return False, "余额不足"
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                return False, f"HTTP {status_code}"
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                return False, "超时"
            return False, type(exc).__name__
        if not getattr(response, "choices", None):
            return False, "空响应"
        if not str(getattr(response.choices[0].message, "content", "") or "").strip():
            return False, "空回复"
        self._provider_exhausted.discard(route.provider)
        return True, "可用"

    async def _chat_completion(
        self,
        *,
        task: str,
        route_name: str,
        request: dict[str, object],
        routes_override: tuple[LLMModelRoute, ...] | None = None,
    ) -> object:
        routes = routes_override if routes_override is not None else self._candidate_routes(route_name)
        last_error: Exception | None = None
        attempt_timeout, total_timeout = self._task_timeouts(task=task, route_name=route_name)
        deadline = time.monotonic() + total_timeout
        for route in routes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = asyncio.TimeoutError(
                    f"LLM total timeout after {total_timeout:g}s for task={task}"
                )
                break
            client = self.clients.get(route.provider)
            if client is None:
                logger.warning(
                    "qq_social_agent llm provider unavailable, trying fallback: "
                    f"task={task} provider={route.provider} model={route.model}"
                )
                continue
            provider = self.config.providers[route.provider]
            provider_request = dict(request)
            provider_request["model"] = route.model
            extra_body = _extra_body_for_route(provider, route)
            if extra_body:
                provider_request["extra_body"] = extra_body
            try:
                current_timeout = max(0.25, min(attempt_timeout, remaining))
                timed_client = client.with_options(
                    timeout=current_timeout,
                    max_retries=0,
                )
                if provider.api == "responses":
                    provider_request["reasoning_effort"] = (
                        provider.reply_reasoning_effort
                        if task in {"reply", "reply_direct", "reply_candidates", "search_answer"}
                        else provider.reasoning_effort
                    )
                    operation = timed_client.responses.create(**_responses_request(provider_request, provider))
                else:
                    operation = timed_client.chat.completions.create(**provider_request)
                response = await asyncio.wait_for(operation, timeout=current_timeout + 0.25)
                if provider.api == "responses":
                    usage = response.usage
                    if usage is not None:
                        logger.info(
                            "qq_social_agent responses usage details: "
                            f"model={route.label} "
                            f"cached_tokens={getattr(getattr(usage, 'input_tokens_details', None), 'cached_tokens', None)} "
                            f"reasoning_tokens={getattr(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens', None)}"
                        )
                    response = _responses_as_chat(response)
            except Exception as exc:
                last_error = exc
                if _is_quota_exhaustion(exc):
                    self._provider_exhausted.add(route.provider)
                    logger.warning(
                        "qq_social_agent llm provider balance exhausted, using fallbacks until restart: "
                        f"provider={route.provider}"
                    )
                self._record_provider_failure(route.provider)
                logger.warning(
                    "qq_social_agent llm provider failed, trying fallback: "
                    f"task={task} provider={route.provider} model={route.model} error={exc}"
                )
                continue
            self._record_provider_success(route.provider)
            if task in {"reply", "reply_direct", "reply_candidates", "search_answer"}:
                choice = response.choices[0]
                logger.info(
                    "qq_social_agent llm reply completion: "
                    f"task={task} model={route.label} "
                    f"finish_reason={getattr(choice, 'finish_reason', None)} "
                    f"content_chars={len(choice.message.content or '')}"
                )
            if self.config.usage_tracking_enabled:
                _log_llm_usage(task, response, model=route.label)
            return response
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No available LLM provider for route={route_name}")

    async def complete_on_model(
        self, *, task: str, route: LLMModelRoute, request: dict[str, object]
    ) -> object:
        """Use a background selection without mutating the live reply route."""
        routes = (route,)
        if route.provider == "lingsuan":
            routes += (self.config.fallback_routes["memory" if task == "mid_memory" else "reply"],)
        return await self._chat_completion(
            task=task,
            route_name="memory" if task == "mid_memory" else "reply",
            request=request,
            routes_override=routes,
        )

    def _task_timeouts(self, *, task: str, route_name: str) -> tuple[float, float]:
        if route_name == "decision" or task == "decision":
            attempt = self.config.decision_timeout_seconds
            total = self.config.decision_total_timeout_seconds
        elif task == "mid_memory":
            # Sixty-message structured summaries are background work and need a
            # wider budget than short utility classifications. This does not sit
            # on the reply path.
            attempt = max(self.config.utility_timeout_seconds, 18.0)
            total = max(self.config.utility_total_timeout_seconds, 40.0)
        elif task == "daily_review":
            attempt = getattr(self.config, "daily_review_timeout_seconds", 35.0)
            total = getattr(self.config, "daily_review_total_timeout_seconds", 75.0)
        elif route_name in {"reply", "search"} or task in {
            "reply",
            "reply_direct",
            "reply_candidates",
            "search_answer",
        }:
            attempt = self.config.reply_timeout_seconds
            total = self.config.reply_total_timeout_seconds
            if route_name == "reply":
                routes = self._candidate_routes(route_name)
                provider = self.config.providers[routes[0].provider] if routes else None
                if provider is not None:
                    attempt = provider.reply_timeout_seconds or attempt
                    total = provider.reply_total_timeout_seconds or total
        elif route_name in {"utility", "jargon", "memory", "style", "member_profile"}:
            attempt = self.config.utility_timeout_seconds
            total = self.config.utility_total_timeout_seconds
        else:
            attempt = float(self.config.timeout_seconds)
            total = float(self.config.timeout_seconds) * max(1, len(self._candidate_routes(route_name)))
        routes = self._candidate_routes(route_name)
        if task in _BACKGROUND_TASKS:
            # Structured summaries write 1-3k tokens; a reply-sized deadline
            # made the primary model time out and the truncating fallback win.
            attempt = max(float(attempt), _BACKGROUND_ATTEMPT_SECONDS)
            total = max(float(total), _BACKGROUND_TOTAL_SECONDS)
        elif routes and routes[0].provider == "lingsuan":
            provider = self.config.providers["lingsuan"]
            attempt = provider.reply_timeout_seconds or attempt
            total = max(total, provider.reply_total_timeout_seconds or total)
        attempt = max(1.0, float(attempt))
        total = max(attempt, float(total))
        return attempt, total

    def _candidate_routes(self, route_name: str) -> tuple[LLMModelRoute, ...]:
        if route_name == "reply" and getattr(self, "_reply_peak_combo_enabled", False):
            routes = [self.config.fallback_routes["reply"]]
            routes.extend(self.config.additional_fallback_routes["reply"])
        else:
            primary = self.route_overrides.get(route_name, self.config.routes[route_name])
            fallback = self.config.fallback_routes.get(route_name)
            routes = [primary]
            if fallback is not None:
                routes.append(fallback)
            routes.extend(getattr(self.config, "additional_fallback_routes", {}).get(route_name, ()))
            provider = self.config.providers[primary.provider]
            if route_name == "reply" and provider.reply_fallback_models:
                routes = [primary, *(self.parse_model_route(model) for model in provider.reply_fallback_models)]
            elif primary.provider == "lingsuan":
                routes = [primary, self.config.fallback_routes[route_name]]
        if route_name == "reply" and route_name not in self.route_overrides and routes[0].provider != "lingsuan":
            # Keep the existing DeepSeek/SiliconFlow peak policy when they are
            # first and second choice, or the two fallbacks after MiMo.
            for index in range(len(routes) - 1):
                if (
                    routes[index].provider == "deepseek"
                    and routes[index + 1].provider == "siliconflow"
                ):
                    if self._is_reply_peak_now():
                        routes[index], routes[index + 1] = routes[index + 1], routes[index]
                    break
        available: list[LLMModelRoute] = []
        seen: set[tuple[str, str]] = set()
        for route in routes:
            route_key = (route.provider, route.model)
            if route_key in seen:
                continue
            seen.add(route_key)
            if route.provider in getattr(self, "_provider_exhausted", set()):
                continue
            if self._provider_circuit_until.get(route.provider, 0.0) > time.monotonic():
                continue
            available.append(route)
        return tuple(available)

    def _is_reply_peak_now(self) -> bool:
        routing = getattr(self.config, "reply_peak_routing", None)
        if routing is None or not routing.enabled or not routing.windows:
            return False
        try:
            now = datetime.now(ZoneInfo(routing.timezone))
        except Exception as exc:
            logger.warning(
                "qq_social_agent invalid reply peak routing timezone, disabling schedule: "
                f"timezone={getattr(routing, 'timezone', '')} error={exc}"
            )
            return False
        return self._is_reply_peak_at(now)

    def _is_reply_peak_at(self, now: datetime) -> bool:
        routing = getattr(self.config, "reply_peak_routing", None)
        if routing is None or not routing.enabled or not routing.windows:
            return False
        if now.weekday() not in routing.weekdays:
            return False
        current_minute = now.hour * 60 + now.minute
        return any(start <= current_minute < end for start, end in routing.windows)

    def _record_provider_failure(self, provider: str) -> None:
        now = time.monotonic()
        failures = [
            observed_at
            for observed_at in self._provider_failures.get(provider, [])
            if now - observed_at <= _PROVIDER_FAILURE_WINDOW_SECONDS
        ]
        failures.append(now)
        self._provider_failures[provider] = failures
        if len(failures) < _PROVIDER_FAILURE_THRESHOLD:
            return
        previous_until = self._provider_circuit_until.get(provider, 0.0)
        until = now + _PROVIDER_CIRCUIT_SECONDS
        self._provider_circuit_until[provider] = until
        if previous_until <= now:
            logger.warning(
                "qq_social_agent llm provider circuit opened: "
                f"provider={provider} failures={len(failures)} cooldown={int(_PROVIDER_CIRCUIT_SECONDS)}s"
            )

    def _record_provider_success(self, provider: str) -> None:
        now = time.monotonic()
        failures = [
            observed_at
            for observed_at in self._provider_failures.get(provider, [])
            if now - observed_at <= _PROVIDER_FAILURE_WINDOW_SECONDS
        ]
        if failures:
            failures.pop(0)
        if failures:
            self._provider_failures[provider] = failures
        else:
            self._provider_failures.pop(provider, None)
        # An open circuit stays open until cooldown even if a later call
        # succeeds. Mixed SiliconFlow timeouts otherwise never trip.

    def parse_model_route(self, value: str, *, default_provider: str = "siliconflow") -> LLMModelRoute:
        if default_provider not in self.config.providers:
            default_provider = "deepseek"
        return parse_llm_model_route(value, self.config.providers, default_provider=default_provider)

    def set_route_override(self, route_name: str, route: LLMModelRoute | None) -> None:
        if route_name not in self.config.routes:
            raise ValueError(f"unknown route: {route_name}")
        if route_name == "reply":
            self._reply_peak_combo_enabled = False
        if route is None:
            self.route_overrides.pop(route_name, None)
            return
        self.route_overrides[route_name] = route

    def set_reply_peak_combo(self) -> None:
        self.route_overrides.pop("reply", None)
        self._reply_peak_combo_enabled = True

    def current_route(self, route_name: str) -> LLMModelRoute:
        if route_name == "reply" and getattr(self, "_reply_peak_combo_enabled", False):
            routes = self._candidate_routes("reply")
            return routes[0] if routes else self.config.fallback_routes["reply"]
        return self.route_overrides.get(route_name, self.config.routes[route_name])


def _responses_request(request: dict, provider: LLMProviderConfig) -> dict:
    result = {
        "model": request["model"],
        "input": request["messages"],
        "store": False,
        "max_output_tokens": request.get("max_tokens", 320),
        "reasoning": {"effort": request.get("reasoning_effort", provider.reasoning_effort) if provider.thinking == "enabled" else "none"},
    }
    if "response_format" in request:
        result["text"] = {"format": request["response_format"]}
    if "prompt_cache_key" in request:
        result["prompt_cache_key"] = request["prompt_cache_key"]
    else:
        correlation_id = current_correlation_id()
        if correlation_id.startswith("group:"):
            result["prompt_cache_key"] = correlation_id.rsplit(":", 1)[0]
    return result


def _responses_as_chat(response: object) -> object:
    usage = response.usage
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=response.output_text),
            finish_reason=response.status,
        )],
        usage=SimpleNamespace(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ) if usage is not None else None,
    )


def set_usage_recorder(recorder: LLMUsageRecorder | None) -> None:
    global _usage_recorder
    _usage_recorder = recorder


def _log_llm_usage(task: str, response: object, *, model: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return
    logger.info(
        "qq_social_agent llm usage: "
        f"task={task} model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
    )
    if _usage_recorder is not None:
        try:
            _usage_recorder(task, model, prompt_tokens, completion_tokens, total_tokens)
        except Exception as exc:
            logger.warning(f"qq_social_agent failed recording llm usage: task={task} error={exc}")


def _extra_body_for_route(provider: LLMProviderConfig, route: LLMModelRoute) -> dict[str, object]:
    if provider.thinking not in {"enabled", "disabled"}:
        return {}
    model = route.model.casefold()
    if provider.name == "siliconflow" and provider.thinking == "disabled" and model.startswith("qwen/"):
        return {"enable_thinking": False}
    if provider.name != "deepseek":
        return {}
    return {"thinking": {"type": provider.thinking}}


def _usage_value(usage: object, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_quota_exhaustion(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 402:
        return True
    message = str(error).casefold()
    return "insufficient balance" in message or "insufficient account balance" in message
