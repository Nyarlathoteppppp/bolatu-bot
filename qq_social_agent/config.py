from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    thinking: str


@dataclass(frozen=True)
class LLMModelRoute:
    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    model: str
    decision_model: str
    reply_model: str
    search_model: str
    utility_model: str
    jargon_model: str
    memory_model: str
    style_model: str
    member_profile_model: str
    thinking: str
    reasoning_effort: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    decision_timeout_seconds: float
    decision_total_timeout_seconds: float
    reply_timeout_seconds: float
    reply_total_timeout_seconds: float
    daily_review_timeout_seconds: float
    daily_review_total_timeout_seconds: float
    utility_timeout_seconds: float
    utility_total_timeout_seconds: float
    max_retries: int
    providers: dict[str, LLMProviderConfig]
    routes: dict[str, LLMModelRoute]
    fallback_routes: dict[str, LLMModelRoute]
    model_catalog: tuple[LLMModelRoute, ...]
    usage_tracking_enabled: bool


@dataclass(frozen=True)
class RateConfig:
    min_interval_seconds: int
    hard_mention_interval_seconds: int
    max_replies_per_10min: int
    max_replies_per_hour: int
    max_consecutive_replies: int
    quiet_hours_enabled: bool
    quiet_hours_start: str
    quiet_hours_end: str


class AppConfig:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        bot = raw.get("bot", {})
        deepseek = raw.get("deepseek", {})
        rate = raw.get("rate_control", {})
        quiet = rate.get("quiet_hours", {})

        self.default_persona = str(bot.get("default_persona", "zhangxuefeng"))
        self.context_limit = int(bot.get("context_limit", 60))
        self.active_reply_on_mention = bool(bot.get("active_reply_on_mention", True))
        self.data_path = PROJECT_ROOT / str(bot.get("data_path", "data/bot.sqlite3"))
        self.persona_dir = PROJECT_ROOT / str(bot.get("persona_dir", "prompts"))
        self.groups = raw.get("groups", {})
        access = raw.get("access_control", {})
        self.allowed_groups = _int_set(access.get("allowed_groups", []))
        self.allowed_private_users = _int_set(access.get("allowed_private_users", []))
        self.user_reply_cooldowns = _int_int_dict(rate.get("user_reply_cooldowns", {}))

        thinking = str(deepseek.get("thinking", "disabled")).lower()
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("deepseek.thinking must be 'enabled' or 'disabled'")

        providers = _llm_providers(deepseek)
        base_model = str(deepseek.get("model", "deepseek-v4-flash"))
        decision_model = str(deepseek.get("decision_model", "deepseek-v4-flash"))
        reply_model = str(deepseek.get("reply_model", base_model))
        search_model = str(deepseek.get("search_model", decision_model))
        utility_model = str(deepseek.get("utility_model", "deepseek-v4-flash"))
        jargon_model = str(deepseek.get("jargon_model", utility_model))
        memory_model = str(deepseek.get("memory_model", utility_model))
        style_model = str(deepseek.get("style_model", utility_model))
        member_profile_model = str(deepseek.get("member_profile_model", style_model))
        fallback_models = deepseek.get("fallback_models", {})
        if not isinstance(fallback_models, dict):
            fallback_models = {}
        routes = {
            "base": parse_llm_model_route(base_model, providers, default_provider="deepseek"),
            "decision": parse_llm_model_route(decision_model, providers, default_provider="deepseek"),
            "reply": parse_llm_model_route(reply_model, providers, default_provider="deepseek"),
            "search": parse_llm_model_route(search_model, providers, default_provider="deepseek"),
            "utility": parse_llm_model_route(utility_model, providers, default_provider="deepseek"),
            "jargon": parse_llm_model_route(jargon_model, providers, default_provider="deepseek"),
            "memory": parse_llm_model_route(memory_model, providers, default_provider="deepseek"),
            "style": parse_llm_model_route(style_model, providers, default_provider="deepseek"),
            "member_profile": parse_llm_model_route(member_profile_model, providers, default_provider="deepseek"),
        }
        fallback_routes = {
            "decision": _parse_model_route(
                str(fallback_models.get("decision", "deepseek-v4-flash")),
                providers,
                default_provider="deepseek",
            ),
            "reply": _parse_model_route(
                str(fallback_models.get("reply", reply_model)),
                providers,
                default_provider="deepseek",
            ),
            "search": _parse_model_route(
                str(fallback_models.get("search", fallback_models.get("reply", reply_model))),
                providers,
                default_provider="deepseek",
            ),
            "utility": _parse_model_route(
                str(fallback_models.get("utility", "deepseek-v4-flash")),
                providers,
                default_provider="deepseek",
            ),
            "jargon": _parse_model_route(
                str(fallback_models.get("jargon", fallback_models.get("utility", "deepseek-v4-flash"))),
                providers,
                default_provider="deepseek",
            ),
            "memory": _parse_model_route(
                str(fallback_models.get("memory", fallback_models.get("utility", "deepseek-v4-flash"))),
                providers,
                default_provider="deepseek",
            ),
            "style": _parse_model_route(
                str(fallback_models.get("style", fallback_models.get("utility", "deepseek-v4-flash"))),
                providers,
                default_provider="deepseek",
            ),
            "member_profile": _parse_model_route(
                str(fallback_models.get("member_profile", fallback_models.get("utility", "deepseek-v4-flash"))),
                providers,
                default_provider="deepseek",
            ),
        }
        self.deepseek = DeepSeekConfig(
            base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
            model=base_model,
            decision_model=decision_model,
            reply_model=reply_model,
            search_model=search_model,
            utility_model=utility_model,
            jargon_model=jargon_model,
            memory_model=memory_model,
            style_model=style_model,
            member_profile_model=member_profile_model,
            thinking=thinking,
            reasoning_effort=str(deepseek.get("reasoning_effort", "high")),
            temperature=float(deepseek.get("temperature", 0.72)),
            max_tokens=int(deepseek.get("max_tokens", 220)),
            timeout_seconds=int(deepseek.get("timeout_seconds", 30)),
            decision_timeout_seconds=float(deepseek.get("decision_timeout_seconds", 10)),
            decision_total_timeout_seconds=float(deepseek.get("decision_total_timeout_seconds", 18)),
            reply_timeout_seconds=float(deepseek.get("reply_timeout_seconds", 18)),
            reply_total_timeout_seconds=float(deepseek.get("reply_total_timeout_seconds", 28)),
            daily_review_timeout_seconds=float(deepseek.get("daily_review_timeout_seconds", 35)),
            daily_review_total_timeout_seconds=float(deepseek.get("daily_review_total_timeout_seconds", 75)),
            utility_timeout_seconds=float(deepseek.get("utility_timeout_seconds", 8)),
            utility_total_timeout_seconds=float(deepseek.get("utility_total_timeout_seconds", 12)),
            max_retries=max(0, min(2, int(deepseek.get("max_retries", 0)))),
            providers=providers,
            routes=routes,
            fallback_routes=fallback_routes,
            model_catalog=_model_catalog(deepseek, routes, fallback_routes, providers),
            usage_tracking_enabled=bool(deepseek.get("usage_tracking_enabled", True)),
        )
        self.rate = RateConfig(
            min_interval_seconds=int(rate.get("min_interval_seconds", 60)),
            hard_mention_interval_seconds=int(rate.get("hard_mention_interval_seconds", 8)),
            max_replies_per_10min=int(rate.get("max_replies_per_10min", 5)),
            max_replies_per_hour=int(rate.get("max_replies_per_hour", 18)),
            max_consecutive_replies=int(rate.get("max_consecutive_replies", 1)),
            quiet_hours_enabled=bool(quiet.get("enabled", True)),
            quiet_hours_start=str(quiet.get("start", "01:00")),
            quiet_hours_end=str(quiet.get("end", "08:00")),
        )

    def group_config(self, group_id: int | str) -> dict[str, Any]:
        default = dict(self.groups.get("default", {}))
        specific = self.groups.get(str(group_id), {})
        default.update(specific)
        return default

    def group_allowed(self, group_id: int | str) -> bool:
        return not self.allowed_groups or int(group_id) in self.allowed_groups

    def private_user_allowed(self, user_id: int | str) -> bool:
        return not self.allowed_private_users or int(user_id) in self.allowed_private_users


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(raw)


def _int_set(values: object) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, (str, int)):
        values = [values]
    return {int(value) for value in values}


def _int_int_dict(values: object) -> dict[int, int]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("expected mapping of int keys to int values")
    return {int(key): int(value) for key, value in values.items()}


def _llm_providers(deepseek: dict[str, Any]) -> dict[str, LLMProviderConfig]:
    raw_providers = deepseek.get("providers", {})
    providers: dict[str, LLMProviderConfig] = {
        "deepseek": LLMProviderConfig(
            name="deepseek",
            base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
            api_key_env=str(deepseek.get("api_key_env", "DEEPSEEK_API_KEY")),
            thinking=str(deepseek.get("thinking", "disabled")).lower(),
        )
    }
    if isinstance(raw_providers, dict):
        for name, raw in raw_providers.items():
            if not isinstance(raw, dict):
                continue
            provider_name = str(name).strip()
            if not provider_name:
                continue
            providers[provider_name] = LLMProviderConfig(
                name=provider_name,
                base_url=str(raw.get("base_url", providers.get(provider_name, providers["deepseek"]).base_url)),
                api_key_env=str(raw.get("api_key_env", f"{provider_name.upper()}_API_KEY")),
                thinking=str(raw.get("thinking", "disabled")).lower(),
            )
    return providers


def parse_llm_model_route(
    value: str,
    providers: dict[str, LLMProviderConfig],
    *,
    default_provider: str,
) -> LLMModelRoute:
    text = value.strip()
    for provider in sorted(providers, key=len, reverse=True):
        prefix = f"{provider}/"
        if text.startswith(prefix):
            return LLMModelRoute(provider=provider, model=text[len(prefix) :])
    return LLMModelRoute(provider=default_provider, model=text)


def _parse_model_route(
    value: str,
    providers: dict[str, LLMProviderConfig],
    *,
    default_provider: str,
) -> LLMModelRoute:
    return parse_llm_model_route(value, providers, default_provider=default_provider)


def _model_catalog(
    deepseek: dict[str, Any],
    routes: dict[str, LLMModelRoute],
    fallback_routes: dict[str, LLMModelRoute],
    providers: dict[str, LLMProviderConfig],
) -> tuple[LLMModelRoute, ...]:
    raw_catalog = deepseek.get("model_catalog")
    values: list[str] = []
    if isinstance(raw_catalog, list):
        values.extend(str(item).strip() for item in raw_catalog if str(item).strip())
    else:
        values.extend(route.label for route in routes.values())
        values.extend(route.label for route in fallback_routes.values())

    seen: set[str] = set()
    catalog: list[LLMModelRoute] = []
    for value in values:
        route = parse_llm_model_route(value, providers, default_provider="deepseek")
        if route.label in seen:
            continue
        seen.add(route.label)
        catalog.append(route)
    return tuple(catalog)
