from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    model: str
    decision_model: str
    reply_model: str
    utility_model: str
    thinking: str
    reasoning_effort: str
    temperature: float
    max_tokens: int
    timeout_seconds: int


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
        self.persona_dir = PROJECT_ROOT / "personas"
        self.groups = raw.get("groups", {})
        access = raw.get("access_control", {})
        self.allowed_groups = _int_set(access.get("allowed_groups", []))
        self.allowed_private_users = _int_set(access.get("allowed_private_users", []))
        self.user_reply_cooldowns = _int_int_dict(rate.get("user_reply_cooldowns", {}))

        thinking = str(deepseek.get("thinking", "disabled")).lower()
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("deepseek.thinking must be 'enabled' or 'disabled'")

        base_model = str(deepseek.get("model", "deepseek-v4-flash"))
        self.deepseek = DeepSeekConfig(
            base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
            model=base_model,
            decision_model=str(deepseek.get("decision_model", "deepseek-v4-flash")),
            reply_model=str(deepseek.get("reply_model", base_model)),
            utility_model=str(deepseek.get("utility_model", "deepseek-v4-flash")),
            thinking=thinking,
            reasoning_effort=str(deepseek.get("reasoning_effort", "high")),
            temperature=float(deepseek.get("temperature", 0.72)),
            max_tokens=int(deepseek.get("max_tokens", 220)),
            timeout_seconds=int(deepseek.get("timeout_seconds", 30)),
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
