from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .market_intent import MarketIntent


@dataclass(frozen=True)
class MarketSnapshot:
    kind: str
    symbol: str
    display_name: str
    price: float
    currency: str
    change_percent: float | None
    volume: float | None
    market_cap: float | None
    source: str
    updated_at: str | None = None

    def to_prompt_line(self) -> str:
        parts = [
            f"{self.display_name}/{self.symbol}",
            f"价格 {_format_number(self.price)} {self.currency}",
        ]
        if self.change_percent is not None:
            parts.append(f"涨跌 {_format_percent(self.change_percent)}")
        if self.volume is not None:
            parts.append(f"成交量 {_format_number(self.volume)}")
        if self.market_cap is not None:
            parts.append(f"市值 {_format_number(self.market_cap)}")
        if self.updated_at:
            parts.append(f"更新时间 {self.updated_at}")
        parts.append(f"来源 {self.source}")
        return "- " + "，".join(parts)


@dataclass(frozen=True)
class MarketLookup:
    intent: MarketIntent
    snapshot: MarketSnapshot | None
    status: str


class MarketTool:
    def __init__(self, *, max_external_queries_per_minute: int = 2, cache_ttl_seconds: int = 60):
        self.max_external_queries_per_minute = max_external_queries_per_minute
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, MarketSnapshot]] = {}
        self._query_times: list[float] = []

    async def context_for(self, intents: list[MarketIntent]) -> str:
        if not intents:
            return ""

        lookups = await self.lookup(intents)
        return _prompt_context_from_lookups(lookups)

    async def report_and_context_for(self, intents: list[MarketIntent]) -> tuple[str, str]:
        lookups = await self.lookup(intents)
        return _chat_report_from_lookups(lookups), _prompt_context_from_lookups(lookups)

    async def lookup(self, intents: list[MarketIntent]) -> list[MarketLookup]:
        lookups: list[MarketLookup] = []
        for intent in intents[:2]:
            lookups.append(await self._lookup_snapshot(intent))
        return lookups

    async def _lookup_snapshot(self, intent: MarketIntent) -> MarketLookup:
        key = (intent.kind, intent.symbol)
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] <= self.cache_ttl_seconds:
            return MarketLookup(intent, cached[1], "ok_cached")

        if not self._allow_external_query(now):
            return MarketLookup(intent, None, "rate_limited")

        if intent.kind == "stock":
            snapshot = await asyncio.to_thread(_fetch_stock_snapshot, intent)
        else:
            snapshot = await _fetch_crypto_snapshot(intent)
        if snapshot:
            self._cache[key] = (now, snapshot)
            return MarketLookup(intent, snapshot, "ok")
        return MarketLookup(intent, None, "failed")

    def _allow_external_query(self, now: float) -> bool:
        self._query_times = [t for t in self._query_times if now - t < 60]
        if len(self._query_times) >= self.max_external_queries_per_minute:
            return False
        self._query_times.append(now)
        return True


def _prompt_context_from_lookups(lookups: list[MarketLookup]) -> str:
    if not lookups:
        return ""

    lines: list[str] = []
    for lookup in lookups:
        if lookup.snapshot is not None:
            lines.append(lookup.snapshot.to_prompt_line())
            continue
        lines.append(_failure_prompt_line(lookup))

    if not lines:
        return ""

    return "\n".join(
        [
            "市场工具结果（成功就引用数据；失败就明确告诉群友查询失败，不要编造价格）：",
            *lines,
            "回复时可以引用这些数据，但不要直接给买卖/满仓/梭哈结论。",
        ]
    )


def _chat_report_from_lookups(lookups: list[MarketLookup]) -> str:
    lines: list[str] = []
    for lookup in lookups:
        if lookup.snapshot is not None:
            lines.append(_snapshot_chat_line(lookup.snapshot))
            continue
        lines.append(_failure_chat_line(lookup))
    return "\n".join(lines)


def _snapshot_chat_line(snapshot: MarketSnapshot) -> str:
    label = snapshot.display_name
    if snapshot.symbol and snapshot.symbol != snapshot.display_name:
        label = f"{snapshot.display_name}/{snapshot.symbol}"
    parts = [
        label,
        f"{_format_number(snapshot.price)} {snapshot.currency}",
    ]
    if snapshot.change_percent is not None:
        parts.append(f"涨跌 {_format_percent(snapshot.change_percent)}")
    if snapshot.volume is not None:
        parts.append(f"量 {_format_number(snapshot.volume)}")
    if snapshot.updated_at:
        parts.append(f"更新 {snapshot.updated_at}")
    parts.append(f"源 {snapshot.source}")
    return "，".join(parts) + "。"


def _failure_chat_line(lookup: MarketLookup) -> str:
    label = f"{lookup.intent.display_name}/{lookup.intent.symbol}"
    if lookup.status == "rate_limited":
        return f"{label} 查询失败：本分钟行情查询到上限了，等会儿再查。"
    return f"{label} 查询失败：行情工具没拿到有效数据，换 ticker/币种再试。"


def _fetch_stock_snapshot(intent: MarketIntent) -> MarketSnapshot | None:
    import yfinance as yf

    try:
        ticker = yf.Ticker(intent.symbol)
        fast = ticker.fast_info
        price = _as_float(_fast_get(fast, "last_price", "lastPrice"))
        previous_close = _as_float(_fast_get(fast, "previous_close", "previousClose"))
        if price is None:
            return None
        change_percent = None
        if previous_close:
            change_percent = (price - previous_close) / previous_close * 100
        return MarketSnapshot(
            kind="stock",
            symbol=intent.symbol,
            display_name=intent.display_name,
            price=price,
            currency=str(_fast_get(fast, "currency") or "USD"),
            change_percent=change_percent,
            volume=_as_float(_fast_get(fast, "last_volume", "lastVolume", "ten_day_average_volume")),
            market_cap=_as_float(_fast_get(fast, "market_cap", "marketCap")),
            source="Yahoo Finance",
        )
    except Exception:
        return None


async def _fetch_crypto_snapshot(intent: MarketIntent) -> MarketSnapshot | None:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": intent.symbol,
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
        raw = response.json().get(intent.symbol)
        if not raw:
            return None
        updated = raw.get("last_updated_at")
        updated_at = None
        if isinstance(updated, (int, float)):
            updated_at = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M")
        return MarketSnapshot(
            kind="crypto",
            symbol=intent.display_name,
            display_name=intent.display_name,
            price=float(raw["usd"]),
            currency="USD",
            change_percent=_as_float(raw.get("usd_24h_change")),
            volume=_as_float(raw.get("usd_24h_vol")),
            market_cap=_as_float(raw.get("usd_market_cap")),
            source="CoinGecko",
            updated_at=updated_at,
        )
    except Exception:
        return None


def _fast_get(source: Any, *keys: str) -> Any:
    for key in keys:
        try:
            value = source[key]
            if value is not None:
                return value
        except Exception:
            pass
        try:
            value = getattr(source, key)
            if value is not None:
                return value
        except Exception:
            pass
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _failure_prompt_line(lookup: MarketLookup) -> str:
    label = f"{lookup.intent.display_name}/{lookup.intent.symbol}"
    if lookup.status == "rate_limited":
        return (
            f"- {label} 查询失败：本分钟外部行情查询已达上限（2 次）。"
            "不要编造价格，告诉群友稍后再查。"
        )
    return (
        f"- {label} 查询失败：行情工具没有拿到有效数据。"
        "不要编造价格，告诉群友查询失败或换 ticker/币种。"
    )
