from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketIntent:
    kind: str
    symbol: str
    display_name: str


STOCK_ALIASES = {
    "英伟达": ("NVDA", "英伟达"),
    "nvidia": ("NVDA", "NVIDIA"),
    "苹果": ("AAPL", "苹果"),
    "apple": ("AAPL", "Apple"),
    "特斯拉": ("TSLA", "特斯拉"),
    "tesla": ("TSLA", "Tesla"),
    "微软": ("MSFT", "微软"),
    "谷歌": ("GOOGL", "谷歌"),
    "google": ("GOOGL", "Google"),
    "亚马逊": ("AMZN", "亚马逊"),
    "amazon": ("AMZN", "Amazon"),
    "meta": ("META", "Meta"),
    "脸书": ("META", "Meta"),
    "amd": ("AMD", "AMD"),
    "超微": ("AMD", "AMD"),
    "奈飞": ("NFLX", "Netflix"),
    "netflix": ("NFLX", "Netflix"),
}

KNOWN_STOCK_TICKERS = {
    "AAPL",
    "AMD",
    "AMZN",
    "COIN",
    "GOOGL",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "PLTR",
    "SMCI",
    "TSLA",
}

CRYPTO_ALIASES = {
    "比特币": ("bitcoin", "BTC"),
    "btc": ("bitcoin", "BTC"),
    "bitcoin": ("bitcoin", "BTC"),
    "以太坊": ("ethereum", "ETH"),
    "eth": ("ethereum", "ETH"),
    "ethereum": ("ethereum", "ETH"),
    "sol": ("solana", "SOL"),
    "solana": ("solana", "SOL"),
    "狗狗币": ("dogecoin", "DOGE"),
    "doge": ("dogecoin", "DOGE"),
    "bnb": ("binancecoin", "BNB"),
    "xrp": ("ripple", "XRP"),
    "ada": ("cardano", "ADA"),
    "sui": ("sui", "SUI"),
}

MARKET_HINTS = (
    "股票",
    "美股",
    "币圈",
    "炒币",
    "加密",
    "crypto",
    "行情",
    "看盘",
    "涨",
    "跌",
    "盘前",
    "盘后",
    "财报",
    "股价",
    "价格",
    "多少",
    "能买吗",
    "还能买",
    "能冲",
    "做多",
    "做空",
    "爆仓",
)


def detect_market_intents(text: str, *, limit: int = 2) -> list[MarketIntent]:
    lowered = text.lower()
    intents: list[MarketIntent] = []
    seen: set[tuple[str, str]] = set()

    for alias, (symbol, display_name) in CRYPTO_ALIASES.items():
        if alias in lowered:
            _append_unique(intents, seen, MarketIntent("crypto", symbol, display_name), limit)

    for alias, (symbol, display_name) in STOCK_ALIASES.items():
        if alias in lowered:
            _append_unique(intents, seen, MarketIntent("stock", symbol, display_name), limit)

    tokens = {
        match.group(1).upper()
        for match in re.finditer(r"(?<![A-Za-z])([A-Za-z]{2,5})(?![A-Za-z])", text)
    }
    has_market_hint = any(hint in lowered for hint in MARKET_HINTS)
    for token in tokens:
        crypto_alias = CRYPTO_ALIASES.get(token.lower())
        if crypto_alias:
            _append_unique(
                intents,
                seen,
                MarketIntent("crypto", crypto_alias[0], crypto_alias[1]),
                limit,
            )
            continue
        if token in KNOWN_STOCK_TICKERS or has_market_hint:
            _append_unique(intents, seen, MarketIntent("stock", token, token), limit)

    return intents[:limit]


def is_market_topic(text: str) -> bool:
    lowered = text.lower()
    if detect_market_intents(text, limit=1):
        return True
    return any(hint in lowered for hint in MARKET_HINTS)


def _append_unique(
    intents: list[MarketIntent],
    seen: set[tuple[str, str]],
    intent: MarketIntent,
    limit: int,
) -> None:
    key = (intent.kind, intent.symbol)
    if key in seen or len(intents) >= limit:
        return
    seen.add(key)
    intents.append(intent)
