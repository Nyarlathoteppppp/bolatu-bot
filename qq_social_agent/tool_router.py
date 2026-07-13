from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .deepseek_client import ReplyDecision, ToolSymbol
from .pipeline_types import ToolKind, ToolRequest
from .tools.fresh_context import FreshIntent
from .tools.market_intent import MarketIntent


URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
ACADEMIC_TOPIC_TERMS = (
    "猜想",
    "定理",
    "公理",
    "引理",
    "学术问题",
    "论文",
    "证明",
    "希尔伯特第六问题",
    "挂谷",
)
ACADEMIC_QUESTION_TERMS = (
    "是什么",
    "讲什么",
    "解决了吗",
    "解决了没",
    "证明了吗",
    "最新进展",
    "有什么进展",
    "怎么理解",
)


@dataclass(frozen=True)
class ToolRoutePlan:
    requests: tuple[ToolRequest, ...] = ()
    source: str = "deterministic"

    def first(self, kind: ToolKind) -> ToolRequest | None:
        return next((item for item in self.requests if item.kind == kind), None)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(item.kind.value for item in self.requests)


@dataclass(frozen=True)
class ToolRouteComparison:
    matched: bool
    legacy_kinds: tuple[str, ...]
    routed_kinds: tuple[str, ...]


def route_tools(
    text: str,
    *,
    market_intents: list[MarketIntent],
    fresh_intent: FreshIntent | None,
    addressed: bool,
    market_required: bool = False,
) -> ToolRoutePlan:
    requests: list[ToolRequest] = []
    if market_intents and market_required:
        requests.append(
            ToolRequest(
                ToolKind.MARKET,
                query=text,
                reason="detected_market_symbol",
                required=True,
                arguments={
                    "symbols": tuple(
                        {"kind": item.kind, "symbol": item.symbol, "display": item.display_name}
                        for item in market_intents[:2]
                    )
                },
            )
        )
    if fresh_intent is not None and (fresh_intent.explicit or fresh_intent.required):
        requests.append(
            ToolRequest(
                ToolKind.FRESH_SEARCH,
                query=fresh_intent.query,
                reason="explicit_or_required_fresh_context",
                required=True,
                arguments={"kind": fresh_intent.kind},
            )
        )
    elif _is_academic_lookup(text):
        requests.append(
            ToolRequest(
                ToolKind.FRESH_SEARCH,
                query=text.strip()[:120],
                reason="academic_concept_or_status",
                confidence=0.92,
                required=True,
                arguments={"kind": "web"},
            )
        )
    if addressed and URL_RE.search(text):
        requests.append(
            ToolRequest(
                ToolKind.DEEP_URL,
                query=text,
                reason="addressed_url",
                required=False,
            )
        )
    return ToolRoutePlan(tuple(_dedupe_requests(requests)))


def apply_tool_plan(decision: ReplyDecision, plan: ToolRoutePlan) -> ReplyDecision:
    """Apply deterministic required routes while preserving the social decision."""

    result = decision
    market = plan.first(ToolKind.MARKET)
    if market is not None and market.required:
        raw_symbols = market.arguments.get("symbols", ())
        symbols = tuple(
            ToolSymbol(
                kind=str(item.get("kind", "")),
                symbol=str(item.get("symbol", "")),
                display=str(item.get("display", "")),
            )
            for item in raw_symbols
            if isinstance(item, dict) and item.get("symbol")
        )
        result = replace(
            result,
            should_reply=True,
            action="market_check",
            need_tool=True,
            tool="market",
            symbols=symbols,
        )
    fresh = plan.first(ToolKind.FRESH_SEARCH)
    if fresh is not None and fresh.required:
        result = replace(
            result,
            should_reply=True,
            action="answer" if result.action in {"ignore", "fresh_context"} else result.action,
            need_fresh_context=True,
            fresh_query=fresh.query,
            fresh_kind=str(fresh.arguments.get("kind", "web")),
        )
    return result


def compare_legacy_decision(decision: ReplyDecision, plan: ToolRoutePlan) -> ToolRouteComparison:
    legacy: list[str] = []
    if decision.need_tool and decision.tool == "market":
        legacy.append(ToolKind.MARKET.value)
    if decision.need_fresh_context:
        legacy.append(ToolKind.FRESH_SEARCH.value)
    legacy_kinds = tuple(sorted(set(legacy)))
    routed_kinds = tuple(sorted(set(plan.kinds)))
    return ToolRouteComparison(legacy_kinds == routed_kinds, legacy_kinds, routed_kinds)


def _dedupe_requests(requests: list[ToolRequest]) -> list[ToolRequest]:
    output: list[ToolRequest] = []
    seen: set[ToolKind] = set()
    for request in requests:
        if request.kind in seen:
            continue
        seen.add(request.kind)
        output.append(request)
    return output


def _is_academic_lookup(text: str) -> bool:
    clean = re.sub(r"\s+", "", text)
    return any(term in clean for term in ACADEMIC_TOPIC_TERMS) and any(
        term in clean for term in ACADEMIC_QUESTION_TERMS
    )
