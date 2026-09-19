from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from .deepseek_client import ReplyDecision, ToolSymbol
from .pipeline_types import PipelineMode, ToolKind, ToolRequest
from .tools.fresh_context import FreshIntent, _compact_search_query, detect_fresh_intent
from .tools.market_intent import MarketIntent


PROBABILITY_TERMS = (
    "概率",
    "几率",
    "可能性",
    "可能有多大",
    "有多大可能",
    "能成吗",
    "算算概率",
    "推演概率",
    "几成把握",
    "拿到 offer",
    "能拿到",
)


def is_probability_lookup(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return any(term in compact for term in PROBABILITY_TERMS)


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
FOLLOWUP_LOOKUP_RE = re.compile(
    r"(?:帮我|给我|你)?\s*(?:继续|接着|再)?\s*"
    r"(?:研究研究|研究一下|深入研究|详细查查|详细查一下|展开查|展开看看|具体看看|继续搜|接着搜|再查查|搜一下|搜索一下|查一下|查查|搜搜)"
)
BARE_FOLLOWUP_LOOKUP_RE = re.compile(
    r"^\s*(?:你)?\s*(?:帮我)?\s*(?:搜一下|搜索一下|查一下|查查|搜搜)\s*[吧呀啊。！!]*\s*$",
    re.IGNORECASE,
)
FOLLOWUP_TOPIC_MAX_AGE_SECONDS = 120.0


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
                query=(_compact_search_query(text) or text.strip())[:120],
                reason="academic_concept_or_status",
                confidence=0.92,
                required=True,
                arguments={"kind": "web"},
            )
        )
    if is_probability_lookup(text) and addressed:
        requests.append(
            ToolRequest(
                ToolKind.PROBABILITY,
                query=text.strip(),
                reason="probability_lookup",
                confidence=0.95,
                required=True,
            )
        )
    if addressed and URL_RE.search(text):
        requests.append(
            ToolRequest(
                ToolKind.DEEP_URL,
                query=text,
                reason="addressed_url",
                required=False,
                arguments={"addressed": True},
            )
        )
    return ToolRoutePlan(tuple(_dedupe_requests(requests)))


def route_mode(plan: ToolRoutePlan) -> PipelineMode:
    if plan.first(ToolKind.MARKET) is not None:
        return PipelineMode.MARKET
    if plan.first(ToolKind.FRESH_SEARCH) is not None:
        return PipelineMode.SEARCH
    if plan.first(ToolKind.PROBABILITY) is not None:
        return PipelineMode.PROBABILITY
    if plan.first(ToolKind.DEEP_URL) is not None:
        return PipelineMode.DEEP_URL
    return PipelineMode.CHAT


def infer_followup_fresh_intent(
    text: str,
    recent_messages: Iterable[object],
    *,
    addressed: bool,
    current_user_id: int | None = None,
    current_at: float | None = None,
) -> FreshIntent | None:
    """Turn an addressed "research this further" turn into a concrete lookup.

    The topic is taken from the latest substantive user message rather than
    from the bot's previous answer, so an earlier hallucination cannot become
    the next search query.
    """

    current_text = _current_reply_text(text) or re.sub(r"\s+", " ", str(text or "")).strip()
    bare_command = is_contextual_followup_lookup(current_text)
    if FOLLOWUP_LOOKUP_RE.search(current_text) is None:
        return None
    if not addressed and not bare_command:
        return None
    current_intent = detect_fresh_intent(current_text)
    if current_intent is not None and current_intent.explicit and current_intent.query:
        return FreshIntent(current_intent.query[:120], current_intent.kind, explicit=True, required=True)
    reply_target = _reply_target_lookup_topic(text, current_text=current_text)
    if reply_target and _useful_followup_topic(reply_target, current_text=current_text):
        kind = "news" if any(token in reply_target for token in ("现在", "今天", "最新", "今年", "刚刚")) else "web"
        return FreshIntent(reply_target[:120], kind, explicit=True, required=True)
    messages = tuple(recent_messages)
    now = float(current_at or 0.0) or max(
        (float(getattr(message, "created_at", 0.0) or 0.0) for message in messages),
        default=0.0,
    )
    passes = (True, False) if current_user_id else (False,)
    for same_user_only in passes:
        for message in reversed(messages):
            if bool(getattr(message, "is_bot", False)):
                continue
            message_user_id = int(getattr(message, "user_id", 0) or 0)
            if same_user_only and message_user_id != int(current_user_id or 0):
                continue
            if bare_command and not addressed and current_user_id and message_user_id != int(current_user_id):
                continue
            message_at = float(getattr(message, "created_at", 0.0) or 0.0)
            if bare_command and now > 0 and message_at > 0 and now - message_at > FOLLOWUP_TOPIC_MAX_AGE_SECONDS:
                continue
            candidate = re.sub(r"\s+", " ", str(getattr(message, "text", "") or "")).strip()
            if not _useful_followup_topic(candidate, current_text=current_text):
                continue
            kind = "news" if any(token in candidate for token in ("现在", "今天", "最新", "今年", "刚刚")) else "web"
            return FreshIntent(candidate[:120], kind, explicit=True, required=True)
    return None


def _current_reply_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if "回复" not in value or "消息【" not in value or not value.endswith("】"):
        return ""
    for separator in ("：", ":"):
        if separator not in value:
            continue
        current = value.rsplit(separator, 1)[-1].removesuffix("】").strip()
        if current:
            return current
    return ""


def _reply_target_lookup_topic(text: str, *, current_text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if "回复" not in value or "消息【" not in value or not value.endswith("】"):
        return ""
    if FOLLOWUP_LOOKUP_RE.search(current_text) is None:
        return ""
    match = re.search(r"消息【.*?说[：:](?P<target>.*?)；.*?回复.*?[：:](?P<current>.*?)】$", value)
    if match is None:
        return ""
    target = re.sub(r"\s+", " ", match.group("target") or "").strip()
    return target[:120]


def is_contextual_followup_lookup(text: str) -> bool:
    """Whether this turn is a query-less command such as ``搜一下``."""

    return BARE_FOLLOWUP_LOOKUP_RE.fullmatch(str(text or "")) is not None


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
    probability = plan.first(ToolKind.PROBABILITY)
    if probability is not None and probability.required:
        result = replace(
            result,
            should_reply=True,
            action="answer" if result.action in {"ignore", "fresh_context"} else result.action,
            need_tool=True,
            tool="probability",
        )
    return result


def compare_legacy_decision(decision: ReplyDecision, plan: ToolRoutePlan) -> ToolRouteComparison:
    legacy: list[str] = []
    if decision.need_tool and decision.tool == "market":
        legacy.append(ToolKind.MARKET.value)
    if decision.need_fresh_context:
        legacy.append(ToolKind.FRESH_SEARCH.value)
    if decision.need_tool and decision.tool == "probability":
        legacy.append(ToolKind.PROBABILITY.value)
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


def _useful_followup_topic(candidate: str, *, current_text: str) -> bool:
    compact = re.sub(r"[\s，。！？,.!?]+", "", candidate)
    if len(compact) < 4 or candidate == current_text:
        return False
    if FOLLOWUP_LOOKUP_RE.search(candidate):
        return False
    if candidate.startswith("[") and any(
        marker in candidate for marker in ("[图片", "[表情包", "[语音", "[视频")
    ):
        return False
    return True
