from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable

from .deepseek_client import DeepSeekClient, ReplyDecision, ToolSymbol
from .member_context import member_label as _member_label
from .memory import ChatMessage
from .pipeline_types import ToolKind, ToolRequest
from .speaker_context import _short_notice_text
from .tool_router import ToolRoutePlan
from .tools.fresh_context import _compact_search_query
from .tools.market_intent import MarketIntent


def _tool_plan_with_runtime_context(
    plan: ToolRoutePlan,
    *,
    addressed: bool,
    group_id: int,
    user_id: int,
    source_message_id: str,
) -> ToolRoutePlan:
    requests: list[ToolRequest] = []
    for request in plan.requests:
        arguments = dict(request.arguments)
        if request.kind is ToolKind.DEEP_URL:
            arguments.update(
                {
                    "addressed": addressed,
                    "group_id": group_id,
                    "user_id": user_id,
                    "source_message_id": source_message_id,
                }
            )
        requests.append(replace(request, arguments=arguments))
    return ToolRoutePlan(tuple(requests), source=plan.source)


def _tool_router_backend_hint(
    fresh_intent: object | None,
    market_intents: list[MarketIntent],
    tool_plan: ToolRoutePlan,
) -> str:
    parts: list[str] = []
    if fresh_intent is not None:
        query = str(getattr(fresh_intent, "query", "") or "").strip()
        kind = str(getattr(fresh_intent, "kind", "web") or "web").strip()
        explicit = bool(getattr(fresh_intent, "explicit", False))
        required = bool(getattr(fresh_intent, "required", False))
        if query:
            signals: list[str] = []
            if explicit:
                signals.append("显式搜索")
            if required:
                signals.append("需要核验最新事实")
            if not signals:
                signals.append("可能涉及最新背景")
            parts.append(f"{'/'.join(signals)}；候选 query={query}；kind={kind}")
    if market_intents:
        symbols = ", ".join(
            f"{item.display_name or item.symbol}({item.kind}:{item.symbol})"
            for item in market_intents[:2]
        )
        if symbols:
            parts.append(f"行情候选：{symbols}")
    if tool_plan.requests:
        planned = ", ".join(
            f"{request.kind.value}:{_short_notice_text(request.query, 48)}"
            for request in tool_plan.requests
        )
        parts.append(f"后端确定性计划：{planned}")
    return "；".join(part for part in parts if part)


def _tool_router_should_run(
    decision: ReplyDecision,
    *,
    text: str,
    addressed_bot: bool,
    fresh_intent: object | None,
    market_intents: list[MarketIntent],
    tool_plan: ToolRoutePlan,
) -> bool:
    if not decision.should_reply:
        return False
    if decision.need_fresh_context or decision.need_tool or tool_plan.requests:
        return True
    if fresh_intent is not None or market_intents:
        return True
    if addressed_bot:
        return True
    if re.search(r"https?://", text, re.IGNORECASE):
        return True
    compact = re.sub(r"\s+", "", text.casefold())
    proactive_terms = (
        "最近",
        "最新",
        "现在",
        "今年",
        "今天",
        "刚刚",
        "刚才",
        "新闻",
        "新专辑",
        "新歌",
        "新版本",
        "发布",
        "官宣",
        "赛程",
        "比分",
        "结果",
        "政策",
        "发生什么",
        "怎么了",
        "股票",
        "美股",
        "币价",
        "比特币",
        "以太坊",
    )
    return any(term in compact for term in proactive_terms)


def _tool_request_from_llm_route(route: object, *, fallback_text: str) -> ToolRequest | None:
    tool = str(getattr(route, "tool", "none") or "none").strip().lower()
    query = str(getattr(route, "query", "") or "").strip()
    reason = str(getattr(route, "reason", "") or "").strip() or "llm_tool_router"
    try:
        confidence = float(getattr(route, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if tool == "fresh_search":
        if not query:
            return None
        kind = str(getattr(route, "kind", "web") or "web").strip().lower()
        if kind not in {"news", "sports", "web"}:
            kind = "web"
        compacted = _compact_search_query(query) or query
        queries = tuple(
            (_compact_search_query(str(item)) or str(item).strip())[:160]
            for item in getattr(route, "queries", ()) or ()
            if str(item or "").strip()
        )
        if compacted and compacted not in queries:
            queries = (compacted, *queries)[:4]
        return ToolRequest(
            ToolKind.FRESH_SEARCH,
            query=compacted[:160],
            reason=f"llm_tool_router:{reason}"[:120],
            confidence=confidence,
            required=True,
            arguments={"kind": kind, "queries": queries},
        )
    if tool == "market":
        raw_symbols = tuple(getattr(route, "symbols", ()) or ())
        symbols = tuple(
            {
                "kind": str(getattr(item, "kind", "") or ""),
                "symbol": str(getattr(item, "symbol", "") or ""),
                "display": str(getattr(item, "display", "") or getattr(item, "symbol", "") or ""),
            }
            for item in raw_symbols
            if str(getattr(item, "symbol", "") or "").strip()
        )
        if not symbols:
            return None
        return ToolRequest(
            ToolKind.MARKET,
            query=query[:160] or fallback_text[:160],
            reason=f"llm_tool_router:{reason}"[:120],
            confidence=confidence,
            required=True,
            arguments={"symbols": symbols},
        )
    if tool == "deep_url":
        if not query:
            query = fallback_text
        if re.search(r"https?://", query, re.IGNORECASE) is None:
            return None
        return ToolRequest(
            ToolKind.DEEP_URL,
            query=query[:500],
            reason=f"llm_tool_router:{reason}"[:120],
            confidence=confidence,
            required=False,
            arguments={},
        )
    if tool == "probability":
        return ToolRequest(
            ToolKind.PROBABILITY,
            query=(query or fallback_text)[:240],
            reason=f"llm_tool_router:{reason}"[:120],
            confidence=confidence,
            required=True,
            arguments={"context": fallback_text[:1200]},
        )
    return None


def _merge_tool_route_plans(base: ToolRoutePlan, extra: ToolRoutePlan) -> ToolRoutePlan:
    if not extra.requests:
        return base
    requests = list(base.requests)
    for request in extra.requests:
        replaced = False
        for index, existing in enumerate(requests):
            if existing.kind is not request.kind:
                continue
            if existing.required and not request.required:
                replaced = True
                break
            requests[index] = request
            replaced = True
            break
        if not replaced:
            requests.append(request)
    source = extra.source if not base.requests else f"{base.source}+{extra.source}"
    return ToolRoutePlan(tuple(requests), source=source)


_NEARBY_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_NEARBY_URL_HINT_RE = re.compile(
    r"仓库|链接|github|不是这个|page not find|404|这是什么|这个链接|这个仓库",
    re.IGNORECASE,
)


def _recent_http_urls(
    text: str,
    recent_messages: list[ChatMessage] | None = None,
    *,
    limit: int = 4,
) -> tuple[str, ...]:
    blobs: list[str] = [str(text or "")]
    for message in reversed(list(recent_messages or ())[-limit:]):
        blobs.append(str(getattr(message, "text", "") or ""))
    found: list[str] = []
    seen: set[str] = set()
    for blob in blobs:
        for match in _NEARBY_URL_RE.findall(blob):
            url = match.rstrip(".,;，。！？)>\"']")
            key = url.casefold()
            if not url or key in seen:
                continue
            seen.add(key)
            found.append(url)
            if len(found) >= 3:
                return tuple(found)
    return tuple(found)


def _nearby_url_tool_plan(
    *,
    text: str,
    recent_messages: list[ChatMessage] | None,
    addressed: bool,
    existing: ToolRoutePlan,
) -> ToolRoutePlan:
    if not addressed or existing.first(ToolKind.DEEP_URL) is not None:
        return ToolRoutePlan((), source="nearby_url")
    urls = _recent_http_urls(text, recent_messages, limit=4)
    if not urls:
        return ToolRoutePlan((), source="nearby_url")
    if _NEARBY_URL_HINT_RE.search(str(text or "")) is None:
        return ToolRoutePlan((), source="nearby_url")
    return ToolRoutePlan(
        (
            ToolRequest(
                ToolKind.DEEP_URL,
                query=urls[0][:500],
                reason="nearby_url_in_recent_messages",
                required=False,
                arguments={},
            ),
        ),
        source="nearby_url",
    )


def _finalize_routed_tool_plan(
    decision: ReplyDecision,
    tool_plan: ToolRoutePlan,
    *,
    text: str,
    context_recent: list[ChatMessage],
    addressed_bot: bool,
    group_id: int,
    user_id: int,
    source_message_id: str,
    routed: object | None = None,
) -> tuple[ReplyDecision, ToolRoutePlan]:
    final_plan = tool_plan
    nearby_url_plan = _nearby_url_tool_plan(
        text=text,
        recent_messages=context_recent,
        addressed=addressed_bot,
        existing=final_plan,
    )
    if nearby_url_plan.requests:
        routed_nearby = _tool_plan_with_runtime_context(
            nearby_url_plan,
            addressed=addressed_bot,
            group_id=group_id,
            user_id=user_id,
            source_message_id=source_message_id,
        )
        final_plan = _merge_tool_route_plans(final_plan, routed_nearby)
    fresh_request = final_plan.first(ToolKind.FRESH_SEARCH)
    market_request = final_plan.first(ToolKind.MARKET)
    if fresh_request is not None and fresh_request.required:
        compacted_query = _compact_search_query(fresh_request.query) or fresh_request.query
        decision = replace(
            decision,
            need_fresh_context=True,
            fresh_query=compacted_query[:120],
            fresh_kind=str(fresh_request.arguments.get("kind", "web") or "web"),
        )
    elif (
        decision.need_fresh_context
        and routed is not None
        and str(getattr(routed, "tool", "none") or "none") == "none"
    ):
        decision = replace(decision, need_fresh_context=False, fresh_query="", fresh_kind="web")
    if market_request is not None and market_request.required:
        symbols = tuple(
            ToolSymbol(
                kind=str(item.get("kind", "")),
                symbol=str(item.get("symbol", "")),
                display=str(item.get("display", "")),
            )
            for item in tuple(market_request.arguments.get("symbols", ()))
            if isinstance(item, dict) and item.get("symbol")
        )
        decision = replace(
            decision,
            should_reply=True,
            action="market_check",
            need_tool=True,
            tool="market",
            symbols=symbols,
            comment_after_tool=bool(getattr(routed, "comment_after_tool", decision.comment_after_tool)),
        )
    probability_request = final_plan.first(ToolKind.PROBABILITY)
    if probability_request is not None and probability_request.required:
        decision = replace(
            decision,
            should_reply=True,
            action="answer" if decision.action in {"ignore", "fresh_context"} else decision.action,
            need_tool=True,
            tool="probability",
        )
    return decision, final_plan


async def _apply_tool_use_router(
    decision: ReplyDecision,
    *,
    tool_plan: ToolRoutePlan,
    persona: object,
    context_recent: list[ChatMessage],
    text: str,
    nickname: str,
    addressed_bot: bool,
    fresh_intent: object | None,
    market_intents: list[MarketIntent],
    speaker_context: str,
    group_id: int,
    user_id: int,
    source_message_id: str,
    chat_label: str = "QQ 群聊",
    client: DeepSeekClient | None,
    record_metric_event: Callable[..., None],
    logger: Any,
) -> tuple[ReplyDecision, ToolRoutePlan]:
    router_should_run = _tool_router_should_run(
        decision,
        text=text,
        addressed_bot=addressed_bot,
        fresh_intent=fresh_intent,
        market_intents=market_intents,
        tool_plan=tool_plan,
    )
    if client is None or not router_should_run:
        record_metric_event(
            "tool_router",
            group_id=group_id,
            user_id=user_id,
            stage="routing",
            action="not_run",
            routed_tool="none",
            decision_should_reply=decision.should_reply,
            decision_need_fresh=decision.need_fresh_context,
            decision_need_tool=decision.need_tool,
            existing_requests=list(tool_plan.kinds),
            reason="client_not_ready" if client is None else "router_gate_false",
        )
        return _finalize_routed_tool_plan(
            decision,
            tool_plan,
            text=text,
            context_recent=context_recent,
            addressed_bot=addressed_bot,
            group_id=group_id,
            user_id=user_id,
            source_message_id=source_message_id,
        )
    backend_hint = _tool_router_backend_hint(fresh_intent, market_intents, tool_plan)
    try:
        routed = await client.route_tool_use(
            persona=persona,
            recent_messages=context_recent,
            current_text=text,
            current_nickname=_member_label(user_id, nickname),
            addressed=addressed_bot,
            decision_action=decision.action,
            decision_reason=decision.reason,
            backend_hint=backend_hint,
            chat_label=chat_label,
            speaker_context=speaker_context,
        )
    except Exception as exc:
        logger.warning(
            "qq_social_agent tool router failed: "
            f"group={group_id} error={exc}"
        )
        record_metric_event(
            "tool_router",
            group_id=group_id,
            user_id=user_id,
            stage="routing",
            action="failed",
            routed_tool="none",
            decision_should_reply=decision.should_reply,
            error=_short_notice_text(str(exc), 200),
            backend_hint=backend_hint,
        )
        return _finalize_routed_tool_plan(
            decision,
            tool_plan,
            text=text,
            context_recent=context_recent,
            addressed_bot=addressed_bot,
            group_id=group_id,
            user_id=user_id,
            source_message_id=source_message_id,
        )
    request = _tool_request_from_llm_route(routed, fallback_text=text)
    final_plan = tool_plan
    if request is not None:
        routed_plan = _tool_plan_with_runtime_context(
            ToolRoutePlan((request,), source="llm_tool_router"),
            addressed=addressed_bot,
            group_id=group_id,
            user_id=user_id,
            source_message_id=source_message_id,
        )
        final_plan = _merge_tool_route_plans(tool_plan, routed_plan)
    record_metric_event(
        "tool_router",
        group_id=group_id,
        user_id=user_id,
        stage="routing",
        action=str(getattr(routed, "tool", "none") or "none"),
        routed_tool=str(getattr(routed, "tool", "none") or "none"),
        query_preview=_short_notice_text(str(getattr(routed, "query", "") or ""), 80),
        kind=str(getattr(routed, "kind", "web") or "web"),
        confidence=round(float(getattr(routed, "confidence", 0.0) or 0.0), 3),
        reason=str(getattr(routed, "reason", "") or ""),
        backend_hint=backend_hint,
        accepted=request is not None,
        final_requests=list(final_plan.kinds),
    )
    return _finalize_routed_tool_plan(
        decision,
        final_plan,
        text=text,
        context_recent=context_recent,
        addressed_bot=addressed_bot,
        group_id=group_id,
        user_id=user_id,
        source_message_id=source_message_id,
        routed=routed,
    )
