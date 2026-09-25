from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from .deepseek_client import ReplyDecision
from .memory import ChatMessage, MemoryStore
from .pipeline_types import ToolKind, ToolRequest
from .rag_retriever import RAGRetrievalResult, RAGService
from .rate_limiter import RateLimiter
from .tool_registry import ToolRegistry
from .tool_router import ToolRoutePlan
from .private_message_types import PrivateToolStage, PrivateTurn


@dataclass(frozen=True)
class PrivateToolServices:
    memory: MemoryStore
    rag_service: RAGService
    rate_limiter: RateLimiter
    personas: Any
    app_config: Any
    get_deepseek_client: Callable[[], Any]
    tool_registry: ToolRegistry
    route_tools: Callable[..., ToolRoutePlan]
    tool_plan_with_runtime_context: Callable[..., ToolRoutePlan]
    apply_backend_tool_decision: Callable[..., ReplyDecision]
    apply_tool_plan: Callable[[ReplyDecision, ToolRoutePlan], ReplyDecision]
    is_explicit_market_lookup: Callable[[str], bool]
    market_intents_from_decision: Callable[..., Any]
    apply_tool_use_router: Callable[..., Awaitable[tuple[ReplyDecision, ToolRoutePlan]]]
    execute_fresh_tool_request: Callable[..., Awaitable[Any]]
    compact_search_query: Callable[[str], str]
    fresh_tool_failure_context: Callable[..., str]
    normalize_rag_query: Callable[[str], Any]
    detect_market_intents: Callable[..., Any]
    detect_fresh_intent: Callable[[str], Any]
    without_current_message: Callable[..., list[ChatMessage]]
    combine_text_sections: Callable[..., str]
    private_conversation_state_context: Callable[[int], str]
    private_priority_context: Callable[[int], str]
    member_label: Callable[[int, str], str]
    record_metric_event: Callable[..., None]
    logger: Any
    private_context_limit: int
    mid_memory_keep_summaries: int


async def plan_and_execute_private_tools(
    turn: PrivateTurn,
    *,
    services: PrivateToolServices,
) -> PrivateToolStage | None:
    state = services.memory.group_state(turn.chat_id)
    if not bool(state["enabled"]):
        services.logger.info(f"qq_social_agent ignored private: user={turn.user_id} disabled")
        return None

    persona_id = str(state["persona"] or services.app_config.default_persona)
    persona = services.personas.get(persona_id)
    recent = services.memory.recent_messages(turn.chat_id, services.private_context_limit)
    context_recent = services.without_current_message(recent, user_id=turn.user_id, text=turn.text)
    normalized_rag_query = services.normalize_rag_query(turn.text)
    context_query = normalized_rag_query.current_utterance or turn.text
    market_intents = services.detect_market_intents(context_query, limit=2)
    rate = services.rate_limiter.allow(turn.chat_id, mentioned=True)
    if not rate.allowed:
        services.logger.info(
            f"qq_social_agent suppressed private by rate: user={turn.user_id} reason={rate.reason}"
        )
        return None

    deepseek_client = services.get_deepseek_client()
    if deepseek_client is None:
        services.logger.warning("qq_social_agent skipped private: deepseek_client_not_ready")
        return None

    market_intents = services.detect_market_intents(context_query, limit=2)
    fresh_intent = services.detect_fresh_intent(context_query)
    tool_plan = services.tool_plan_with_runtime_context(
        services.route_tools(
            context_query,
            market_intents=market_intents,
            fresh_intent=fresh_intent,
            addressed=True,
            market_required=bool(market_intents) and services.is_explicit_market_lookup(context_query),
        ),
        addressed=True,
        group_id=turn.chat_id,
        user_id=turn.user_id,
        source_message_id=turn.source_message_id,
    )
    decision = services.apply_backend_tool_decision(
        ReplyDecision(
            should_reply=True,
            confidence=1.0,
            reason="private_direct_conversation",
            mode="reply",
            action="answer",
        ),
        text=context_query,
        market_intents=market_intents,
        fresh_intent=fresh_intent,
    )
    decision = services.apply_tool_plan(decision, tool_plan)
    speaker_context = (
        f"当前是和{services.member_label(turn.user_id, turn.nickname)}的一对一私聊。"
        "不要把普通代词误当成群友或风雪自己；不要艾特第三人、点群表情或假装在群里说话。"
    )
    private_state_context = services.private_conversation_state_context(turn.chat_id)
    decision, tool_plan = await services.apply_tool_use_router(
        decision,
        tool_plan=tool_plan,
        persona=persona,
        context_recent=context_recent,
        text=context_query,
        nickname=turn.nickname,
        addressed_bot=True,
        fresh_intent=fresh_intent,
        market_intents=market_intents,
        speaker_context=speaker_context,
        group_id=turn.chat_id,
        user_id=turn.user_id,
        source_message_id=turn.source_message_id,
        chat_label="QQ 私聊",
    )
    services.record_metric_event(
        "private_tool_route_plan",
        group_id=turn.chat_id,
        user_id=turn.user_id,
        stage="routing",
        action=decision.action,
        need_fresh=decision.need_fresh_context,
        need_tool=decision.need_tool,
        requests=[request.kind.value for request in tool_plan.requests],
    )

    # Start RAG before external tools so retrieval remains parallel with tool execution.
    rag_task: asyncio.Task[RAGRetrievalResult] = asyncio.create_task(
        services.rag_service.retrieve(
            group_id=turn.chat_id,
            query=context_query,
            addressed=True,
            related_user_ids=[turn.user_id],
            excluded_user_ids=[turn.self_id],
            include_conversation=False,
        )
    )
    market_context = ""
    if decision.need_tool and decision.tool == "market":
        requested_intents = services.market_intents_from_decision(
            decision,
            fallback_text=context_query,
            fallback_intents=market_intents,
        )
        market_request = tool_plan.first(ToolKind.MARKET) or ToolRequest(
            ToolKind.MARKET,
            query=context_query,
            reason="private_reply_requires_market",
            required=True,
            arguments={
                "symbols": tuple(
                    {"kind": item.kind, "symbol": item.symbol, "display": item.display_name}
                    for item in requested_intents[:2]
                )
            },
        )
        market_result = await services.tool_registry.execute(market_request)
        market_context = market_result.context
        services.record_metric_event(
            "tool_call",
            group_id=turn.chat_id,
            user_id=turn.user_id,
            stage="private_market",
            action="registry_execute",
            tool_kind=ToolKind.MARKET.value,
            success=market_result.ok,
            status=market_result.status,
            latency_ms=market_result.elapsed_ms,
            error=market_result.error,
            **dict(market_result.metadata),
        )

    fresh_context = ""
    if decision.need_fresh_context:
        raw_query = decision.fresh_query.strip() or context_query
        query = services.compact_search_query(raw_query) or raw_query
        fresh_result = await services.execute_fresh_tool_request(
            ToolRequest(
                ToolKind.FRESH_SEARCH,
                query=query,
                reason="private_reply_requires_fresh_context",
                required=True,
                arguments={"kind": decision.fresh_kind},
            ),
            metric_stage="private_fresh_context",
            group_id=turn.chat_id,
            user_id=turn.user_id,
        )
        fresh_context = fresh_result.context
        if str(fresh_result.status) != "ok" and not fresh_context.strip():
            fresh_context = services.fresh_tool_failure_context(
                query,
                status=str(fresh_result.status),
                reason=str(fresh_result.error or fresh_result.status or "搜索工具没有返回可用结果"),
            )
    deep_request = tool_plan.first(ToolKind.DEEP_URL)
    if deep_request is not None:
        deep_result = await services.tool_registry.execute(deep_request)
        if deep_result.context:
            fresh_context = services.combine_text_sections(fresh_context, deep_result.context)
        services.record_metric_event(
            "tool_call",
            group_id=turn.chat_id,
            user_id=turn.user_id,
            stage="private_deep_url_reader",
            action="registry_execute",
            tool_kind=ToolKind.DEEP_URL.value,
            success=deep_result.ok,
            status=deep_result.status,
            latency_ms=deep_result.elapsed_ms,
            error=deep_result.error,
            **dict(deep_result.metadata),
        )
    probability_request = tool_plan.first(ToolKind.PROBABILITY)
    if probability_request is not None:
        probability_result = await services.tool_registry.execute(
            replace(
                probability_request,
                arguments={
                    **dict(probability_request.arguments),
                    "context": str(probability_request.arguments.get("context") or context_query)[:1200],
                },
            )
        )
        if probability_result.context:
            fresh_context = services.combine_text_sections(fresh_context, probability_result.context)
        services.record_metric_event(
            "tool_call",
            group_id=turn.chat_id,
            user_id=turn.user_id,
            stage="private_probability",
            action="registry_execute",
            tool_kind=ToolKind.PROBABILITY.value,
            success=probability_result.ok,
            status=probability_result.status,
            latency_ms=probability_result.elapsed_ms,
            error=probability_result.error,
            **dict(probability_result.metadata),
        )
    return PrivateToolStage(
        turn=turn,
        persona=persona,
        context_recent=tuple(context_recent),
        context_query=context_query,
        decision=decision,
        tool_plan=tool_plan,
        speaker_context=speaker_context,
        private_state_context=private_state_context,
        market_context=market_context,
        fresh_context=fresh_context,
        rag_task=rag_task,
    )
