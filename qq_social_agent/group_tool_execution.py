from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .approval_models import PendingApprovalCandidate
from .deepseek_client import ReplyDecision
from .pipeline_types import PipelineState, ToolKind, ToolRequest, ToolResult
from .speaker_context import _short_notice_text
from .tool_router import ToolRoutePlan
from .tool_registry import ToolRegistry
from .tools.fresh_context import _compact_search_query
from .tools.market_intent import MarketIntent


@dataclass(frozen=True)
class GroupToolExecutionResult:
    market_context: str
    market_report: str
    fresh_context: str
    direct_candidates: tuple[PendingApprovalCandidate, ...] = ()


async def execute_group_tools(
    *,
    decision: ReplyDecision,
    tool_plan: ToolRoutePlan,
    pipeline_state: PipelineState,
    group_id: int,
    user_id: int,
    text: str,
    market_intents: list[MarketIntent],
    market_context_task: asyncio.Task[ToolResult] | None,
    prefetched_market_request: ToolRequest | None,
    tool_registry: ToolRegistry,
    market_intents_from_decision: Callable[..., list[MarketIntent]],
    execute_fresh_tool_request: Callable[..., Awaitable[ToolResult]],
    fresh_tool_failure_context: Callable[..., str],
    combine_text_sections: Callable[..., str],
    record_metric_event: Callable[..., None],
    logger: Any,
) -> GroupToolExecutionResult:
    market_context = ""
    market_report = ""
    if decision.need_tool and decision.tool == "market":
        requested_intents = market_intents_from_decision(
            decision,
            fallback_text=text,
            fallback_intents=market_intents,
        )
        market_request = tool_plan.first(ToolKind.MARKET) or ToolRequest(
            ToolKind.MARKET,
            query=text,
            reason="decision_requires_market",
            required=True,
            arguments={
                "symbols": tuple(
                    {
                        "kind": item.kind,
                        "symbol": item.symbol,
                        "display": item.display_name,
                    }
                    for item in requested_intents[:2]
                )
            },
        )
        market_result = (
            await market_context_task
            if market_context_task is not None and market_request == prefetched_market_request
            else await tool_registry.execute(market_request)
        )
        pipeline_state.add_tool_result(market_result)
        market_report = market_result.evidence
        market_context = market_result.context
        record_metric_event(
            "tool_call",
            group_id=group_id,
            user_id=user_id,
            stage="market",
            action="registry_execute",
            tool_kind=ToolKind.MARKET.value,
            success=market_result.ok,
            status=market_result.status,
            latency_ms=market_result.elapsed_ms,
            error=market_result.error,
            **dict(market_result.metadata),
        )
        if market_report:
            logger.info(
                "qq_social_agent pending market report approval: "
                f"group={group_id} chars={len(market_report)}"
            )
            if not decision.comment_after_tool:
                market_candidates = (
                    PendingApprovalCandidate(
                        1,
                        market_report,
                        "market_check",
                        "行情工具报告，不额外编判断",
                    ),
                )
                return GroupToolExecutionResult(
                    market_context=market_context,
                    market_report=market_report,
                    fresh_context="",
                    direct_candidates=market_candidates,
                )

    fresh_context = ""
    if decision.need_fresh_context:
        query = _compact_search_query(decision.fresh_query.strip() or text.strip()) or (
            decision.fresh_query.strip() or text.strip()
        )
        fresh_result = await execute_fresh_tool_request(
            ToolRequest(
                ToolKind.FRESH_SEARCH,
                query=query,
                reason="reply_requires_fresh_context",
                required=True,
                arguments={"kind": decision.fresh_kind},
            ),
            metric_stage="fresh_context",
            group_id=group_id,
            user_id=user_id,
        )
        pipeline_state.add_tool_result(fresh_result)
        fresh_context = fresh_result.context
        if str(fresh_result.status) != "ok" and not fresh_context.strip():
            query_text = decision.fresh_query.strip() or text.strip()
            failure_reason = fresh_result.error or fresh_result.status or "搜索工具没有返回可用结果"
            fresh_context = fresh_tool_failure_context(
                query_text,
                status=str(fresh_result.status),
                reason=str(failure_reason),
            )
            record_metric_event(
                "fresh_context_failure_injected",
                group_id=group_id,
                user_id=user_id,
                stage="fresh_context",
                action="generation_context",
                query_preview=_short_notice_text(query_text, 80),
                status=str(fresh_result.status),
                error=_short_notice_text(str(failure_reason), 160),
            )
    deep_request = tool_plan.first(ToolKind.DEEP_URL)
    if deep_request is not None:
        deep_result = await tool_registry.execute(deep_request)
        pipeline_state.add_tool_result(deep_result)
        record_metric_event(
            "tool_call",
            group_id=group_id,
            user_id=user_id,
            stage="deep_url_reader",
            action="registry_execute",
            tool_kind=ToolKind.DEEP_URL.value,
            success=deep_result.ok,
            status=deep_result.status,
            latency_ms=deep_result.elapsed_ms,
            error=deep_result.error,
            **dict(deep_result.metadata),
        )
        if deep_result.context:
            fresh_context = combine_text_sections(fresh_context, deep_result.context)
    probability_request = tool_plan.first(ToolKind.PROBABILITY)
    if probability_request is not None:
        probability_result = await tool_registry.execute(probability_request)
        pipeline_state.add_tool_result(probability_result)
        record_metric_event(
            "tool_call",
            group_id=group_id,
            user_id=user_id,
            stage="probability",
            action="registry_execute",
            tool_kind=ToolKind.PROBABILITY.value,
            success=probability_result.ok,
            status=probability_result.status,
            latency_ms=probability_result.elapsed_ms,
            error=probability_result.error,
            **dict(probability_result.metadata),
        )
        if probability_result.context:
            fresh_context = combine_text_sections(fresh_context, probability_result.context)

    return GroupToolExecutionResult(
        market_context=market_context,
        market_report=market_report,
        fresh_context=fresh_context,
    )
