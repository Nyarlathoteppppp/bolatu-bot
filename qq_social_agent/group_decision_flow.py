from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot

from .decision_gate import (
    PreDecisionGateResult,
    apply_backend_tool_decision as _apply_backend_tool_decision,
)
from .deepseek_client import DeepSeekClient, ReplyDecision
from .discourse_state import DiscourseState
from .jev_policy import GroupReplyBudget
from .member_context import member_label as _member_label
from .memory import ChatMessage
from .pipeline_stages import apply_decision as _pipeline_apply_decision, mark_gated as _pipeline_mark_gated
from .pipeline_types import PipelineState
from .resolver_result import RESOLVED
from .speaker_context import _short_notice_text
from .tool_router import ToolRoutePlan, apply_tool_plan as _apply_tool_plan, route_mode as _tool_route_mode
from .tools.market_intent import MarketIntent


@dataclass(frozen=True)
class GroupDecisionServices:
    record_metric_event: Callable[..., None]
    record_tool_router_shadow: Callable[..., None]
    send_suppression_notice: Callable[..., Awaitable[None]]
    schedule_group_learning: Callable[[int], None]
    decision_failure_fallback: Callable[..., ReplyDecision | None]
    looks_like_addressed_question: Callable[[str], bool]
    apply_tool_use_router: Callable[..., Awaitable[tuple[ReplyDecision, ToolRoutePlan]]]
    enforce_addressed_reply_decision: Callable[..., ReplyDecision]
    maybe_apply_speaking_action: Callable[..., Awaitable[ReplyDecision]]
    maybe_apply_ask_back: Callable[..., Awaitable[ReplyDecision]]
    logger: Any


@dataclass(frozen=True)
class ResolvedGroupDecision:
    decision: ReplyDecision
    tool_plan: ToolRoutePlan


async def resolve_group_reply_decision(
    *,
    bot: Bot,
    client: DeepSeekClient,
    pre_decision: PreDecisionGateResult,
    pipeline_state: PipelineState,
    reply_budget: GroupReplyBudget,
    tool_plan: ToolRoutePlan,
    persona: Any,
    context_recent: list[ChatMessage],
    text: str,
    nickname: str,
    speaker_context: str,
    discourse_state: DiscourseState,
    group_id: int,
    user_id: int,
    source_message_id: str,
    addressed_bot: bool,
    direct_addressed_bot: bool,
    synthetic_addressed_bot: bool,
    followup_addressed: bool,
    mentioned: bool,
    replied_to_bot: bool,
    market_intents: list[MarketIntent],
    fresh_intent: object | None,
    decision_started_at: float,
    flow_started_at: float,
    services: GroupDecisionServices,
) -> ResolvedGroupDecision | None:
    _record_metric_event = services.record_metric_event
    _record_tool_router_shadow = services.record_tool_router_shadow
    _send_approval_suppression_notice = services.send_suppression_notice
    _schedule_group_learning = services.schedule_group_learning
    _decision_failure_fallback = services.decision_failure_fallback
    _looks_like_addressed_question = services.looks_like_addressed_question
    _apply_tool_use_router = services.apply_tool_use_router
    _enforce_addressed_reply_decision = services.enforce_addressed_reply_decision
    _maybe_apply_speaking_action = services.maybe_apply_speaking_action
    _maybe_apply_ask_back = services.maybe_apply_ask_back
    logger = services.logger
    reference_resolution = discourse_state.reference
    ellipsis_resolution = discourse_state.ellipsis
    repair_resolution = discourse_state.repair
    ambiguity_resolution = discourse_state.ambiguity_resolution
    decision = pre_decision.decision
    decision_source = "local" if decision is not None else ""
    _pipeline_mark_gated(pipeline_state)
    _record_tool_router_shadow(
        group_id=group_id,
        user_id=user_id,
        decision=decision or ReplyDecision(False, 0.0, "legacy_pending", action="ignore"),
        tool_plan=tool_plan,
    )
    _record_metric_event(
        "tool_route_plan",
        group_id=group_id,
        user_id=user_id,
        stage="routing",
        action="deterministic",
        pipeline_mode=pipeline_state.mode.value,
        requests=[
            {
                "kind": request.kind.value,
                "required": request.required,
                "reason": request.reason,
                "query": _short_notice_text(request.query, 80),
            }
            for request in pipeline_state.tool_requests
        ],
    )
    if decision is None and any(request.required for request in tool_plan.requests):
        decision_source = "required_tool"
        decision = _apply_tool_plan(
            ReplyDecision(
                should_reply=True,
                confidence=1.0,
                reason="deterministic_required_tool",
                mode="tool",
                action="answer",
            ),
            tool_plan,
        )

    if decision is None and (direct_addressed_bot or mentioned or replied_to_bot):
        decision_source = "addressed"
        decision = ReplyDecision(
            should_reply=True,
            confidence=1.0,
            reason="addressed_skip_timing_gate",
            mode="addressed",
            action="answer" if _looks_like_addressed_question(text) else "reply",
        )
        logger.info(
            "qq_social_agent skipped timing_gate for addressed message: "
            f"group={group_id} user={user_id}"
        )
    if (
        decision is None
        and discourse_state.addressee.status == RESOLVED
        and discourse_state.addressee.target_id is not None
        and discourse_state.addressee.target_id != int(bot.self_id)
    ):
        decision_source = "discourse"
        decision = ReplyDecision(
            should_reply=False,
            confidence=discourse_state.addressee.confidence,
            reason="resolved_other_addressee",
            action="ignore",
        )
    if decision is None:
        decision_source = "jev"
        try:
            timing = await client.timing_gate(
                persona=persona,
                recent_messages=context_recent,
                current_text=text,
                current_nickname=_member_label(user_id, nickname),
                chat_label="QQ 群聊",
                speaker_context=speaker_context,
                discourse_state=discourse_state,
            )
            decision = timing.to_reply_decision()
        except Exception as exc:
            decision = _decision_failure_fallback(
                addressed_bot=addressed_bot,
                reason="timing_gate_error",
            )
            logger.warning(
                "qq_social_agent timing gate failed: "
                f"group={group_id} addressed={addressed_bot} error={exc}"
            )
            if decision is None:
                await _send_approval_suppression_notice(
                    bot,
                    group_id=group_id,
                    user_id=user_id,
                    nickname=nickname,
                    text=text,
                    stage="llm_decision_error",
                    reason=f"Timing Gate 调用失败，且非点名没有兜底回复：{exc}",
                )
                _schedule_group_learning(group_id)
                return
    else:
        logger.info(
            "qq_social_agent local pre-decision: "
            f"group={group_id} should_reply={decision.should_reply} "
            f"action={decision.action} mode={decision.mode} reason={decision.reason}"
        )
    if decision.reason == "invalid_json":
        fallback_decision = _decision_failure_fallback(
            addressed_bot=addressed_bot,
            reason="decision_invalid_json",
        )
        if fallback_decision is None:
            logger.warning(
                "qq_social_agent decision invalid json ignored: "
                f"group={group_id} addressed={addressed_bot}"
            )
            await _send_approval_suppression_notice(
                bot,
                group_id=group_id,
                user_id=user_id,
                nickname=nickname,
                text=text,
                stage="llm_invalid_json",
                reason="decision LLM 返回 invalid_json，且非点名没有兜底回复。",
            )
            _schedule_group_learning(group_id)
            return
        logger.warning(
            "qq_social_agent decision invalid json fallback: "
            f"group={group_id} addressed={addressed_bot}"
        )
        decision = fallback_decision
    decision = _apply_backend_tool_decision(
        decision,
        text=text,
        market_intents=market_intents,
        fresh_intent=fresh_intent,
    )
    decision = _apply_tool_plan(decision, tool_plan)
    decision, tool_plan = await _apply_tool_use_router(
        decision,
        tool_plan=tool_plan,
        persona=persona,
        context_recent=context_recent,
        text=text,
        nickname=nickname,
        addressed_bot=addressed_bot,
        fresh_intent=fresh_intent,
        market_intents=market_intents,
        speaker_context=speaker_context,
        group_id=group_id,
        user_id=user_id,
        source_message_id=source_message_id,
    )
    pipeline_state.mode = _tool_route_mode(tool_plan)
    pipeline_state.tool_requests = tool_plan.requests
    _record_metric_event(
        "tool_route_plan",
        group_id=group_id,
        user_id=user_id,
        stage="routing",
        action="final",
        pipeline_mode=pipeline_state.mode.value,
        requests=[
            {
                "kind": request.kind.value,
                "required": request.required,
                "reason": request.reason,
                "query": _short_notice_text(request.query, 80),
            }
            for request in pipeline_state.tool_requests
        ],
    )
    decision = _enforce_addressed_reply_decision(
        decision,
        addressed_bot=direct_addressed_bot
        or synthetic_addressed_bot
        or (followup_addressed and _looks_like_addressed_question(text)),
        text=text,
    )
    if reply_budget.skip("speaking_action", time.monotonic()):
        _record_metric_event(
            "reply_budget",
            group_id=group_id,
            user_id=user_id,
            stage="decision",
            action="skip_speaking_action",
            remaining_ms=int(reply_budget.remaining(time.monotonic()) * 1000),
        )
    else:
        decision = await _maybe_apply_speaking_action(
            decision,
            text=text,
            current_label=_member_label(user_id, nickname),
            addressed_bot=addressed_bot,
            speaker_context=speaker_context,
            recent_messages=context_recent,
            group_id=group_id,
            user_id=user_id,
            looks_like_question=_looks_like_addressed_question(text),
            unresolved_reference=reference_resolution.unresolved,
            unresolved_ellipsis=ellipsis_resolution.unresolved,
            unresolved_repair=repair_resolution.unresolved,
            unresolved_ambiguity=ambiguity_resolution.unresolved,
            ambiguity_kind=ambiguity_resolution.kind,
        )
    if reply_budget.skip("ask_back", time.monotonic()):
        _record_metric_event(
            "reply_budget",
            group_id=group_id,
            user_id=user_id,
            stage="decision",
            action="skip_ask_back",
            remaining_ms=int(reply_budget.remaining(time.monotonic()) * 1000),
        )
    else:
        decision = await _maybe_apply_ask_back(
            decision,
            text=text,
            addressed_bot=addressed_bot,
            group_id=group_id,
            user_id=user_id,
        )
    _pipeline_apply_decision(
        pipeline_state,
        should_reply=decision.should_reply,
        action=decision.action,
        reason=decision.reason,
        confidence=decision.confidence,
        side_reaction=decision.side_reaction,
        elapsed_ms=int((time.monotonic() - decision_started_at) * 1000),
    )
    _record_metric_event(
        "group_gate",
        group_id=group_id,
        user_id=user_id,
        stage="reply_decision",
        action="passed" if decision.should_reply else "blocked",
        reason=decision.reason,
        source=decision_source,
        channel=pipeline_state.output_channel.value,
    )
    if addressed_bot and "非点名" in decision.reason:
        logger.warning(
            "qq_social_agent decision state mismatch: "
            f"group={group_id} addressed=True reason={decision.reason}"
        )
    logger.info(
        "qq_social_agent llm decision: "
        f"group={group_id} should_reply={decision.should_reply} "
        f"confidence={decision.confidence:.2f} action={decision.action} mode={decision.mode} "
        f"side_reaction={decision.side_reaction} "
        f"need_fresh={decision.need_fresh_context} fresh_query={decision.fresh_query!r} "
        f"reason={decision.reason}"
    )
    _record_metric_event(
        "decision_result",
        group_id=group_id,
        user_id=user_id,
        stage="llm" if pre_decision.decision is None else "backend",
        action=decision.action,
        should_reply=decision.should_reply,
        fresh_query=_short_notice_text(decision.fresh_query, 120),
        fresh_kind=decision.fresh_kind,
        confidence=round(decision.confidence, 3),
        decision_reason=decision.reason,
        side_reaction=decision.side_reaction,
        need_fresh=decision.need_fresh_context,
        direct_addressed=direct_addressed_bot,
        synthetic_addressed=synthetic_addressed_bot,
        followup_addressed=followup_addressed,
        elapsed_ms=int((time.monotonic() - decision_started_at) * 1000),
        flow_elapsed_ms=int((time.monotonic() - flow_started_at) * 1000),
    )
    _schedule_group_learning(group_id)
    return ResolvedGroupDecision(decision=decision, tool_plan=tool_plan)
