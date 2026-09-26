"""Behavior at the extracted group decision, generation, and approval boundaries."""

import asyncio
import time
from types import SimpleNamespace

from qq_social_agent.approval_models import PendingApprovalCandidate
from qq_social_agent.decision_gate import PreDecisionGateResult
from qq_social_agent.deepseek_client import ReplyDecision
from qq_social_agent.discourse_effects import RepairResolution
from qq_social_agent.discourse_state import Binding, DiscourseState
from qq_social_agent.ellipsis_resolver import EllipsisResolution
from qq_social_agent.group_approval_dispatch import queue_group_reply_approval
from qq_social_agent.group_decision_flow import GroupDecisionServices, resolve_group_reply_decision
from qq_social_agent.group_discourse_flow import resolve_group_discourse_context
from qq_social_agent.group_generation_context import GroupContextLimits, build_group_generation_context
from qq_social_agent.group_reply_generation import generate_group_reply
from qq_social_agent.group_tool_execution import execute_group_tools
from qq_social_agent.jev_policy import GroupReplyBudget
from qq_social_agent.pipeline_types import ContextPacket, OutputChannel, PipelineMode, PipelineStage, PipelineState, ToolKind, ToolRequest, ToolResult
from qq_social_agent.rag_retriever import RAGRetrievalResult
from qq_social_agent.rag_router import RAGQueryPlan
from qq_social_agent.reference_resolver import ReferenceResolution
from qq_social_agent.reference_resolver import ReplyHint
from qq_social_agent.tool_router import ToolRoutePlan


def _pipeline() -> PipelineState:
    return PipelineState("trace", 123, 456, "群友", "看看这个", True, self_id=789)


def _logger():
    return SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)


def test_group_decision_stage_returns_routed_answer_and_updates_pipeline() -> None:
    metrics = []

    async def keep_decision(decision, **_kwargs):
        return decision

    async def keep_tool_plan(decision, *, tool_plan, **_kwargs):
        return decision, tool_plan

    async def unused_notice(*_args, **_kwargs):
        raise AssertionError("local decision should not need a suppression notice")

    services = GroupDecisionServices(
        record_metric_event=lambda event, **kwargs: metrics.append((event, kwargs)),
        record_tool_router_shadow=lambda **_: None,
        send_suppression_notice=unused_notice,
        schedule_group_learning=lambda _group_id: None,
        decision_failure_fallback=lambda **_: None,
        looks_like_addressed_question=lambda _text: True,
        apply_tool_use_router=keep_tool_plan,
        enforce_addressed_reply_decision=lambda decision, **_: decision,
        maybe_apply_speaking_action=keep_decision,
        maybe_apply_ask_back=keep_decision,
        logger=_logger(),
    )
    state = _pipeline()
    started = time.monotonic()
    result = asyncio.run(
        resolve_group_reply_decision(
            bot=object(),
            client=object(),
            pre_decision=PreDecisionGateResult(ReplyDecision(True, 1.0, "local", action="answer")),
            pipeline_state=state,
            reply_budget=GroupReplyBudget.start(started, seconds=120),
            tool_plan=ToolRoutePlan(),
            persona=object(),
            context_recent=[],
            text="看看这个",
            nickname="群友",
            speaker_context="当前触发人是群友",
            discourse_state=DiscourseState(),
            group_id=123,
            user_id=456,
            source_message_id="m1",
            addressed_bot=True,
            direct_addressed_bot=True,
            synthetic_addressed_bot=False,
            followup_addressed=False,
            mentioned=True,
            replied_to_bot=False,
            market_intents=[],
            fresh_intent=None,
            decision_started_at=started,
            flow_started_at=started,
            services=services,
        )
    )

    assert result is not None and result.decision.action == "answer"
    assert state.output_channel is OutputChannel.TEXT
    assert state.stage is PipelineStage.DECIDED
    assert any(event == "decision_result" for event, _ in metrics)


def test_group_decision_skips_timing_when_addressee_is_another_member() -> None:
    async def keep_decision(decision, **_kwargs):
        return decision

    async def keep_tool_plan(decision, *, tool_plan, **_kwargs):
        return decision, tool_plan

    async def unexpected_timing(**_kwargs):
        raise AssertionError("timing gate should not decide a message addressed to another member")

    services = GroupDecisionServices(
        record_metric_event=lambda *_args, **_kwargs: None,
        record_tool_router_shadow=lambda **_kwargs: None,
        send_suppression_notice=lambda *_args, **_kwargs: None,
        schedule_group_learning=lambda _group_id: None,
        decision_failure_fallback=lambda **_kwargs: None,
        looks_like_addressed_question=lambda _text: True,
        apply_tool_use_router=keep_tool_plan,
        enforce_addressed_reply_decision=lambda decision, **_kwargs: decision,
        maybe_apply_speaking_action=keep_decision,
        maybe_apply_ask_back=keep_decision,
        logger=_logger(),
    )
    started = time.monotonic()
    result = asyncio.run(resolve_group_reply_decision(
        bot=SimpleNamespace(self_id=789),
        client=SimpleNamespace(timing_gate=unexpected_timing),
        pre_decision=PreDecisionGateResult(None),
        pipeline_state=_pipeline(),
        reply_budget=GroupReplyBudget.start(started, seconds=120),
        tool_plan=ToolRoutePlan(),
        persona=object(),
        context_recent=[],
        text="你怎么想？",
        nickname="群友",
        speaker_context="",
        discourse_state=DiscourseState(addressee=Binding(status="RESOLVED", target="另一位群友", target_id=456)),
        group_id=123,
        user_id=111,
        source_message_id="m2",
        addressed_bot=False,
        direct_addressed_bot=False,
        synthetic_addressed_bot=False,
        followup_addressed=False,
        mentioned=False,
        replied_to_bot=False,
        market_intents=[],
        fresh_intent=None,
        decision_started_at=started,
        flow_started_at=started,
        services=services,
    ))
    assert result is not None
    assert result.decision.should_reply is False
    assert result.decision.reason == "resolved_other_addressee"


def test_group_generation_returns_reviewed_candidate() -> None:
    candidate = PendingApprovalCandidate(1, "你好", "answer", "自然")
    calls = []

    async def reply_candidates(**kwargs):
        calls.append(kwargs)
        return ["你好"]

    async def review_draft(**_kwargs):
        return None, None

    result = asyncio.run(
        generate_group_reply(
            client=SimpleNamespace(reply_candidates=reply_candidates, review_draft=review_draft),
            decision=ReplyDecision(True, 1.0, "local", action="answer"),
            persona=object(),
            recent_messages=[],
            text="你好",
            nickname="群友",
            current_label="群友[#00456]",
            addressed_bot=True,
            addressed_repeat_count=1,
            cue_repeat_context="",
            market_context="",
            fresh_context="",
            context_packet=ContextPacket(),
            mode=PipelineMode.CHAT,
            mention_targets_context="",
            priority_context="",
            speaker_context="当前触发人是群友",
            memory_context="",
            reference_resolution=ReferenceResolution(),
            ellipsis_resolution=EllipsisResolution(),
            repair_resolution=RepairResolution(),
            discourse_state=DiscourseState(),
            direct_single_reply=False,
            market_report="",
            reply_budget=GroupReplyBudget.start(time.monotonic(), seconds=120),
            group_id=123,
            user_id=456,
            build_approval_candidates=lambda _drafts, **_: [candidate],
            combine_text_sections=lambda *parts: "\n".join(part for part in parts if part),
            record_metric_event=lambda *_args, **_kwargs: None,
            logger=_logger(),
        )
    )

    assert result is not None and result.candidates == (candidate,)
    assert result.prompt_flow == "reply_candidates"
    assert calls[0]["speaker_context"] == "当前触发人是群友"
    assert calls[0]["include_bot_history"] is True


def test_market_report_skips_generation_and_enters_approval() -> None:
    state = _pipeline()
    state.transition(PipelineStage.DECIDED)
    request = ToolRequest(ToolKind.MARKET, query="TSLA", required=True)

    async def execute(_request):
        return ToolResult(ToolKind.MARKET, "ok", evidence="TSLA 报告")

    async def unused_fresh(*_args, **_kwargs):
        raise AssertionError("market report should bypass fresh search")

    result = asyncio.run(
        execute_group_tools(
            decision=ReplyDecision(True, 1.0, "market", action="market_check", need_tool=True, tool="market"),
            tool_plan=ToolRoutePlan((request,)),
            pipeline_state=state,
            group_id=123,
            user_id=456,
            text="TSLA",
            market_intents=[],
            market_context_task=None,
            prefetched_market_request=None,
            tool_registry=SimpleNamespace(execute=execute),
            market_intents_from_decision=lambda *_args, **_kwargs: [],
            execute_fresh_tool_request=unused_fresh,
            fresh_tool_failure_context=lambda *_args, **_kwargs: "",
            combine_text_sections=lambda *parts: "\n".join(parts),
            record_metric_event=lambda *_args, **_kwargs: None,
            logger=_logger(),
        )
    )
    approvals = []

    async def request_approval(_bot, approval):
        approvals.append(approval)

    asyncio.run(
        queue_group_reply_approval(
            object(),
            group_id=123,
            user_id=456,
            nickname="群友",
            text="TSLA",
            persona_name="风雪",
            self_id=789,
            candidates=result.direct_candidates,
            mention_targets={},
            trigger_sequence=1,
            pipeline_state=state,
            source_message_id="m1",
            correlation_id="trace",
            new_approval_id=lambda _group_id: "approval-1",
            request_approval=request_approval,
            apply_candidates_first=True,
        )
    )

    assert result.direct_candidates[0].text == "TSLA 报告"
    assert state.stage is PipelineStage.APPROVAL_PENDING
    assert approvals[0].candidates == result.direct_candidates


def test_group_context_builder_returns_memory_packet_without_rag_lookup() -> None:
    calls = []

    class Memory:
        def __getattr__(self, name):
            if name.startswith("relevant_") or name.startswith("recent_") or name == "member_impressions_for_context":
                return lambda *_args, **_kwargs: ["summary"] if name == "relevant_memory_summaries" else []
            raise AttributeError(name)

    async def unused_retrieve(**_kwargs):
        raise AssertionError("the optional RAG budget is exhausted")

    async def select_jargon(*_args, **_kwargs):
        return ""

    result = asyncio.run(
        build_group_generation_context(
            group_id=123,
            user_id=456,
            nickname="群友",
            text="你好",
            self_id=789,
            context_query="你好",
            recent_messages=[],
            reference_resolution=ReferenceResolution(),
            addressed_bot=True,
            mode=PipelineMode.CHAT,
            reply_budget=GroupReplyBudget(0, 0),
            memory=Memory(),
            rag_service=SimpleNamespace(retrieve=unused_retrieve),
            social_action_service=SimpleNamespace(recent_reaction_context=lambda _group_id: ""),
            limits=GroupContextLimits(4, 900, 8, 6, 4, 2, 240, 2, 3, 4),
            select_jargon_context=select_jargon,
            format_memory_context=lambda _summaries: "旧回想",
            format_memory_atom_context=lambda _atoms: "",
            format_style_context=lambda _rules: "",
            format_raw_corpus_context=lambda _examples: "",
            format_recall_feedback_context=lambda _items: "",
            format_positive_feedback_context=lambda _items: "",
            record_metric_event=lambda event, **kwargs: calls.append((event, kwargs)),
        )
    )

    assert result.get("memory") == "旧回想"
    assert any(event == "context_assembled" for event, _ in calls)


def test_group_context_builder_merges_rag_with_memory_summary() -> None:
    metrics = []
    memory = SimpleNamespace(
        relevant_memory_summaries=lambda *_args, **_kwargs: ["summary"],
        member_impressions_for_context=lambda *_args, **_kwargs: [],
        relevant_memory_atoms=lambda *_args, **_kwargs: [],
        relevant_style_rules=lambda *_args, **_kwargs: [],
        relevant_raw_corpus_examples=lambda *_args, **_kwargs: [],
        recent_recalled_reply_feedback=lambda *_args, **_kwargs: [],
        recent_approved_reply_feedback=lambda *_args, **_kwargs: [],
    )

    async def retrieve(**_kwargs):
        return RAGRetrievalResult(RAGQueryPlan(True, True, False, "test"), (), "RAG 证据", 1, 0, 0)

    async def select_jargon(*_args, **_kwargs):
        return ""

    result = asyncio.run(
        build_group_generation_context(
            group_id=123,
            user_id=456,
            nickname="群友",
            text="那件事怎么样",
            self_id=789,
            context_query="那件事怎么样",
            recent_messages=[],
            reference_resolution=ReferenceResolution(),
            addressed_bot=True,
            mode=PipelineMode.CHAT,
            reply_budget=GroupReplyBudget.start(time.monotonic(), seconds=120),
            memory=memory,
            rag_service=SimpleNamespace(retrieve=retrieve),
            social_action_service=SimpleNamespace(recent_reaction_context=lambda _group_id: ""),
            limits=GroupContextLimits(4, 900, 8, 6, 4, 2, 240, 2, 3, 4),
            select_jargon_context=select_jargon,
            format_memory_context=lambda _summaries: "旧回想",
            format_memory_atom_context=lambda _atoms: "",
            format_style_context=lambda _rules: "",
            format_raw_corpus_context=lambda _examples: "",
            format_recall_feedback_context=lambda _items: "",
            format_positive_feedback_context=lambda _items: "",
            record_metric_event=lambda event, **kwargs: metrics.append((event, kwargs)),
        )
    )

    assert "RAG 证据" in result.get("memory")
    assert "旧回想" in result.get("memory")
    assert any(event == "rag_retrieval" and details["action"] == "merged" for event, details in metrics)


def test_discourse_stage_returns_single_state_and_speaker_context(monkeypatch) -> None:
    import qq_social_agent.group_discourse_flow as flow

    async def resolve(**_kwargs):
        return DiscourseState()

    monkeypatch.setattr(flow, "resolve_group_discourse", resolve)
    metrics = []
    result = asyncio.run(
        resolve_group_discourse_context(
            group_id=123,
            user_id=456,
            nickname="群友",
            text="你好",
            normalized_text="你好",
            self_id=789,
            recent_messages=[],
            reply_hint=ReplyHint(),
            at_user_ids=(),
            current_has_media=False,
            reply_has_media=False,
            mentioned=True,
            replied_to_bot=False,
            addressed_bot=True,
            followup_addressed=False,
            followup_soft=False,
            source_message_id="m1",
            client=object(),
            memory=object(),
            rag_service=SimpleNamespace(resolve_named_user_ids=lambda *_: ()),
            record_metric_event=lambda event, **kwargs: metrics.append((event, kwargs)),
            logger=_logger(),
        )
    )

    assert result.state.reference == ReferenceResolution()
    assert "当前触发人：群友[#456]" in result.speaker_context
    assert [event for event, _ in metrics] == ["discourse_decision_trace", "message_relation"]
