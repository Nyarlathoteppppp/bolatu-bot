from __future__ import annotations

import asyncio

from qq_social_agent.background_learning import BackgroundLearningCoordinator
from qq_social_agent.context_assembler import assemble_generation_context
from qq_social_agent.memory import ChatMessage, MemoryStore
from qq_social_agent.pipeline_types import (
    ContextPacket,
    OutputChannel,
    PipelineMode,
    PipelineStage,
    PipelineState,
    SocialIntent,
    ToolKind,
)
from qq_social_agent.pipeline_stages import (
    apply_candidates,
    apply_context,
    apply_decision,
    mark_approval_pending,
    mark_completed,
    mark_gated,
    mark_sending,
    mark_sent,
)
from qq_social_agent.timing_gate import parse_timing_decision
from qq_social_agent.tool_router import (
    apply_tool_plan,
    compare_legacy_decision,
    infer_followup_fresh_intent,
    route_mode,
    route_tools,
)
from qq_social_agent.deepseek_client import DeepSeekClient, ReplyDecision, _parse_mid_memory
from qq_social_agent.tools.fresh_context import FreshIntent
from qq_social_agent.tools.market_intent import MarketIntent
from types import SimpleNamespace


def test_context_assembler_keeps_memory_and_structured_context() -> None:
    packet = assemble_generation_context(
        memory_context="群聊历史证据",
        member_context="旧人物选择器",
        memory_atoms_context="长期事实",
        style_context="短句接话",
        rag_document_ids=(7, 8),
        rag_document_types=("conversation", "member"),
    )

    assert packet.get("memory") == "群聊历史证据"
    assert packet.get("member") == "旧人物选择器"
    assert packet.get("memory_atoms") == "长期事实"
    assert packet.get("style") == "短句接话"


def test_search_context_drops_old_chat_evidence() -> None:
    packet = assemble_generation_context(
        memory_context="旧聊天里有人说 ASIC 只用于挖矿",
        member_context="人物画像",
        memory_atoms_context="长期记忆",
        style_context="群聊风格",
        raw_corpus_context="群友原话",
        jargon_context="ASIC：专用集成电路",
        mode=PipelineMode.SEARCH,
    )

    assert packet.mode is PipelineMode.SEARCH
    assert packet.get("jargon") == "ASIC：专用集成电路"
    assert packet.get("memory") == ""
    assert packet.get("raw_corpus") == ""
    assert "memory" in packet.dropped_sections


def test_tool_router_routes_required_tools_without_social_action_choice() -> None:
    plan = route_tools(
        "帮我查一下英伟达现在股价",
        market_intents=[MarketIntent(kind="stock", symbol="NVDA", display_name="英伟达")],
        fresh_intent=FreshIntent("英伟达现在股价", "web", explicit=True, required=True),
        addressed=True,
        market_required=True,
    )
    decision = ReplyDecision(True, 0.8, "回答问题", action="answer")
    routed = apply_tool_plan(decision, plan)
    comparison = compare_legacy_decision(routed, plan)

    assert set(plan.kinds) == {ToolKind.MARKET.value, ToolKind.FRESH_SEARCH.value}
    assert routed.need_tool and routed.tool == "market"
    assert routed.need_fresh_context
    assert routed.symbols[0].symbol == "NVDA"
    assert comparison.matched


def test_timing_gate_has_small_channel_and_intent_surface() -> None:
    timing = parse_timing_decision(
        {
            "channel": "react",
            "intent": "play",
            "confidence": 0.81,
            "reason": "笑一下就够了",
            "reaction": "laugh",
        }
    )
    decision = timing.to_reply_decision()

    assert timing.channel is OutputChannel.REACT
    assert timing.intent is SocialIntent.PLAY
    assert decision.action == "react"
    assert decision.reaction == "laugh"


def test_timing_gate_converts_text_reaction_to_side_reaction() -> None:
    timing = parse_timing_decision(
        {
            "channel": "text",
            "intent": "chat",
            "confidence": 0.7,
            "reason": "能接一句",
            "reaction": "laugh",
        }
    )
    decision = timing.to_reply_decision()

    assert timing.channel is OutputChannel.TEXT
    assert timing.reaction == ""
    assert timing.side_reaction == "laugh"
    assert decision.action == "reply"
    assert decision.side_reaction == "laugh"


def test_tool_router_searches_academic_concept_without_asking_timing_model() -> None:
    plan = route_tools(
        "三维挂谷猜想是什么，有什么最新进展",
        market_intents=[],
        fresh_intent=None,
        addressed=True,
    )

    request = plan.first(ToolKind.FRESH_SEARCH)
    assert request is not None
    assert request.required
    assert request.arguments["kind"] == "web"


def test_followup_research_inherits_latest_real_user_topic() -> None:
    messages = [
        ChatMessage(1, 7, "群友", "你知道 ASIC 芯片吗，相关股票有前景吗", False, 10.0),
        ChatMessage(1, 99, "张风雪", "这个得具体查，不能瞎推荐", True, 11.0),
    ]

    intent = infer_followup_fresh_intent("你帮我研究研究", messages, addressed=True)

    assert intent is not None
    assert "ASIC" in intent.query
    assert intent.required


def test_bare_search_command_inherits_same_users_previous_message() -> None:
    messages = [
        ChatMessage(1, 7, "甲", "三维挂谷猜想和希尔伯特第六问题最近分别有什么进展", False, 100.0),
        ChatMessage(1, 8, "乙", "我觉得都很难", False, 105.0),
    ]

    intent = infer_followup_fresh_intent(
        "搜一下",
        messages,
        addressed=False,
        current_user_id=7,
        current_at=110.0,
    )

    assert intent is not None
    assert intent.query.startswith("三维挂谷猜想")
    assert intent.explicit and intent.required


def test_bare_search_command_does_not_steal_old_or_other_users_topic() -> None:
    messages = [ChatMessage(1, 8, "乙", "帮忙看看这个很长的话题", False, 100.0)]

    assert infer_followup_fresh_intent(
        "搜一下",
        messages,
        addressed=False,
        current_user_id=7,
        current_at=110.0,
    ) is None
    assert infer_followup_fresh_intent(
        "搜一下",
        [ChatMessage(1, 7, "甲", "两分钟前的旧问题是什么", False, 100.0)],
        addressed=False,
        current_user_id=7,
        current_at=300.0,
    ) is None


def test_tool_route_mode_prefers_market_when_search_is_also_required() -> None:
    plan = route_tools(
        "查 NVDA 现在股价",
        market_intents=[MarketIntent(kind="stock", symbol="NVDA", display_name="英伟达")],
        fresh_intent=FreshIntent("NVDA 现在股价", "web", explicit=True, required=True),
        addressed=True,
        market_required=True,
    )

    assert route_mode(plan) is PipelineMode.MARKET


def test_pipeline_state_crosses_decision_context_approval_and_delivery() -> None:
    state = PipelineState("cid", 1, 2, "甲", "你怎么看", True, source_message_id="88")
    candidate = SimpleNamespace(index=1, text="风雪觉得可以", action="answer", style="直接回答")

    mark_gated(state)
    apply_decision(
        state,
        should_reply=True,
        action="answer",
        reason="addressed",
        confidence=1.0,
        elapsed_ms=4,
    )
    apply_context(state, ContextPacket(mode=PipelineMode.CHAT))
    apply_candidates(state, [candidate], elapsed_ms=20)
    mark_approval_pending(state, "approval-1")
    mark_sending(state)
    mark_sent(state, 99)
    mark_completed(state, elapsed_ms=8)

    assert state.stage is PipelineStage.COMPLETED
    assert state.approval_id == "approval-1"
    assert state.sent_message_ids == ("99",)
    assert state.candidates[0].text == "风雪觉得可以"
    assert state.stage_history == [
        "received",
        "gated",
        "decided",
        "context_ready",
        "generated",
        "approval_pending",
        "sending",
        "completed",
    ]


def test_mid_summary_page_can_exclude_bot_messages(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(8):
        memory.add_message(
            1,
            9000 if index % 2 else 1000 + index,
            "张风雪" if index % 2 else "群友",
            f"消息{index}",
            is_bot=bool(index % 2),
            created_at=1000 + index,
        )
    rows = memory.messages_for_mid_summary(
        1,
        keep_recent=1,
        batch_size=20,
        include_bot=False,
    )

    assert rows
    assert all(not row.is_bot for row in rows)
    assert [row.text for row in rows] == ["消息0", "消息2", "消息4", "消息6"]
    memory.conn.close()


def test_mid_memory_recovers_summary_from_truncated_structured_json() -> None:
    draft = _parse_mid_memory(
        '{"summary":"总览：群友讨论了算法和留学；按人：A[#10001]想继续准备",'
        '"recall_cues":["算法"],"facts":[{"kind":"event"'
    )

    assert draft.summary.startswith("总览：群友讨论了算法和留学")
    assert draft.facts == ()


def test_mid_memory_has_background_timeout_budget() -> None:
    client = object.__new__(DeepSeekClient)
    client.config = SimpleNamespace(
        decision_timeout_seconds=10.0,
        decision_total_timeout_seconds=18.0,
        reply_timeout_seconds=18.0,
        reply_total_timeout_seconds=28.0,
        daily_review_timeout_seconds=35.0,
        daily_review_total_timeout_seconds=75.0,
        utility_timeout_seconds=8.0,
        utility_total_timeout_seconds=12.0,
        timeout_seconds=30.0,
    )

    assert client._task_timeouts(task="mid_memory", route_name="memory") == (18.0, 40.0)


def test_background_learning_uses_one_worker_and_defers_busy_group() -> None:
    calls: list[int] = []
    busy = {1}

    async def maintain(group_id: int) -> None:
        calls.append(group_id)

    async def scenario() -> None:
        coordinator = BackgroundLearningCoordinator(
            maintain,
            target_groups=lambda: (),
            is_busy=lambda group_id: group_id in busy,
            sweep_seconds=30,
            busy_retry_seconds=0.01,
        )
        coordinator.start()
        coordinator.notify(1)
        await asyncio.sleep(0.03)
        assert calls == []
        busy.clear()
        await asyncio.sleep(0.04)
        await coordinator.close()

    asyncio.run(scenario())
    assert calls == [1]
