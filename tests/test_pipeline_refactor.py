from __future__ import annotations

import asyncio

from qq_social_agent.background_learning import BackgroundLearningCoordinator
from qq_social_agent.context_assembler import assemble_generation_context
from qq_social_agent.memory import MemoryStore
from qq_social_agent.pipeline_types import OutputChannel, SocialIntent, ToolKind
from qq_social_agent.timing_gate import parse_timing_decision
from qq_social_agent.tool_router import apply_tool_plan, compare_legacy_decision, route_tools
from qq_social_agent.deepseek_client import ReplyDecision, _parse_mid_memory
from qq_social_agent.tools.fresh_context import FreshIntent
from qq_social_agent.tools.market_intent import MarketIntent


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
