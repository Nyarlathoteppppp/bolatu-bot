"""Tool routing remains independent from the NoneBot plugin runtime."""

import asyncio
from types import SimpleNamespace

from qq_social_agent.deepseek_client import ReplyDecision, ToolRoutingDecision
from qq_social_agent.conversation_tool_routing import _apply_tool_use_router
from qq_social_agent.pipeline_types import ToolKind
from qq_social_agent.tool_router import ToolRoutePlan


def test_llm_url_route_preserves_runtime_context_and_metric() -> None:
    calls = []
    metrics = []

    async def route_tool_use(**kwargs):
        calls.append(kwargs)
        return ToolRoutingDecision(
            tool="deep_url",
            query="https://example.com/repo",
            confidence=0.9,
            reason="user_requested_repo",
        )

    decision, plan = asyncio.run(
        _apply_tool_use_router(
            ReplyDecision(True, 0.9, "test", action="answer"),
            tool_plan=ToolRoutePlan(),
            persona=object(),
            context_recent=[],
            text="看看这个仓库 https://example.com/repo",
            nickname="测试群友",
            addressed_bot=True,
            fresh_intent=None,
            market_intents=[],
            speaker_context="当前触发人是测试群友",
            group_id=123,
            user_id=456,
            source_message_id="message-1",
            client=SimpleNamespace(route_tool_use=route_tool_use),
            record_metric_event=lambda *args, **kwargs: metrics.append((args, kwargs)),
            logger=SimpleNamespace(warning=lambda *_: None),
        )
    )

    request = plan.first(ToolKind.DEEP_URL)
    assert decision.should_reply
    assert request is not None
    assert request.arguments == {
        "addressed": True,
        "group_id": 123,
        "user_id": 456,
        "source_message_id": "message-1",
    }
    assert calls[0]["speaker_context"] == "当前触发人是测试群友"
    assert metrics[0][0] == ("tool_router",)
    assert metrics[0][1]["action"] == "deep_url"
