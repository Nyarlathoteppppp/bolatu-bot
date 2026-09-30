"""Direct mentions must reach generation without a buffered-message list."""

import asyncio
import time
from types import SimpleNamespace

import nonebot
import pytest

nonebot.init()

from nonebot.adapters.onebot.v11 import Message, MessageSegment
from qq_social_agent import plugin
from qq_social_agent.deepseek_client import ReplyDecision
from qq_social_agent.discourse_state import DiscourseState
from qq_social_agent.memory import MemoryStore
from qq_social_agent.pipeline_stages import apply_decision
from qq_social_agent.pipeline_types import ContextPacket


@pytest.mark.parametrize("buffered_messages", [None, []])
def test_direct_mention_reaches_generation_and_excludes_current_message(
    monkeypatch, tmp_path, buffered_messages,
):
    store = MemoryStore(tmp_path / "bot.sqlite3")
    group_id, user_id, self_id = 1026813421, 1535071184, 1801507496
    now = time.time()
    text = "怎么办，列宁不会复活了"
    message = Message([MessageSegment.at(self_id), MessageSegment.text(text)])
    event = SimpleNamespace(
        group_id=group_id, user_id=user_id, self_id=self_id,
        message_id=542255427, time=now, message=message, raw_message=str(message),
        sender=SimpleNamespace(card="", nickname="奈亚子"),
        reply=None, get_plaintext=lambda: text,
    )
    store.add_message(group_id, user_id, "奈亚子", "列宁给我托梦了",
                      created_at=now - 10, source_message_id="previous")
    monkeypatch.setattr(plugin, "memory", store)
    monkeypatch.setattr(plugin, "app_config", SimpleNamespace(
        group_allowed=lambda _: True, group_config=lambda _: {"enabled": True},
        default_persona="default", context_limit=40,
    ))
    monkeypatch.setattr(plugin, "personas", SimpleNamespace(get=lambda _: SimpleNamespace(name="风雪")))
    monkeypatch.setattr(plugin, "deepseek_client", object())
    monkeypatch.setattr(plugin, "rate_limiter", SimpleNamespace(
        allow=lambda *_args, **_kwargs: SimpleNamespace(allowed=True),
    ))
    monkeypatch.setattr(plugin, "rag_service", SimpleNamespace(
        config=SimpleNamespace(exclude_recent_seconds=900),
    ))
    monkeypatch.setattr(plugin, "_schedule_group_learning", lambda _: None)
    monkeypatch.setattr(plugin, "_record_metric_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plugin, "_record_addressed_event", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(plugin, "_ai_work_intensity_percent", lambda: 100)
    monkeypatch.setattr(plugin, "_refresh_self_mute_state_if_stale",
                        lambda *_args: asyncio.sleep(0, result=0.0))
    monkeypatch.setattr(plugin, "_resolved_image_context",
                        lambda *_args, **_kwargs: asyncio.sleep(0, result=""))
    monkeypatch.setattr(plugin, "resolve_group_discourse_context", lambda **_: asyncio.sleep(0, result=SimpleNamespace(
        state=DiscourseState(), speaker_context="", memory_effect=None,
        memory_candidate=None, invalidated_layers=[], recomputed_layers=[],
    )))

    async def decide(**kwargs):
        assert kwargs["mentioned"] and kwargs["addressed_bot"]
        apply_decision(kwargs["pipeline_state"], should_reply=True, action="answer",
                       reason="local_addressed_reply", confidence=1.0)
        return SimpleNamespace(decision=ReplyDecision(True, 1.0, "local_addressed_reply", action="answer"),
                               tool_plan=kwargs["tool_plan"])

    monkeypatch.setattr(plugin, "resolve_group_reply_decision", decide)
    monkeypatch.setattr(plugin, "build_group_generation_context",
                        lambda **_: asyncio.sleep(0, result=ContextPacket()))
    monkeypatch.setattr(plugin, "execute_group_tools", lambda **_: asyncio.sleep(0, result=SimpleNamespace(
        market_context="", market_report=None, fresh_context="", direct_candidates=(),
    )))
    generated = []

    async def generate(**kwargs):
        generated.append(kwargs)
        return None

    monkeypatch.setattr(plugin, "generate_group_reply", generate)
    asyncio.run(plugin._handle_group_message_locked(
        SimpleNamespace(self_id=self_id), event,
        buffered_messages=buffered_messages, preprocessed_text=text,
    ))
    assert len(generated) == 1
    assert generated[0]["addressed_bot"]
    assert generated[0]["text"] == text
    assert [item.source_message_id for item in generated[0]["recent_messages"]] == ["previous"]
