"""Private buffering and follow-up scheduling retain their live semantics."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import nonebot

nonebot.init()

from nonebot.adapters.onebot.v11 import Message

from qq_social_agent import plugin
from qq_social_agent.private_session_service import PrivateFollowupServices, PrivateSessionService
from qq_social_agent.private_message_types import BufferedPrivateMessage


def _private_event(message_id: str, user_id: int = 99887766):
    return SimpleNamespace(
        message_id=message_id,
        user_id=user_id,
        self_id=1801507496,
        time=1_800_000_000,
        message=Message(message_id),
        sender=SimpleNamespace(nickname="A"),
    )


def _clear_plugin_session_state() -> None:
    service = plugin.private_session_service
    service.processing_locks.clear()
    service.message_buffers.clear()
    service.buffer_tasks.clear()
    service.generation_inflight.clear()
    service.inbound_message_counts.clear()
    service.followup_tasks.clear()


def test_continuous_private_messages_flush_in_original_order(monkeypatch):
    _clear_plugin_session_state()
    service = plugin.private_session_service
    monkeypatch.setattr(service, "schedule_buffer_flush", lambda *args, **kwargs: None)
    handled = []
    followups = []

    async def handle(bot, event, **kwargs):
        assert event.message_id == "private-2"
        handled.append([item.text for item in kwargs["buffered_messages"]])
        assert 99887766 in service.generation_inflight

    monkeypatch.setattr(plugin, "_handle_private_message_scoped", handle)
    monkeypatch.setattr(
        plugin,
        "_schedule_private_followup_if_due",
        lambda user_id, *, added_messages: followups.append((user_id, added_messages)),
    )
    bot = SimpleNamespace(self_id=1801507496)
    plugin._buffer_private_message(bot, _private_event("private-1"), text="第一条", correlation_id="c1")
    plugin._buffer_private_message(bot, _private_event("private-2"), text="第二条", correlation_id="c2")

    asyncio.run(plugin._flush_private_buffer_after_delay(99887766, delay=0))

    assert handled == [["第一条", "第二条"]]
    assert followups == [(99887766, 2)]
    assert service.message_buffers.get(99887766, []) == []
    _clear_plugin_session_state()


def test_new_private_messages_wait_while_generation_is_inflight(monkeypatch):
    _clear_plugin_session_state()
    service = plugin.private_session_service
    user_id = 99887766
    item = BufferedPrivateMessage(
        bot=SimpleNamespace(self_id=1801507496),
        event=_private_event("deferred-1"),
        text="生成期间到达",
        user_id=user_id,
        nickname="A",
        created_at=1_800_000_000,
        source_message_id="deferred-1",
        correlation_id="deferred-correlation",
    )
    service.message_buffers[user_id] = [item]
    service.generation_inflight.add(user_id)
    rescheduled = []
    monkeypatch.setattr(
        service,
        "schedule_buffer_flush",
        lambda uid, **kwargs: rescheduled.append((uid, kwargs["delay"])),
    )
    handled = []
    monkeypatch.setattr(plugin, "_handle_private_message_scoped", lambda *a, **kw: handled.append(kw))

    asyncio.run(plugin._flush_private_buffer_after_delay(user_id, delay=0))

    assert handled == []
    assert service.message_buffers[user_id] == [item]
    assert rescheduled == [(user_id, plugin.PRIVATE_INFLIGHT_BUFFER_RETRY_SECONDS)]
    service.generation_inflight.clear()
    _clear_plugin_session_state()


def test_followup_counter_delay_and_cancellation_are_preserved():
    service = PrivateSessionService()
    started = []
    cancelled = []

    async def followup(user_id, *, expected_message_count, delay):
        started.append((user_id, expected_message_count, delay))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(expected_message_count)
            raise

    async def run():
        service.schedule_followup_if_due(1, added_messages=1, delay=10.0, run_followup=followup)
        assert 1 not in service.followup_tasks
        service.schedule_followup_if_due(1, added_messages=1, delay=10.0, run_followup=followup)
        first = service.followup_tasks[1]
        await asyncio.sleep(0)
        service.schedule_followup_if_due(1, added_messages=2, delay=10.0, run_followup=followup)
        second = service.followup_tasks[1]
        await asyncio.sleep(0)
        second.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(first, second)

    asyncio.run(run())
    assert service.inbound_message_counts[1] == 4
    assert started == [(1, 2, 10.0), (1, 4, 10.0)]
    assert cancelled == [2, 4]
    assert plugin.private_session_service.followup_probability(
        1903297906,
        probability_by_user=plugin.PRIVATE_FOLLOWUP_PROBABILITY_BY_USER,
        default=plugin.PRIVATE_FOLLOWUP_PROBABILITY,
    ) == 0.08


def test_followup_probability_gate_records_skip_without_generating(monkeypatch):
    service = PrivateSessionService()
    user_id = 1
    service.inbound_message_counts[user_id] = 2
    metrics = []
    generated = []
    monkeypatch.setattr("qq_social_agent.private_session_service.random.random", lambda: 0.9)
    dependencies = PrivateFollowupServices(
        memory=SimpleNamespace(),
        get_deepseek_client=lambda: generated.append("client") or None,
        private_user_can_chat=lambda user: True,
        private_chat_id=lambda user: 10_000 + user,
        rate_limiter=SimpleNamespace(),
        personas=SimpleNamespace(),
        default_persona="default",
        format_memory_context=lambda values: "",
        private_priority_context=lambda user: "",
        member_label=lambda user, nickname: nickname,
        first_connected_onebot_bot=lambda: None,
        send_private_message=lambda *args, **kwargs: None,
        record_metric_event=lambda *args, **kwargs: metrics.append((args, kwargs)),
        short_notice_text=lambda text, limit: text[:limit],
        sanitize_generated_text=lambda text: text,
        blocked_backend_fallback_texts=frozenset(),
        probability_by_user={user_id: 0.2},
        default_probability=0.5,
        private_context_limit=40,
        mid_memory_keep_summaries=4,
        logger=plugin.logger,
    )

    asyncio.run(service.run_followup_after_delay(
        user_id,
        expected_message_count=2,
        delay=0,
        services=dependencies,
    ))

    assert metrics[0][0] == ("private_followup",)
    assert metrics[0][1]["stage"] == "probability"
    assert metrics[0][1]["action"] == "skipped"
    assert metrics[0][1]["probability"] == 0.2
    assert generated == []
