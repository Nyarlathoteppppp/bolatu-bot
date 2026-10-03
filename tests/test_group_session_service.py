from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qq_social_agent.group_message_types import BufferedGroupMessage
from qq_social_agent.group_session_service import GroupSessionService, GroupSessionState


LOGGER = SimpleNamespace(info=lambda *_args: None)


def item(user_id=11, text='问题', *, addressed=False, source='original'):
    return BufferedGroupMessage(
        bot=SimpleNamespace(self_id=18), event=SimpleNamespace(group_id=1, user_id=user_id),
        user_id=user_id, nickname=str(user_id), text=text, created_at=100,
        addressed=addressed, source_message_id=source, correlation_id='turn-1',
    )


def flush(service, handle, *, schedule=lambda *_args, **_kwargs: None):
    return service.flush_after_delay(
        1, delay=0, retry_delay=1, handle_group_message=handle,
        schedule_buffer_flush=schedule, record_metric_event=lambda *_args, **_kwargs: None,
        logger=LOGGER,
    )


def test_state_is_shared_with_external_generators_and_views():
    state = GroupSessionState()
    first, second = GroupSessionService(state), GroupSessionService(state)
    assert first.processing_lock(1) is second.processing_lock(1)
    assert first.processing_lock(2) is not first.processing_lock(1)
    assert first.next_inbound_sequence(1) == 1
    assert second.next_inbound_sequence(1) == 2
    assert first.next_inbound_sequence(2) == 1
    state.generation_inflight.add(1)  # e.g. proactive service using the shared set
    assert second.should_defer_reply(1)
    state.generation_inflight.clear()
    first.begin_addressed(1)
    second.begin_addressed(1)
    first.end_addressed(1)
    assert state.addressed_waiters == {1: 1}
    assert second.should_defer_reply(1)
    second.end_addressed(1)
    assert state.addressed_waiters == {}
    assert not first.should_defer_reply(1)


def test_live_scheduler_deduplicates_then_flushes_new_messages_after_generation():
    async def run():
        state = GroupSessionState()
        service = GroupSessionService(state)
        started, release = asyncio.Event(), asyncio.Event()
        turns = []
        async def handle(bot, event, *, buffered_messages):
            turns.append([entry.source_message_id for entry in buffered_messages])
            assert 1 in state.generation_inflight
            if len(turns) == 1:
                started.set()
                await release.wait()
        def schedule(group_id, *, delay=0):
            service.schedule_buffer_flush(group_id, delay=delay, flush=dispatch)
        async def dispatch(group_id, *, delay):
            await service.flush_after_delay(
                group_id, delay=delay, retry_delay=0, handle_group_message=handle,
                schedule_buffer_flush=schedule, record_metric_event=lambda *_args, **_kwargs: None,
                logger=LOGGER,
            )
        service.buffer_message(1, item(source='a'), schedule_buffer_flush=schedule, logger=LOGGER)
        first_task = state.buffer_tasks[1]
        await started.wait()
        service.buffer_message(1, item(22, source='b'), schedule_buffer_flush=schedule, logger=LOGGER)
        assert state.buffer_tasks[1] is first_task
        release.set()
        await first_task
        second_task = state.buffer_tasks[1]
        assert second_task is not first_task
        await second_task
        assert turns == [['a'], ['b']]
        assert not state.buffer_tasks and not state.generation_inflight and not state.message_buffers
    asyncio.run(run())


def test_addressed_fifo_preserves_same_user_continuations_and_original_trigger():
    state = GroupSessionState(message_buffers={1: [
        item(text='先说主题', source='topic'), item(addressed=True, source='a1'),
        item(text='补充', source='extra'), item(22, addressed=True, source='b1'),
        item(11, addressed=True, source='a2'), item(33, source='chat'),
    ]})
    service = GroupSessionService(state)
    turns = []
    async def handle(bot, event, *, buffered_messages):
        turns.append((event.user_id, [entry.source_message_id for entry in buffered_messages]))
    async def run():
        for _ in range(4):
            await flush(service, handle)
    asyncio.run(run())
    assert turns == [(11, ['topic', 'a1', 'extra']), (22, ['b1']), (11, ['a2']), (33, ['chat'])]
    assert not service.should_defer_reply(1)


@pytest.mark.parametrize('busy', ['addressed', 'generation'])
def test_buffer_waits_for_addressed_or_external_generation_without_losing_messages(busy):
    original = item(source='waiting')
    state = GroupSessionState(message_buffers={1: [original]})
    service = GroupSessionService(state)
    if busy == 'addressed':
        service.begin_addressed(1)
    else:
        state.generation_inflight.add(1)
    scheduled = []
    async def handle(*_args, **_kwargs):
        pytest.fail('Busy group must not enter generation')
    asyncio.run(flush(service, handle, schedule=lambda gid, **kwargs: scheduled.append((gid, kwargs))))
    assert state.message_buffers[1] == [original]
    assert scheduled == [(1, {'delay': 1})]


@pytest.mark.parametrize('cancelled', [False, True])
def test_generation_failure_cleans_own_task_and_reschedules_messages_received_during_generation(cancelled):
    async def run():
        state = GroupSessionState(message_buffers={1: [item(source='first')]})
        service = GroupSessionService(state)
        scheduled = []
        async def handle(*_args, **_kwargs):
            state.message_buffers[1] = [item(22, source='later')]
            if cancelled:
                raise asyncio.CancelledError
            raise RuntimeError('model failed')
        task = asyncio.create_task(flush(service, handle, schedule=lambda gid, **kwargs: scheduled.append(gid)))
        state.buffer_tasks[1] = task
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await task
        assert not state.generation_inflight and not state.buffer_tasks
        assert [entry.source_message_id for entry in state.message_buffers[1]] == ['later']
        assert scheduled == [1]
    asyncio.run(run())


def test_contextual_search_drains_and_cancels_only_its_group_task():
    cancelled = []
    waiting = item(source='context')
    state = GroupSessionState(message_buffers={1: [waiting], 2: [item(22)]}, buffer_tasks={
        1: SimpleNamespace(done=lambda: False, cancel=lambda: cancelled.append(1)),
        2: SimpleNamespace(done=lambda: False, cancel=lambda: cancelled.append(2)),
    })
    assert GroupSessionService(state).drain_for_contextual_search(1) == [waiting]
    assert cancelled == [1]
    assert set(state.message_buffers) == {2} and set(state.buffer_tasks) == {2}
