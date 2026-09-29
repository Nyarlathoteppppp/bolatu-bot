from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import nonebot
import pytest

nonebot.init()

from nonebot.adapters.onebot.v11 import Message, MessageSegment
from qq_social_agent import plugin
from qq_social_agent.discourse_state import resolve_group_discourse
from qq_social_agent.ellipsis_resolver import EllipsisJudgement, EllipsisResolution
from qq_social_agent.image_read_state import resolved_image_message
from qq_social_agent.media_context import ImageOcrContext, ImageOcrService
from qq_social_agent.memory import MemoryStore
from qq_social_agent.resolver_result import AMBIGUOUS, RESOLVED


class SlowVision:
    def __init__(self, text='一只猫在伸手要零食'):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.calls = []
        self.text = text

    async def recognize(self, target):
        self.calls.append(target)
        self.started.set()
        await self.finish.wait()
        return self.text


class NoApiBot:
    self_id = 1801507496

    async def call_api(self, api, **kwargs):
        raise AssertionError(f'Existing image URL should not call {api}')


def history(store, group, limit):
    return store.images.enrich(store.recent_messages(group, limit))


def add_image(store, group=1, source='42', url='https://example.com/cat.png'):
    store.add_message(group, 100, 'A', '[图片]', created_at=1000,
                      source_message_id=source,
                      message_segments_json=json.dumps([{'type': 'image', 'data': {'url': url, 'file': 'cat.png'}}]))
    return store.admin_message_by_source(group, source)


def resolved(source='42', status=RESOLVED):
    return SimpleNamespace(ellipsis=EllipsisResolution(
        kind='ITEM_DEIXIS', status=status, source_message_id=source,
        source_key=f's{source}', source_text='[图片]', confidence=0.95))


def test_pending_followup_joins_original_and_result_keeps_source_order(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    row = add_image(store)
    store.add_message(1, 200, 'B', '这个图什么意思', source_message_id='43', created_at=1001)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)
    monkeypatch.setattr(plugin, 'memory', store)
    monkeypatch.setattr(plugin, 'image_ocr_service', service)

    async def run():
        first = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await vision.started.wait()
        snapshot = history(store, 1, 10)
        assert '正在识别' in snapshot[0].text
        assert '一只猫' not in snapshot[0].text
        second = asyncio.create_task(plugin._resolved_image_context(
            NoApiBot(), resolved(), snapshot, group_id=1))
        await asyncio.sleep(0)
        assert not second.done()
        vision.finish.set()
        result, context = await asyncio.gather(first, second)
        assert result.text in context
        refreshed = store.images.enrich(snapshot)
        assert '正在识别' not in refreshed[0].text
        assert refreshed[0].text.count('[图片接收与识别状态]') == 1
        assert [(m.source_message_id, m.created_at) for m in refreshed] == [('42', 1000), ('43', 1001)]
        assert store.admin_message_by_source(1, '42')['id'] == row['id']
        assert len(vision.calls) == 1
    asyncio.run(run())


def test_cancelled_follower_does_not_cancel_reading(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    row = add_image(store)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)

    async def run():
        first = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await vision.started.wait()
        follower = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await asyncio.sleep(0)
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        vision.finish.set()
        assert (await first).text
        assert 'ready' in history(store, 1, 5)[0].text
        assert len(vision.calls) == 1
    asyncio.run(run())


def test_same_image_in_different_source_jobs_has_one_vision_call(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    first = add_image(store, group=1)
    second = add_image(store, group=2)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)

    async def run():
        tasks = [asyncio.create_task(store.images.read(NoApiBot(), service, row)) for row in (first, second)]
        await vision.started.wait()
        await asyncio.sleep(0)
        vision.finish.set()
        await asyncio.gather(*tasks)
        assert len(vision.calls) == 1
        assert 'ready' in history(store, 1, 1)[0].text
        assert 'ready' in history(store, 2, 1)[0].text
    asyncio.run(run())


def test_empty_result_means_received_but_unreadable_not_no_image(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    row = add_image(store)
    service = ImageOcrService(napcat_ocr_enabled=False)
    result = asyncio.run(store.images.read(NoApiBot(), service, row))
    assert not result.text
    text = history(store, 1, 1)[0].text
    assert '图片已收到，但本次识别没有获得内容' in text
    assert 'unavailable' in text


def test_restart_keeps_ready_and_does_not_claim_old_job_is_running(tmp_path):
    path = tmp_path / 'm.db'
    store = MemoryStore(path)
    row = add_image(store)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)

    async def run():
        live = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await vision.started.wait()
        reopened = MemoryStore(path)
        assert '尚无识别结果' in history(reopened, 1, 1)[0].text
        assert '正在识别' not in history(reopened, 1, 1)[0].text
        # A second reader must not reset the original process's active task.
        assert '正在识别' in history(store, 1, 1)[0].text
        vision.finish.set()
        await live
        again = MemoryStore(path)
        assert '一只猫' in history(again, 1, 1)[0].text
        assert 'ready' in history(again, 1, 1)[0].text
    asyncio.run(run())


def test_clear_context_cancels_job_and_does_not_resurrect_message(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    row = add_image(store)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)

    async def run():
        task = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await vision.started.wait()
        store.reset_group_messages(1)
        with pytest.raises(asyncio.CancelledError):
            await task
        vision.finish.set()
        await service.aclose()
        assert history(store, 1, 10) == []
        assert store.conn.execute('select count(*) from image_read_states').fetchone()[0] == 0
    asyncio.run(run())


def test_binding_does_not_guess_latest_image_and_is_scoped_to_group(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    add_image(store, source='42')
    add_image(store, source='44', url='https://example.com/dog.png')
    add_image(store, group=2, source='99')
    recent = history(store, 1, 10)
    assert resolved_image_message(resolved('42').ellipsis, recent).source_message_id == '42'
    assert resolved_image_message(resolved('42', AMBIGUOUS).ellipsis, recent) is None
    assert resolved_image_message(resolved('99').ellipsis, recent) is None


def test_unquoted_resolved_historical_image_sets_canonical_media_present(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    add_image(store)

    class Jev:
        async def resolve_ellipsis(self, **kwargs):
            return EllipsisJudgement(kind='ITEM_DEIXIS', inherit_from='s42', confidence=0.99)

    state = asyncio.run(resolve_group_discourse(
        current_text='这个图什么意思', current_user_id=200, current_nickname='B', self_id=999,
        recent_messages=history(store, 1, 10), jev=Jev()))
    assert state.ellipsis.source_message_id == '42'
    assert state.media_present


def test_text_only_quote_uses_recorded_image_and_joins_job(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    row = add_image(store)
    monkeypatch.setattr(plugin, 'memory', store)
    vision = SlowVision()
    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=vision)
    monkeypatch.setattr(plugin, 'image_ocr_service', service)
    event = SimpleNamespace(group_id=1, user_id=200, message=MessageSegment.reply(42) + Message('什么意思'), reply=None)

    async def run():
        original = asyncio.create_task(store.images.read(NoApiBot(), service, row))
        await vision.started.wait()
        reply = asyncio.create_task(plugin._image_ocr_context_for_event(
            NoApiBot(), event, group_allowed=True, group_id=1, user_id=200, correlation_id='quoted'))
        await asyncio.sleep(0)
        vision.finish.set()
        _, result = await asyncio.gather(original, reply)
        assert '引用消息 42 的图片' in result.text
        assert '一只猫' in result.text
        assert len(vision.calls) == 1
        hint = plugin._reply_hint_for_reference(event, current_text='什么意思', self_id=999)
        assert hint.exists and hint.message_id == '42' and hint.author_id == 100
    asyncio.run(run())


def test_image_is_visible_before_first_network_await(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    monkeypatch.setattr(plugin, 'memory', store)
    monkeypatch.setattr(plugin, '_record_metric_event', lambda *args, **kwargs: None)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_reply_reference(*args, **kwargs):
        entered.set()
        await release.wait()
        return None

    monkeypatch.setattr(plugin, '_resolve_reply_reference_for_event', slow_reply_reference)
    event = SimpleNamespace(
        group_id=1026813421, user_id=100, self_id=NoApiBot.self_id,
        message_id=42, time=time.time(), message=Message(MessageSegment.image('cat.png')),
        sender=SimpleNamespace(card='', nickname='A'), reply=None, to_me=False,
        get_plaintext=lambda: '',
    )

    async def run():
        task = asyncio.create_task(plugin._handle_group_message_scoped(NoApiBot(), event, correlation_id='image'))
        waiter = asyncio.create_task(entered.wait())
        await asyncio.wait([task, waiter], return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
            pytest.fail('Image ingress did not reach reference resolution')
        await waiter
        recent = history(store, event.group_id, 5)
        assert len(recent) == 1
        assert recent[0].source_message_id == '42'
        assert '尚无识别结果' in recent[0].text
        assert json.loads(recent[0].message_segments_json)[0]['type'] == 'image'
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())


def test_group_text_only_quote_passes_ingress_media_gate(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    group = 1026813421
    add_image(store, group=group)
    monkeypatch.setattr(plugin, 'memory', store)
    monkeypatch.setattr(plugin, '_record_metric_event', lambda *args, **kwargs: None)
    seen = []

    class Vision:
        async def recognize(self, target):
            return '猫在伸手要零食'

    async def empty(*args, **kwargs):
        return ''

    async def no_reference(*args, **kwargs):
        return None

    async def worth(**kwargs):
        seen.append(kwargs['item_count'])
        return True

    async def locked(*args, **kwargs):
        seen.append(kwargs['preprocessed_text'])
        assert kwargs['pipeline_state'].reply_image_present

    class Content:
        async def context_for_event(self, *args, **kwargs):
            return SimpleNamespace(text='', file_count=0, voice_count=0, file_status='none', voice_status='none')

    monkeypatch.setattr(plugin, '_resolve_reply_reference_for_event', no_reference)
    monkeypatch.setattr(plugin, 'file_metadata_context_for_event', empty)
    monkeypatch.setattr(plugin, 'content_ingestion_service', Content())
    monkeypatch.setattr(plugin, '_media_worth_reading', worth)
    monkeypatch.setattr(plugin, '_should_defer_group_reply_flow', lambda *args, **kwargs: False)
    monkeypatch.setattr(plugin, '_handle_group_message_locked', locked)
    monkeypatch.setattr(plugin, 'image_ocr_service', ImageOcrService(napcat_ocr_enabled=False, primary_ocr=Vision()))
    text = '这个图什么意思'
    event = SimpleNamespace(
        group_id=group, user_id=1535071184, self_id=NoApiBot.self_id,
        message_id=43, time=time.time(),
        message=MessageSegment.reply(42) + MessageSegment.at(NoApiBot.self_id) + Message(text),
        sender=SimpleNamespace(card='', nickname='B'), reply=None, to_me=False,
        get_plaintext=lambda: text,
    )
    asyncio.run(plugin._handle_group_message_scoped(NoApiBot(), event, correlation_id='question'))
    assert seen[0] == 1
    assert '猫在伸手' in seen[1]


def test_usable_url_skips_get_image_round_trip():
    calls = []

    class Bot:
        async def call_api(self, api, **kwargs):
            calls.append(api)
            return {}

    class Vision:
        async def recognize(self, target):
            assert target == 'https://example.com/cat.png'
            return '猫'

    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=Vision())
    result = asyncio.run(service.ocr_image_segment(Bot(), {'file': 'cat.png', 'url': 'https://example.com/cat.png'}))
    assert result.text == '猫'
    assert calls == []


def test_partial_recognition_keeps_actual_image_number():
    class Vision:
        async def recognize(self, target):
            return '狗' if target.endswith('dog.png') else ''

    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=Vision())
    event = SimpleNamespace(message=[
        {'type': 'image', 'data': {'url': 'https://example.com/cat.png'}},
        {'type': 'image', 'data': {'url': 'https://example.com/dog.png'}},
    ])
    result = asyncio.run(service.context_for_event(NoApiBot(), event))
    assert result.image_count == 2 and result.ocr_count == 1
    assert result.text == '第2张图：狗'


def test_followup_retries_unread_image_without_rereading_successful_one(tmp_path):
    store = MemoryStore(tmp_path / 'm.db')
    segments = [{'type': 'image', 'data': {'url': f'https://example.com/{name}.png'}} for name in ('cat', 'dog')]
    store.add_message(1, 100, 'A', '[图片] [图片]', source_message_id='42', message_segments_json=json.dumps(segments))
    row = store.admin_message_by_source(1, '42')
    calls = []

    class Vision:
        async def recognize(self, target):
            calls.append(target)
            if target.endswith('cat.png') and calls.count(target) == 1:
                return ''
            return '猫' if target.endswith('cat.png') else '狗'

    service = ImageOcrService(napcat_ocr_enabled=False, primary_ocr=Vision())

    async def run():
        partial = await store.images.read(NoApiBot(), service, row)
        assert partial.ocr_count == 1 and partial.image_count == 2
        full = await store.images.read(NoApiBot(), service, row)
        assert full.ocr_count == 2
        assert '第1张图：猫' in full.text and '第2张图：狗' in full.text
        assert calls.count('https://example.com/cat.png') == 2
        assert calls.count('https://example.com/dog.png') == 1
    asyncio.run(run())
