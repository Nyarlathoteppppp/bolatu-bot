"""Incoming content services can run without assembling the main plugin."""

import asyncio
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from qq_social_agent.forward_context import ForwardContextPolicy, ForwardContextService
from qq_social_agent.media_context import ImageOcrResult
from qq_social_agent.message_summary import MessageSummaryPolicy, MessageSummaryService


def test_forward_records_preserve_speaker_and_share_image_budget_across_payloads():
    calls = []

    class Images:
        async def ocr_image_segment(self, bot, data):
            calls.append(data['url'])
            if data['url'] == 'failed':
                raise RuntimeError('OCR unavailable')
            return ImageOcrResult(image_key=data['url'], text='成绩单')

    def record(user_id, name, image):
        return {'sender': {'user_id': user_id, 'nickname': name},
                'content': [{'type': 'image', 'data': {'url': image}}]}

    service = ForwardContextService(
        summaries=MessageSummaryService(None, MessageSummaryPolicy()), images=Images(),
        policy=ForwardContextPolicy(timezone=ZoneInfo('Asia/Shanghai'), max_images=2),
    )
    lines = asyncio.run(service.records_from_payloads(None, [
        {'messages': [record(10001, '甲', 'failed')]},
        {'messages': [record(10002, '乙', 'ok'), record(10003, '丙', 'not-read')]},
    ]))
    assert calls == ['failed', 'ok']
    assert '[图:成绩单]' not in lines[0]
    assert '乙[#10002]:' in lines[1] and '[图:成绩单]' in lines[1]
    assert '丙[#10003]:' in lines[2] and '[图:成绩单]' not in lines[2]


def test_summary_uses_injected_client_and_preserves_forward_attribution():
    calls = []

    class Summarizer:
        async def summarize_long_message(self, **kwargs):
            calls.append(kwargs)
            return '  原发言人\n讨论课程  '

    policy = MessageSummaryPolicy(threshold=10, forward_threshold=20, source_limit=40)
    service = MessageSummaryService(Summarizer(), policy)
    raw = '甲：讨论课程\n乙：提供资料\n' * 10
    summary = asyncio.run(service.forward_records(raw, nickname='转发人'))
    assert summary == '原发言人 讨论课程'
    assert calls[0]['speaker_label'] == '多位原发言人（由转发人转发）'
    assert calls[0]['text'] == raw[:40]
    assert calls[0]['original_chars'] == len(raw)
    assert asyncio.run(service.message_text('  短话  ', nickname='甲', chat_label='QQ 群聊')) == '  短话  '
    assert len(calls) == 1


def test_forward_fetch_failure_keeps_trying_the_next_existing_forward_id():
    calls = []

    class Bot:
        async def call_api(self, api, **kwargs):
            calls.append((api, kwargs['id']))
            if kwargs['id'] == 'missing':
                raise RuntimeError('missing record')
            return {'messages': [{'sender': {'nickname': '原发言人'}, 'content': '一句话'}]}

    event = SimpleNamespace(message=[
        SimpleNamespace(type='forward', data={'id': 'missing'}),
        SimpleNamespace(type='forward', data={'id': 'existing'}),
    ])
    service = ForwardContextService(
        summaries=MessageSummaryService(None, MessageSummaryPolicy()), images=None,
        policy=ForwardContextPolicy(timezone=ZoneInfo('Asia/Shanghai')),
    )
    text = asyncio.run(service.context_text(Bot(), event, nickname='转发人'))
    assert text == '转发人传了聊天记录，内容如下：\n原发言人: 一句话'
    assert calls == [('get_forward_msg', 'missing'), ('get_forward_msg', 'existing')]
