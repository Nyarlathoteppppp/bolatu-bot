import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import nonebot
import pytest

nonebot.init()
from nonebot.adapters.onebot.v11 import Message
from qq_social_agent import plugin
from qq_social_agent.deepseek_client import ReplyDecision, _parse_tool_routing_decision
from qq_social_agent.conversation_tool_routing import _tool_request_from_llm_route, _finalize_routed_tool_plan
from qq_social_agent.group_tool_execution import execute_group_tools
from qq_social_agent.pipeline_types import GeneratedImage, PipelineState, ToolKind, ToolRequest, ToolResult
from qq_social_agent.private_message_types import PrivateGenerationContext, PrivateTurn
from qq_social_agent.private_reply_delivery import PrivateReplyServices, generate_and_send_private_reply
from qq_social_agent.tool_registry import ToolRegistry, ToolSpec
from qq_social_agent.tool_router import ToolRoutePlan
from qq_social_agent.tools import image_generation
from qq_social_agent.jev_client import JevClient
from test_plugin_low_value import _pending_approval, _use_temp_plugin_memory, FakeApprovalBot

IMAGE = GeneratedImage(base64.b64encode(b'test-png').decode(), '雪花', image_generation.IMAGE_MODEL)
REQUEST = ToolRequest(ToolKind.IMAGE_GENERATION, query='画一张雪花', required=True)


def test_image_api_fixed_low_and_artifacts_excluded_from_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv('VERYSADAI_API_KEY', 'test-image-key')
    monkeypatch.setattr(image_generation, 'IMAGE_DIRECTORY', tmp_path / '.pi' / 'generated-images')
    original_client = httpx.AsyncClient
    def handler(request):
        assert str(request.url) == image_generation.IMAGE_ENDPOINT
        assert request.headers['authorization'] == 'Bearer test-image-key'
        assert request.headers['accept-encoding'] == 'identity'
        assert json.loads(request.content) == {
            'model': image_generation.IMAGE_MODEL, 'prompt': REQUEST.query, 'n': 1,
            'quality': 'low', 'size': '1024x1024', 'output_format': 'png',
        }
        return httpx.Response(200, json={'data': [{'b64_json': IMAGE.base64_data}], 'size': '1254x1254',
                                        'usage': {'input_tokens': 16, 'output_tokens': 515}})
    monkeypatch.setattr(image_generation.httpx, 'AsyncClient',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    registry = ToolRegistry()
    registry.register(ToolSpec(ToolKind.IMAGE_GENERATION, '生成图片', image_generation.generate_image))
    result = asyncio.run(registry.execute(REQUEST))
    assert result.ok
    assert result.generated_images[0].file_ref == IMAGE.file_ref
    assert result.metadata['actual_size'] == '1254x1254'
    from pathlib import Path
    assert Path(result.metadata['saved_path']).read_bytes() == b'test-png'
    assert IMAGE.base64_data not in repr(result)
    assert IMAGE.base64_data not in json.dumps(dict(result.metadata))


@pytest.mark.parametrize('status', [403, 500])
def test_image_http_failure_has_no_artifact(monkeypatch, status):
    monkeypatch.setenv('VERYSADAI_API_KEY', 'test-image-key')
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={'error': 'unavailable'}))
    monkeypatch.setattr(image_generation.httpx, 'AsyncClient',
                        lambda **kwargs: original_client(transport=transport, **kwargs))
    registry = ToolRegistry()
    registry.register(ToolSpec(ToolKind.IMAGE_GENERATION, '生成图片', image_generation.generate_image))
    result = asyncio.run(registry.execute(REQUEST))
    assert result.status == 'error'
    assert not result.generated_images
    assert 'test-image-key' not in result.error


def test_missing_image_key_is_unavailable(monkeypatch):
    monkeypatch.delenv('VERYSADAI_API_KEY', raising=False)
    result = asyncio.run(image_generation.generate_image(REQUEST))
    assert result.status == 'unavailable'
    assert not result.generated_images


def test_image_router_keeps_long_visual_prompt_and_replaces_ignore_action():
    prompt = '水彩风格，一只白猫坐在校园树下，旁边的牌子上写着晚安，横向构图。' * 8
    route = _parse_tool_routing_decision(json.dumps({'tool': 'image_generation', 'query': prompt, 'confidence': 1}))
    request = _tool_request_from_llm_route(route, fallback_text='帮我画这个')
    assert request.kind == ToolKind.IMAGE_GENERATION and request.query == prompt
    decision, plan = _finalize_routed_tool_plan(
        ReplyDecision(False, 1, 'test', action='ignore'), ToolRoutePlan((request,)),
        text='帮我画这个', context_recent=[], addressed_bot=True, group_id=1, user_id=2, source_message_id='3',
    )
    assert decision.should_reply and decision.action == 'answer' and decision.tool == 'image_generation'
    assert plan.first(ToolKind.IMAGE_GENERATION).query == prompt
    assert _parse_tool_routing_decision('{"tool":"image_generation","query":""}').tool == 'none'


def test_jev_image_choice_is_valid():
    client = JevClient(api_key='test-key')
    async def evaluate(**kwargs):
        assert 'image_generation' in kwargs['questions']['tool_choice']['criteria']
        return {'answers': {'need_tool': {'noul': .99},
                            'tool_choice': {'choice': 'image_generation', 'confidence': .99}}}
    client.evaluate = evaluate
    result = asyncio.run(client.route_tool(persona=SimpleNamespace(name='风雪'), recent_messages=[],
                                          current_text='帮我画一张猫', current_nickname='A', addressed=True))
    assert result.tool == 'image_generation'


def test_group_tool_result_queues_image_before_any_send():
    registry = ToolRegistry()
    registry.register(ToolSpec(ToolKind.IMAGE_GENERATION, '生成图片',
                              lambda request: ToolResult(request.kind, 'ok', generated_images=(IMAGE,))))
    state = PipelineState('image-test', 1, 2, 'A', REQUEST.query, True)
    result = asyncio.run(execute_group_tools(
        decision=ReplyDecision(True, 1, 'image', action='answer', need_tool=True, tool='image_generation'),
        tool_plan=ToolRoutePlan((REQUEST,)), pipeline_state=state, group_id=1, user_id=2, text=REQUEST.query,
        market_intents=[], market_context_task=None, prefetched_market_request=None, tool_registry=registry,
        market_intents_from_decision=lambda *args, **kwargs: [], execute_fresh_tool_request=None,
        fresh_tool_failure_context=lambda *args, **kwargs: '', combine_text_sections=plugin._combine_text_sections,
        record_metric_event=lambda *args, **kwargs: None, logger=plugin.logger,
    ))
    assert result.direct_candidates and result.direct_candidates[0].text == ''
    assert state.tool_result(ToolKind.IMAGE_GENERATION).generated_images == (IMAGE,)
    assert state.sent_message_ids == ()


def test_group_image_only_delivery_quotes_and_does_not_resend(monkeypatch, tmp_path):
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    state = PipelineState('image-test', 1026813421, 184589072, 'A', REQUEST.query, True)
    state.add_tool_result(ToolResult(ToolKind.IMAGE_GENERATION, 'ok', generated_images=(IMAGE,)))
    candidate = plugin.PendingApprovalCandidate(1, '', 'answer', '生图工具结果')
    approval = replace(_pending_approval(), pipeline_state=state, source_message_id='123', candidates=(candidate,))
    bot = FakeApprovalBot()
    async def run():
        for _ in range(2):
            await plugin._send_approved_group_reply(bot, approval, candidate, approver_id=None,
                                                    high_quality=False, notify_success=False)
    asyncio.run(run())
    assert len(bot.group_messages) == 1
    message = Message(bot.group_messages[0][1])
    assert message[0].type == 'reply'
    assert message['image'][0].data['file'] == IMAGE.file_ref
    assert '[生成图片：雪花]' in store.recent_messages(approval.group_id, 5)[-1].text


def test_group_unknown_image_send_not_retried(monkeypatch, tmp_path):
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    state = PipelineState('image-test', 1026813421, 184589072, 'A', REQUEST.query, True)
    state.add_tool_result(ToolResult(ToolKind.IMAGE_GENERATION, 'ok', generated_images=(IMAGE,)))
    candidate = plugin.PendingApprovalCandidate(1, '', 'answer', '生图工具结果')
    approval = replace(_pending_approval(), pipeline_state=state, candidates=(candidate,))
    attempts = []
    async def send(*args, **kwargs):
        attempts.append(args)
        raise asyncio.TimeoutError()
    monkeypatch.setattr(plugin, '_send_group_message', send)
    async def run():
        for _ in range(2):
            await plugin._send_approved_group_reply(FakeApprovalBot(), approval, candidate, approver_id=None,
                                                    high_quality=False, notify_success=False)
    asyncio.run(run())
    assert len(attempts) == 1
    assert approval.delivery_progress[''].uncertain_index == 0


def test_group_private_redirect_sends_image_once_and_records_private_memory(monkeypatch, tmp_path):
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    state = PipelineState('image-test', 1026813421, 3370998238, 'A', REQUEST.query, True, private_reply_user_id=3370998238)
    state.add_tool_result(ToolResult(ToolKind.IMAGE_GENERATION, 'ok', generated_images=(IMAGE,)))
    candidate = plugin.PendingApprovalCandidate(1, '', 'answer', '生图工具结果')
    approval = replace(_pending_approval(), pipeline_state=state, candidates=(candidate,))
    bot = FakeApprovalBot()
    async def run():
        for _ in range(2):
            await plugin._send_approved_group_reply(bot, approval, candidate, approver_id=None,
                                                    high_quality=False, notify_success=False)
    asyncio.run(run())
    assert not bot.group_messages and len(bot.private_messages) == 1
    assert state.stage.value == 'completed'
    assert Message(bot.private_messages[0][1])['image'][0].data['file'] == IMAGE.file_ref
    assert '[生成图片：雪花]' in store.recent_messages(plugin._private_chat_id(3370998238), 5)[-1].text


def test_private_image_delivery_needs_no_text_generation(monkeypatch, tmp_path):
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    turn = PrivateTurn(SimpleNamespace(self_id=1801507496), 987654321, plugin._private_chat_id(987654321),
                       1801507496, '奈亚子', '42', 'image-test', REQUEST.query, '', 1)
    context = PrivateGenerationContext(
        persona=SimpleNamespace(name='张风雪'), recent_messages=(), current_text=turn.text,
        current_nickname=turn.nickname, decision=ReplyDecision(True, 1, 'image', action='answer'),
        generated_images=(IMAGE,), **dict.fromkeys(('market_context', 'fresh_context', 'memory_context',
        'member_context', 'memory_atoms_context', 'style_context', 'raw_corpus_context', 'jargon_context',
        'recall_feedback_context', 'speaker_context', 'priority_context'), ''),
    )
    sent = []
    async def send(bot, **kwargs): sent.append(kwargs['message'])
    services = PrivateReplyServices(
        memory=store, private_meme_library=SimpleNamespace(turn_gate=lambda *args, **kwargs:
                    SimpleNamespace(allowed=False, reason='test', messages_since_last_meme=1)),
        get_deepseek_client=lambda: None, send_private_message=send,
        message_with_reply_quote=plugin._message_with_reply_quote, sanitize_generated_text=lambda text: text,
        private_meme_context_eligible=lambda **kwargs: False, action_failed_summary=str,
        record_metric_event=lambda *args, **kwargs: None, logger=plugin.logger,
    )
    asyncio.run(generate_and_send_private_reply(turn, context, services=services))
    assert len(sent) == 1 and sent[0]['reply'][0].data['id'] == '42'
    assert sent[0]['image'][0].data['file'] == IMAGE.file_ref
    assert store.recent_messages(turn.chat_id, 5)[-1].text == '\n[生成图片：雪花]'


def test_tool_manifest_and_active_reply_fallback_display(monkeypatch, tmp_path):
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    plugin.local_plugin_registry.reload()
    plugin._register_plugin_runtime_tools()
    assert any(spec.kind == ToolKind.IMAGE_GENERATION for spec in plugin.tool_registry.available())
    assert plugin._model_fallback_labels('reply', 'verysadai/gpt-6.1-sol') == (
        'lingsuan/gpt-6.1-sol', 'deepseek/deepseek-flash',
    )
    state = plugin._admin_tools_state(None)
    reply = next(row for row in state['models'] if row['route'] == 'reply')
    assert reply['fallback'] == 'lingsuan/gpt-6.1-sol → deepseek/deepseek-flash'
