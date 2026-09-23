"""Dialogue regressions; synthetic events only, no real messages sent."""
import asyncio
import json
from types import SimpleNamespace

import nonebot
import pytest
nonebot.init()
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from qq_social_agent import plugin
from qq_social_agent.deepseek_client import DeepSeekClient, ReplyDecision, ToolRoutingDecision
from qq_social_agent.jev_client import JevClient, _timing_looks_like_reply_to_other
from qq_social_agent.reference_resolver import ReferenceResolution
from qq_social_agent.speaker_context import _format_speaker_reference_context
from qq_social_agent.tool_router import route_tools
from qq_social_agent.pipeline_types import ToolKind


def event(text, *, at=False, reply=None):
    message = Message(text)
    if at:
        message = Message(MessageSegment.at(1801507496)) + message
    return SimpleNamespace(user_id=2061999520, group_id=1026813421, self_id=1801507496,
        to_me=False, message=message, reply=reply,
        sender=SimpleNamespace(card="测试群友", nickname=""), get_plaintext=lambda: text)


@pytest.mark.parametrize("at,text", [(True,"如何用C实现闭包"), (False,"风雪，如何用C实现闭包")])
def test_target_user_direct_address_is_not_silenced(at, text):
    bot=SimpleNamespace(self_id=1801507496)
    incoming=event(text, at=at)
    assert plugin._mentioned_bot(incoming, bot)
    policy=plugin.app_config.group_user_policy(2061999520)
    assert not policy.memory_only
    assert not policy.addressed_question_private_reply
    result=plugin._enforce_addressed_reply_decision(
        ReplyDecision(False, 0.1, "test", action="ignore"), addressed_bot=True, text=text)
    assert result.should_reply and result.action=="answer"
    assert "注：" not in plugin._message_context_text(incoming, bot_id=bot.self_id)


def test_reply_keeps_current_request_tail_and_identity_stays_out_of_text():
    text="请分析这段代码："+"x = x + 1; "*25+"最后请解释闭包环境的生命周期"
    reply=SimpleNamespace(user_id=1801507496,sender=SimpleNamespace(card="风雪",nickname=""),
                          message=Message("前面的代码是示例"))
    result=plugin._message_context_text(event(text,reply=reply),bot_id=1801507496)
    assert text in result
    assert "你自己" not in result
    context=_format_speaker_reference_context(current_user_id=2061999520,current_nickname="测试群友",
        current_text=text,recent_messages=[],reference_resolution=ReferenceResolution(),
        mentioned=False,replied_to_bot=True,addressed_bot=True,self_id=1801507496)
    assert "风雪和张风雪都是你自己" in context


@pytest.mark.parametrize("text", ["消息回复不了，怎么修？", "怎么回复这条消息？", "服务器收不到消息回复"])
def test_technical_reply_words_are_not_a_reply_to_another_person(text):
    assert not _timing_looks_like_reply_to_other(text)
    client=JevClient(api_key="test")
    async def evaluate(**kwargs):
        return {"answers":{"following_bot":{"noul":0},"has_concrete_content":{"noul":.99}}}
    client.evaluate=evaluate
    result=asyncio.run(client.timing_gate(persona=SimpleNamespace(decision_prompt=""),
        recent_messages=[],current_text=text,current_nickname="测试群友"))
    assert result.channel.value=="text"


def test_real_reply_envelope_to_other_still_suppressed():
    assert _timing_looks_like_reply_to_other("甲[#11111]回复乙[#22222]消息【乙说：hi；甲回复乙：你好】")
    assert not _timing_looks_like_reply_to_other("甲[#11111]回复风雪[#07496]消息【风雪说：hi；甲回复风雪：你好】")


@pytest.mark.parametrize("text", ["我是不是永远拿不到offer了", "这段代码会不会崩", "硬币正面概率怎么算", "概率论怎么复习"])
def test_keywords_do_not_force_subjective_forecasting(text):
    plan=route_tools(text,market_intents=[],fresh_intent=None,addressed=True)
    assert plan.first(ToolKind.PROBABILITY) is None


@pytest.mark.parametrize("tool,reply,expected", [
 ("market", {"tool":"market","symbols":[{"kind":"stock","symbol":"TSLA"}]}, "market"),
 ("fresh_search", {"tool":"fresh_search","query":"Python 3.13 最新补丁版本"}, "fresh_search"),
 ("fresh_search", {"tool":"market","symbols":[{"kind":"stock","symbol":"TSLA"}]}, "none"),
])
def test_positive_jev_route_gets_grounded_llm_parameters(tool,reply,expected):
    client=DeepSeekClient.__new__(DeepSeekClient)
    client.prompts=SimpleNamespace(render=lambda *args,**kw: json.dumps(kw,ensure_ascii=False))
    async def route(**kwargs):
        assert kwargs['speaker_context']=='已解析对象 Python 3.13'
        return ToolRoutingDecision(tool=tool,query="那个现在呢",confidence=.9)
    client.jev_client=SimpleNamespace(available=True,route_tool=route)
    calls=[]
    async def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])
    client._chat_completion=completion
    result=asyncio.run(client.route_tool_use(persona=SimpleNamespace(name="风雪",decision_prompt=""),
        recent_messages=[],current_text="那个现在呢",current_nickname="A",addressed=True,
        decision_action="answer",decision_reason="test",speaker_context="已解析对象 Python 3.13"))
    assert len(calls)==1
    assert result.tool==expected
    if expected=='market':assert result.symbols[0].symbol=='TSLA'
    if expected=='fresh_search':assert result.query=='Python 3.13 最新补丁版本'


def test_confident_no_tool_avoids_llm_routing():
    client=DeepSeekClient.__new__(DeepSeekClient)
    client.prompts=SimpleNamespace(render=lambda *args,**kw: '')
    async def route(**kwargs):return ToolRoutingDecision(tool='none')
    async def completion(**kwargs):pytest.fail('no LLM router needed')
    client.jev_client=SimpleNamespace(available=True,route_tool=route)
    client._chat_completion=completion
    result=asyncio.run(client.route_tool_use(persona=SimpleNamespace(name='风雪',decision_prompt=''),
        recent_messages=[],current_text='今天好冷',current_nickname='A',addressed=True,
        decision_action='reply',decision_reason='test'))
    assert result.tool=='none'
