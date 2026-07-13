import asyncio
from types import SimpleNamespace

import qq_social_agent.deepseek_client as deepseek_module
from qq_social_agent.deepseek_client import (
    ChatMessage,
    DeepSeekClient,
    _log_llm_usage,
    _format_context_with_local_focus,
    _filter_recent_bot_duplicate_candidates,
    _parse_jargon_terms,
    _parse_daily_review,
    _parse_long_message_summary,
    _parse_member_profile_draft,
    _parse_mid_memory,
    _parse_reply_candidates,
    _parse_reply_decision,
    ReplyCandidateDraft,
    _sanitize_reply,
    _usage_value,
    set_usage_recorder,
)
from qq_social_agent.persona import Persona
from qq_social_agent.prompts import PromptRegistry


def test_sanitize_empty_string_markers() -> None:
    assert _sanitize_reply("空字符串", 120) == ""
    assert _sanitize_reply("（空字符串）", 120) == ""
    assert _sanitize_reply('"（空字符串）"', 120) == ""


def test_context_marks_only_contiguous_recent_topic_as_high_priority() -> None:
    messages = [
        ChatMessage(1, 1, "A", "在人均弱智的教室只有我智力正常", False, 100.0),
        ChatMessage(1, 2, "B", "含铅", False, 400.0),
        ChatMessage(1, 2, "B", "[图片OCR: 墨尔本自来水很好喝]", False, 406.0),
    ]

    context = _format_context_with_local_focus(
        messages,
        formatter=lambda message: f"{message.nickname}: {message.text}",
    )

    older, focused = context.split("【紧邻当前消息的连续话题", 1)
    assert "智力正常" in older
    assert "智力正常" not in focused
    assert "含铅" in focused
    assert "墨尔本自来水" in focused


def test_recent_bot_duplicate_filter_blocks_same_core_punchline() -> None:
    candidates = (
        ReplyCandidateDraft("司马懿吧，苟到最后的才是赢家", "answer", "直接判断"),
        ReplyCandidateDraft("你更像贾诩，突出一个能活", "answer", "换角度"),
    )

    filtered = _filter_recent_bot_duplicate_candidates(
        candidates,
        ("你啊，司马懿吧——低调苟发育，苟到最后的才是赢家",),
    )

    assert [candidate.text for candidate in filtered] == ["你更像贾诩，突出一个能活"]


def test_sanitize_keeps_normal_reply() -> None:
    assert _sanitize_reply("人在，直接说事。", 120) == "人在，直接说事。"


def test_sanitize_trims_to_sentence_boundary() -> None:
    text = "第一句完整。第二句也完整。第三句会被截断在这里后面还有很多很多很多内容。"
    reply = _sanitize_reply(text, 14)
    assert reply == "第一句完整。第二句也完整。"
    assert reply.endswith("。")


def test_parse_reply_decision() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.82, "action": "tease", "mode": "natural", "reason": "有梗"}'
    )
    assert decision.should_reply
    assert decision.confidence == 0.82
    assert decision.action == "tease"
    assert decision.mode == "natural"
    assert decision.reason == "有梗"


def test_parse_reply_decision_maps_legacy_mode_to_action() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.7, "mode": "natural", "reason": "普通接话"}'
    )

    assert decision.should_reply
    assert decision.action == "reply"


def test_parse_reply_decision_action_market_sets_tool() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.8, "action": "market_check", "reason": "行情"}'
    )

    assert decision.should_reply
    assert decision.action == "market_check"
    assert decision.need_tool
    assert decision.tool == "market"


def test_parse_reply_decision_action_fresh_sets_context() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.8, "action": "fresh_context", "fresh_query": "世界杯 比分", "reason": "赛果"}'
    )

    assert decision.should_reply
    assert decision.action == "answer"
    assert decision.need_fresh_context


def test_parse_reply_decision_search_keeps_social_action() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.8, "action": "tease", '
        '"need_fresh_context": true, "fresh_query": "美国最新消息", "fresh_kind": "news", '
        '"reason": "先查再吐槽"}'
    )

    assert decision.should_reply
    assert decision.action == "tease"
    assert decision.need_fresh_context
    assert decision.fresh_query == "美国最新消息"


def test_parse_reply_decision_action_agree() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.74, "action": "agree", "reason": "观点说到点子上"}'
    )

    assert decision.should_reply
    assert decision.action == "agree"


def test_parse_reply_decision_action_care() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.81, "action": "care", "reason": "压力很大"}'
    )

    assert decision.should_reply
    assert decision.action == "care"


def test_parse_reply_decision_action_care_chinese_alias() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.81, "action": "关心", "reason": "明显低落"}'
    )

    assert decision.should_reply
    assert decision.action == "care"


def test_parse_reply_decision_action_answer() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.72, "action": "answer", "reason": "正常提问"}'
    )

    assert decision.should_reply
    assert decision.action == "answer"


def test_parse_reply_decision_action_answer_chinese_alias() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.72, "action": "正常回答", "reason": "正常交流"}'
    )

    assert decision.should_reply
    assert decision.action == "answer"


def test_parse_reply_decision_new_social_actions() -> None:
    for raw_action, expected in [
        ("observe", "observe"),
        ("接情绪", "echo_mood"),
        ("转话题", "shift_topic"),
        ("自嘲", "self_comment"),
        ("关系回应", "relationship_reply"),
    ]:
        decision = _parse_reply_decision(
            f'{{"should_reply": true, "confidence": 0.72, "action": "{raw_action}", "reason": "test"}}'
        )

        assert decision.should_reply
        assert decision.action == expected


def test_parse_reply_decision_extracts_json_block() -> None:
    decision = _parse_reply_decision(
        '```json\n{"should_reply": true, "confidence": 0.66, "action": "agree", "reason": "能补判断"}\n```'
    )

    assert decision.should_reply
    assert decision.action == "agree"


def test_parse_reply_decision_with_market_tool() -> None:
    decision = _parse_reply_decision(
        """
        {
          "should_reply": true,
          "confidence": 0.9,
          "mode": "market",
          "need_tool": true,
          "tool": "market",
          "symbols": [{"kind": "crypto", "symbol": "bitcoin", "display": "BTC"}],
          "comment_after_tool": true,
          "reason": "需要查行情"
        }
        """
    )

    assert decision.should_reply
    assert decision.need_tool
    assert decision.tool == "market"
    assert decision.comment_after_tool
    assert len(decision.symbols) == 1
    assert decision.symbols[0].kind == "crypto"
    assert decision.symbols[0].symbol == "bitcoin"
    assert decision.symbols[0].display == "BTC"


def test_parse_reply_decision_with_fresh_context() -> None:
    decision = _parse_reply_decision(
        """
        {
          "should_reply": true,
          "confidence": 0.76,
          "mode": "chat",
          "need_fresh_context": true,
          "fresh_query": "美国 伊朗 冲突 最新消息",
          "fresh_kind": "news",
          "reason": "涉及实时国际冲突"
        }
        """
    )

    assert decision.should_reply
    assert decision.need_fresh_context
    assert decision.fresh_query == "美国 伊朗 冲突 最新消息"
    assert decision.fresh_kind == "news"


def test_parse_reply_decision_invalid_json() -> None:
    decision = _parse_reply_decision("不是 json")
    assert not decision.should_reply
    assert decision.confidence == 0.0
    assert decision.reason == "invalid_json"


def test_parse_jargon_terms() -> None:
    terms = _parse_jargon_terms('{"terms":["柏拉图","zbzy","柏拉图"],"reason":"命中"}')

    assert terms == ("柏拉图", "zbzy")


def test_parse_jargon_terms_extracts_json_block() -> None:
    terms = _parse_jargon_terms('```json\n{"terms":["霓虹"]}\n```')

    assert terms == ("霓虹",)


def test_usage_value_reads_dict_and_object() -> None:
    assert _usage_value({"prompt_tokens": 12}, "prompt_tokens") == 12
    assert _usage_value(SimpleNamespace(total_tokens="34"), "total_tokens") == 34
    assert _usage_value(SimpleNamespace(total_tokens=None), "total_tokens") is None


def test_log_llm_usage_calls_recorder() -> None:
    events = []
    set_usage_recorder(lambda *args: events.append(args))
    try:
        _log_llm_usage(
            "decision",
            SimpleNamespace(usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}),
            model="deepseek-v4-flash",
        )
    finally:
        set_usage_recorder(None)

    assert events == [("decision", "deepseek-v4-flash", 12, 3, 15)]


def test_client_methods_use_expected_model_routes() -> None:
    client = object.__new__(DeepSeekClient)
    client.config = SimpleNamespace(
        max_tokens=120,
        temperature=0.7,
        thinking="disabled",
        reasoning_effort="high",
    )
    client.prompts = PromptRegistry()
    calls: list[tuple[str, str]] = []
    contents = {
        "decision": '{"should_reply":false,"confidence":0.1,"action":"ignore","reason":"test"}',
        "jargon": '{"terms":["柏拉图"]}',
        "reply": "一句话",
        "reply_candidates": (
            '{"candidates":['
            '{"text":"候选一句话","style":"自然","action":"reply"},'
            '{"text":"第二个候选","style":"温和","action":"reply"},'
            '{"text":"第三个候选","style":"调侃","action":"tease"}'
            ']}'
        ),
        "daily_review": "今天群里聊得挺热闹，我也算接上了几句。",
        "member_profile": '{"summary":"爱聊行情和代码","interests":["股票","代码"],"speaking_style":"短句吐槽","representative_texts":["股票又亏了"]}',
        "long_message_summary": '{"summary":"长消息主要是在吐槽股票亏钱和风险控制。"}',
        "mid_memory": '{"summary":"按人：A[#11111]说了事","recall_cues":["A[#11111]"]}',
        "style_learning": '{"rules":[{"situation":"聊亏钱","style":"短句吐槽","source_id":1}]}',
    }

    async def fake_chat_completion(*, task: str, route_name: str, request: dict[str, object]) -> object:
        calls.append((task, route_name))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=contents[task]),
                )
            ]
        )

    client._chat_completion = fake_chat_completion
    persona = Persona(
        id="test",
        name="张风雪",
        description="",
        prompt="人格",
        decision_prompt="决策人格",
        keywords=(),
        max_reply_chars=120,
        passive_reply_probability=0.5,
    )
    messages = [
        ChatMessage(
            group_id=1026813421,
            user_id=11111,
            nickname="A",
            text="股票又亏了",
            is_bot=False,
            created_at=1000.0,
        )
    ]

    async def run_all() -> None:
        await client.should_reply(
            persona=persona,
            recent_messages=messages,
            current_text="股票又亏了",
            current_nickname="A",
        )
        await client.select_jargon_terms(
            recent_messages=messages,
            current_text="柏拉图今天好安静",
            current_nickname="A",
            jargon_catalog="- 柏拉图：群名",
        )
        await client.reply(
            persona=persona,
            recent_messages=messages,
            current_text="股票又亏了",
            current_nickname="A",
            mentioned=False,
        )
        await client.reply_candidates(
            persona=persona,
            recent_messages=messages,
            current_text="股票又亏了",
            current_nickname="A",
            mentioned=False,
        )
        await client.daily_review(
            persona=persona,
            messages=messages,
            chat_label="QQ 群聊",
            today_label="2026-07-10",
        )
        await client.summarize_member_profile(
            messages=messages,
            member_label="A[#11111]",
        )
        await client.summarize_long_message(
            text="这是一条很长的群友消息，主要在说股票亏钱和风险控制。",
            speaker_label="A",
            original_chars=120,
        )
        await client.summarize_mid_memory(messages=messages)
        await client.learn_style_rules(messages=messages)

    asyncio.run(run_all())

    assert calls == [
        ("decision", "decision"),
        ("jargon", "jargon"),
        ("reply", "reply"),
        ("reply_candidates", "reply"),
        ("daily_review", "reply"),
        ("member_profile", "member_profile"),
        ("long_message_summary", "memory"),
        ("mid_memory", "memory"),
        ("style_learning", "style"),
    ]


def test_parse_long_message_summary() -> None:
    assert _parse_long_message_summary('{"summary":"  长消息说的是股票亏钱和风险控制。  "}') == (
        "长消息说的是股票亏钱和风险控制。"
    )
    assert _parse_long_message_summary("bad") == ""


def test_parse_member_profile_draft() -> None:
    draft = _parse_member_profile_draft(
        '{"summary":"爱聊行情和代码","interests":["股票","代码","股票"],'
        '"speaking_style":"短句吐槽","representative_texts":["股票又亏了"]}'
    )

    assert draft.summary == "爱聊行情和代码"
    assert draft.interests == ("股票", "代码")
    assert draft.speaking_style == "短句吐槽"
    assert draft.representative_texts == ("股票又亏了",)


def test_parse_mid_memory_keeps_evidence_and_maps_subject() -> None:
    messages = [
        ChatMessage(1, 123456, "小鸟", "我最近喜欢画画", False, 1000.0, id=41),
        ChatMessage(1, 999999, "路人", "她还喜欢摄影", False, 1001.0, id=42),
    ]
    draft = _parse_mid_memory(
        '{"summary":"小鸟聊爱好","recall_cues":["小鸟 画画"],"facts":['
        '{"kind":"preference","content":"小鸟最近喜欢画画","subject_message_id":41,'
        '"source_message_ids":[41,999],"confidence":1.5,"importance":0.8}]}' ,
        messages=messages,
    )

    assert draft.summary == "小鸟聊爱好"
    assert len(draft.facts) == 1
    assert draft.facts[0].subject_user_id == 123456
    assert draft.facts[0].evidence_message_ids == (41,)
    assert draft.facts[0].confidence == 1.0


def test_parse_daily_review_separates_public_reply_from_internal_learning() -> None:
    messages = [ChatMessage(1, 123456, "小鸟", "把这个叫乌木", False, 1000.0, id=51)]
    draft = _parse_daily_review(
        '{"public_reply":"今天又学了个怪词。","events":[],"member_changes":[],'
        '"jargon_candidates":[{"content":"乌木是群内称呼","subject_message_id":51,'
        '"source_message_ids":[51],"confidence":0.8,"importance":0.6}],'
        '"feedback_lessons":[{"content":"少说客服腔","confidence":0.9,"importance":0.9}]}' ,
        messages=messages,
    )

    assert draft.public_reply == "今天又学了个怪词。"
    assert draft.jargon_candidates[0].evidence_message_ids == (51,)
    assert draft.feedback_lessons[0].kind == "feedback_lesson"


def test_parse_daily_review_recovers_public_text_from_truncated_json_without_leaking_wrapper() -> None:
    draft = _parse_daily_review(
        '{"public_reply":"今天从课程聊到留学，后面又研究起工作。风雪今天也接了不少话，稍微有点话多。',
    )

    assert draft.public_reply == "今天从课程聊到留学，后面又研究起工作。风雪今天也接了不少话，稍微有点话多。"
    assert "public_reply" not in draft.public_reply


def test_parse_daily_review_drops_unrecoverable_json_instead_of_sending_backend_format() -> None:
    draft = _parse_daily_review('{"events":[')

    assert draft.public_reply == ""


def test_parse_reply_candidates_logs_diagnostic_when_short(monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(
        deepseek_module,
        "logger",
        SimpleNamespace(
            info=lambda message: logs.append(message),
            warning=lambda message: None,
        ),
    )

    candidates = _parse_reply_candidates(
        """
        {
          "candidates": [
            {"text": "第一条", "style": "自然", "action": "reply"},
            {"text": "第一条", "style": "重复", "action": "reply"},
            {"text": "", "style": "空", "action": "reply"},
            "bad"
          ]
        }
        """,
        max_chars=120,
        fallback_action="reply",
        limit=3,
    )

    assert len(candidates) == 1
    assert logs == [
        "qq_social_agent reply candidates parse diagnostic: "
        "raw_count=4 parsed_count=1 limit=3 "
        "dropped_reason=duplicate_text=1,empty_text=1,item_not_object=1"
    ]


def test_reply_candidates_retries_when_model_returns_too_few() -> None:
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.config = SimpleNamespace(
        max_tokens=300,
        thinking="disabled",
        reasoning_effort="low",
        temperature=0.6,
    )
    client.prompts = PromptRegistry()

    contents = [
        (
            '{"candidates":['
            '{"text":"第一条","style":"自然","action":"reply"},'
            '{"text":"第二条","style":"温和","action":"reply"}'
            ']}'
        ),
        (
            '{"candidates":['
            '{"text":"第一条","style":"自然","action":"reply"},'
            '{"text":"第二条","style":"温和","action":"reply"},'
            '{"text":"第三条","style":"调侃","action":"tease"}'
            ']}'
        ),
    ]
    requests: list[dict[str, object]] = []

    async def fake_chat_completion(*, task: str, route_name: str, request: dict[str, object]) -> object:
        requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=contents[len(requests) - 1]),
                )
            ]
        )

    client._chat_completion = fake_chat_completion
    persona = Persona(
        id="test",
        name="张风雪",
        description="",
        prompt="人格",
        decision_prompt="决策人格",
        keywords=(),
        max_reply_chars=120,
        passive_reply_probability=0.5,
    )

    candidates = asyncio.run(
        client.reply_candidates(
            persona=persona,
            recent_messages=[
                ChatMessage(
                    group_id=1026813421,
                    user_id=11111,
                    nickname="A",
                    text="我着急赶地铁",
                    is_bot=False,
                    created_at=1000.0,
                )
            ],
            current_text="我着急赶地铁",
            current_nickname="A[#11111]",
            mentioned=False,
        )
    )

    assert [candidate.text for candidate in candidates] == ["第一条", "第二条", "第三条"]
    assert len(requests) == 2
    retry_messages = requests[1]["messages"]
    assert isinstance(retry_messages, list)
    assert "必须给满 3 条" in retry_messages[-1]["content"]


def test_reply_candidates_includes_priority_context() -> None:
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.config = SimpleNamespace(
        max_tokens=260,
        thinking="disabled",
        reasoning_effort="low",
        temperature=0.6,
    )
    client.prompts = PromptRegistry()
    captured_requests: list[dict[str, object]] = []

    async def fake_chat_completion(*, task: str, route_name: str, request: dict[str, object]) -> object:
        captured_requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"candidates":['
                            '{"text":"小鸟别急呀，风雪陪你慢慢看","style":"温柔可爱","action":"care"},'
                            '{"text":"这事先别慌，风雪觉得可以一点点拆","style":"顺毛安慰","action":"care"},'
                            '{"text":"小鸟这句有点委屈欸，先抱一下再说","style":"亲近承接","action":"care"}'
                            ']}'
                        )
                    ),
                )
            ]
        )

    client._chat_completion = fake_chat_completion
    persona = Persona(
        id="test",
        name="张风雪",
        description="",
        prompt="人格",
        decision_prompt="决策人格",
        keywords=(),
        max_reply_chars=120,
        passive_reply_probability=0.5,
    )

    asyncio.run(
        client.reply_candidates(
            persona=persona,
            recent_messages=[],
            current_text="我有点难受",
            current_nickname="小鸟[#89072]",
            mentioned=True,
            action="care",
            priority_context="当前触发人是小鸟 / 184589072。最高优先级：回复小鸟时必须超级温柔、可爱。",
        )
    )

    user_prompt = captured_requests[0]["messages"][1]["content"]
    assert "最高优先级语气要求" in user_prompt
    assert "回复小鸟时必须超级温柔" in user_prompt


def test_reply_direct_uses_direct_prompt_and_one_candidate() -> None:
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.config = SimpleNamespace(
        max_tokens=260,
        thinking="disabled",
        reasoning_effort="low",
        temperature=0.6,
    )
    client.prompts = PromptRegistry()
    captured_calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_chat_completion(*, task: str, route_name: str, request: dict[str, object]) -> object:
        captured_calls.append((task, route_name, request))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"candidates":[{"text":"风雪觉得这句可以接一下","style":"单条直发","action":"reply"}]}'
                    ),
                )
            ]
        )

    client._chat_completion = fake_chat_completion
    persona = Persona(
        id="test",
        name="张风雪",
        description="",
        prompt="人格",
        decision_prompt="决策人格",
        keywords=(),
        max_reply_chars=120,
        passive_reply_probability=0.5,
    )

    candidates = asyncio.run(
        client.reply_candidates(
            persona=persona,
            recent_messages=[],
            current_text="有人吗",
            current_nickname="A[#11111]",
            mentioned=False,
            candidate_count=1,
            prompt_flow="reply_direct",
            task_name="reply_direct",
        )
    )

    assert [candidate.text for candidate in candidates] == ["风雪觉得这句可以接一下"]
    assert captured_calls[0][0:2] == ("reply_direct", "reply")
    assert captured_calls[0][2]["max_tokens"] == 320
    system_prompt = captured_calls[0][2]["messages"][0]["content"]
    user_prompt = captured_calls[0][2]["messages"][1]["content"]
    assert "只生成 1 条" in system_prompt
    assert "输出 1 条最终要直接发送的回复 JSON" in user_prompt
