from __future__ import annotations

import asyncio
from unittest.mock import patch
import os
import httpx
from qq_social_agent.jev_client import JevClient, set_jev_telemetry_recorder
from qq_social_agent.memory import ChatMessage
from qq_social_agent.persona import Persona


def _persona() -> Persona:
    return Persona(
        id="zhangxuefeng",
        name="张风雪",
        description="",
        prompt="prompt",
        decision_prompt="决策摘要",
        max_reply_chars=220,
        passive_reply_probability=0.95,
    )


def _timing_answers(
    *,
    wants: float = 0.1,
    care: float = 0.1,
    other: float = 0.1,
    silent: float = 0.8,
    answer: float = 0.1,
    social: float = 0.1,
) -> dict:
    return {"answers": {
        "wants_answer": {"noul": wants},
        "needs_care": {"noul": care},
        "to_other": {"noul": other},
        "timing_route": {"type": "choice", "choice": "silent", "probabilities": {
            "silent": silent, "answer": answer, "care": 0.0,
            "continue_bot": 0.0, "social_join": social, "other": 0.0,
        }},
    }}


def test_jev_route_tool_returns_none_below_threshold() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "need_tool": {"noul": 0.2},
                "tool_choice": {"choice": "probability", "confidence": 0.9},
            }
        }

    client.evaluate = fake_evaluate
    routed = asyncio.run(
        client.route_tool(
            persona=_persona(),
            recent_messages=[],
            current_text="今天吃什么",
            current_nickname="群友",
            addressed=True,
        )
    )
    assert routed.tool == "none"


def test_jev_route_tool_keeps_probability_choice() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "need_tool": {"noul": 0.81},
                "tool_choice": {"choice": "probability", "confidence": 0.9},
            }
        }

    client.evaluate = fake_evaluate
    routed = asyncio.run(
        client.route_tool(
            persona=_persona(),
            recent_messages=[],
            current_text="拿到 offer 概率多大",
            current_nickname="群友",
            addressed=True,
        )
    )
    assert routed.tool == "probability"
    assert routed.query == "拿到 offer 概率多大"


def test_jev_should_reply_imports_decision_on_silent_path() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "should_reply": {"noul": 0.05},
                "action": {"choice": "ignore"},
            }
        }

    client.evaluate = fake_evaluate
    decision = asyncio.run(
        client.should_reply(
            persona=_persona(),
            recent_messages=[ChatMessage(1, 2, "A", "哈哈", False, 1.0)],
            current_text="哈哈",
            current_nickname="A",
        )
    )
    assert decision.should_reply is False
    assert decision.action == "ignore"



def test_jev_timing_gate_concrete_statement_does_not_force_reply() -> None:
    client = JevClient(api_key="test-key")
    called = {"n": 0}

    async def fake_evaluate(**kwargs):
        called["n"] += 1
        return _timing_answers()

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(), recent_messages=[],
        current_text="现在感觉 AI 有点特指 Transformer 了", current_nickname="A",
    ))
    assert called["n"] == 1
    assert timing.channel.value == "silent"


def test_jev_timing_gate_text_answer() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        assert set(kwargs["questions"]) == {"wants_answer", "needs_care", "to_other", "timing_route"}
        assert kwargs["state"]["current"]["text"] == "CMU 难申吗"
        assert kwargs["state"]["bot"] == "张风雪"
        return _timing_answers(wants=0.85, silent=0.2, answer=0.65)

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(), recent_messages=[],
        current_text="CMU 难申吗", current_nickname="A",
    ))
    assert timing.channel.value == "text"
    assert timing.intent.value == "answer"


def test_jev_timing_gate_directed_other_is_silent() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return _timing_answers(wants=0.92, other=0.89, silent=0.1, answer=0.8)

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(), recent_messages=[],
        current_text="[@12345] 有没有讲集体化的文章", current_nickname="A",
    ))
    assert timing.channel.value == "silent"
    assert timing.reason.startswith("jev_to_other")


def test_jev_timing_gate_distress_uses_care_intent() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return _timing_answers(care=0.63, silent=0.05, answer=0.0, social=0.0)

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(), recent_messages=[],
        current_text="我害怕明天", current_nickname="A",
    ))
    assert timing.channel.value == "text"
    assert timing.intent.value == "care"


def test_jev_timing_gate_social_opening_keeps_chat_action_available() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return _timing_answers(silent=0.12, answer=0.01, social=0.82)

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(), recent_messages=[],
        current_text="想到一个天才提示词，一脚把 agent 踩冒烟了", current_nickname="A",
    ))
    assert timing.channel.value == "text"
    assert timing.intent.value == "chat"
    assert timing.to_reply_decision().action == "reply"


def test_jev_timing_gate_peer_invitation_does_not_trigger_on_question_mark_alone() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        assert kwargs["state"]["recent"][0]["speaker"] == "风雪"
        return _timing_answers(wants=0.7, silent=0.32, answer=0.05, social=0.62)

    client.evaluate = fake_evaluate
    timing = asyncio.run(client.timing_gate(
        persona=_persona(),
        recent_messages=[ChatMessage(1, 2, "风雪", "第三次就敢玩", True, 1.0)],
        current_text="一起吗", current_nickname="A",
    ))
    assert timing.channel.value == "silent"


def test_jev_timing_gate_blocks_short_flame_even_after_bot() -> None:
    client = JevClient(api_key="test-key")
    called = {"n": 0}

    async def fake_evaluate(**kwargs):
        called["n"] += 1
        return {
            "answers": {
                "following_bot": {"noul": 0.70},
                "has_concrete_content": {"noul": 0.70},
            }
        }

    client.evaluate = fake_evaluate
    timing = asyncio.run(
        client.timing_gate(
            persona=_persona(),
            recent_messages=[
                ChatMessage(1, 2, "风雪", "Gemini 地区限制", True, 1.0),
            ],
            current_text="是不是死妈",
            current_nickname="A",
        )
    )
    assert called["n"] == 0
    assert timing.channel.value == "silent"
    assert timing.reason == "code_silent_ack"


def test_jev_audit_rejects_near_duplicate() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"send": {"noul": 0.2}}}

    client.evaluate = fake_evaluate
    send, reason = asyncio.run(
        client.audit_proactive_reply(
            persona=_persona(),
            recent_messages=[ChatMessage(1, 2, "风雪", "刚才那句我说过了", True, 1.0)],
            candidate="刚才那句我说过了啦",
        )
    )
    assert send is False
    assert "复读" in reason


def test_jev_select_meme_picks_candidate_id() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "send": {"noul": 0.9},
                "meme_id": {"choice": "12"},
            }
        }

    client.evaluate = fake_evaluate
    choice = asyncio.run(
        client.select_meme(
            current_text="嘿嘿",
            reply_text="你又来了",
            candidates="- ID 12: 害羞捂脸；标签：害羞\n- ID 13: 无语；标签：无语",
        )
    )
    assert choice.send is True
    assert choice.meme_id == 12


def test_jev_search_useful_and_followup() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(*, questions, **kwargs):
        key = next(iter(questions))
        if key == "useful":
            return {"answers": {"useful": {"noul": 0.2}}}
        return {"answers": {"need_another_round": {"noul": 0.9}}}

    client.evaluate = fake_evaluate
    useful, _ = asyncio.run(client.judge_search_useful(query="OpenFOAM 是什么", evidence="今日股市大涨"))
    assert useful is False
    hop = asyncio.run(
        client.should_followup_search(
            query="OpenFOAM 是什么",
            kind="web",
            evidence="今日股市大涨",
            current_round=1,
            remaining_seconds=4.0,
        )
    )
    assert hop is True



def test_jev_search_keeps_evidence_when_unsure() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"useful": {"noul": 0.40}}}

    client.evaluate = fake_evaluate
    useful, _ = asyncio.run(
        client.judge_search_useful(query="OpenFOAM 是什么", evidence="OpenFOAM 是开源 CFD 工具箱")
    )
    assert useful is True


def test_jev_private_continue_skips_when_closed() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"continue": {"noul": 0.2}}}

    client.evaluate = fake_evaluate
    send, reason = asyncio.run(
        client.should_continue_private_chat(
            persona=_persona(),
            recent_messages=[ChatMessage(1, 2, "A", "晚安", False, 1.0)],
        )
    )
    assert send is False
    assert "jev_private_continue" in reason


def test_jev_select_jargon_terms_filters_unrelated() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "need_jargon": {"noul": 0.8},
                "term_0": {"noul": 0.9},
                "term_1": {"noul": 0.1},
            }
        }

    client.evaluate = fake_evaluate
    terms = asyncio.run(
        client.select_jargon_terms(
            current_text="柏拉图今天好安静",
            heuristic_terms=("柏拉图", "zbzy"),
            jargon_catalog="- 柏拉图：群名",
        )
    )
    assert terms == ("柏拉图",)



def test_jev_rank_context_messages_returns_scores() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {
            "answers": {
                "m0": {"noul": 0.1},
                "m1": {"noul": 0.9},
            }
        }

    client.evaluate = fake_evaluate
    scores = asyncio.run(
        client.rank_context_messages(
            current_nickname="A",
            current_text="刚才说的那个学校",
            candidates=[
                ChatMessage(1, 1, "B", "晚饭吃啥", False, 1.0),
                ChatMessage(1, 2, "A", "我在申 CMU", False, 2.0),
            ],
        )
    )
    assert scores == [0.1, 0.9]



def test_jev_resolve_referent_picks_candidate() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        assert "reference_kind" in kwargs["questions"]
        assert "person_target" in kwargs["questions"]
        assert "person_confidence" not in kwargs["questions"]
        return {
            "answers": {
                "reference_kind": {"choice": "PERSON", "confidence": 0.93},
                "person_target": {"choice": "u7", "confidence": 0.81},
            }
        }

    client.evaluate = fake_evaluate
    judged = asyncio.run(
        client.resolve_referent(
            current_text="她现在还考吗",
            current_label="甲[#00007]",
            candidates=[
                ("u7", "小鸟[#89072]", "小鸟以前准备考研"),
                ("u8", "乙[#00008]", "今天吃什么"),
            ],
        )
    )
    assert judged.kind == "PERSON"
    assert judged.person_key == "u7"
    assert judged.confidence == 0.81


def test_jev_ask_back_false_below_threshold() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"ask_back": {"noul": 0.2}}}

    client.evaluate = fake_evaluate
    assert asyncio.run(client.should_ask_back(current_text="哈哈", action="reply", addressed=False)) is False


def test_jev_ask_back_does_not_override_addressed_answer_when_unsure() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        instructions = kwargs["questions"]["ask_back"]["instructions"]
        assert "点名提问" in instructions
        return {"answers": {"ask_back": {"noul": 0.89}}}

    client.evaluate = fake_evaluate
    assert (
        asyncio.run(
            client.should_ask_back(
                current_text="你打算申请哪",
                action="answer",
                addressed=True,
            )
        )
        is False
    )


def test_jev_media_gate_addressed_is_easier() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"worth_reading": {"noul": 0.5}}}

    client.evaluate = fake_evaluate
    addressed = asyncio.run(
        client.should_read_media(kind="ocr", caption="这图写了啥", addressed=True, item_count=1)
    )
    silent = asyncio.run(
        client.should_read_media(kind="ocr", caption="", addressed=False, item_count=1)
    )
    assert addressed is True
    assert silent is False



def test_jev_speaking_action_none_keeps_baseline() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        criteria = kwargs["questions"]["speaking_action"]["criteria"]
        assert "none" in criteria
        assert "protect" in criteria
        assert "mirror_style" in criteria
        assert len(criteria) == 26
        return {"answers": {"speaking_action": {"choice": "none"}}}

    client.evaluate = fake_evaluate
    choice, reason = asyncio.run(
        client.select_speaking_action(
            current_text="哈哈",
            current_label="甲",
            addressed=False,
            baseline_action="reply",
        )
    )
    assert choice == "none"
    assert "jev_speak_none" in reason


def test_jev_speaking_action_unknown_falls_to_none() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"speaking_action": {"choice": "unhinged"}}}

    client.evaluate = fake_evaluate
    choice, _reason = asyncio.run(
        client.select_speaking_action(
            current_text="草",
            current_label="甲",
            addressed=False,
            baseline_action="reply",
        )
    )
    assert choice == "none"


def test_jev_resolve_ellipsis_same_predicate() -> None:
    from qq_social_agent.ellipsis_resolver import EllipsisSource

    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        assert "ellipsis_kind" in kwargs["questions"]
        assert "inherit_from" in kwargs["questions"]
        assert "ellipsis_confidence" not in kwargs["questions"]
        return {
            "answers": {
                "ellipsis_kind": {"choice": "SAME_PREDICATE", "confidence": 0.95},
                "inherit_from": {"choice": "m101", "confidence": 0.91},
            }
        }

    client.evaluate = fake_evaluate
    judged = asyncio.run(
        client.resolve_ellipsis(
            current_text="Gemini 呢？",
            current_label="甲[#00007]",
            sources=[
                EllipsisSource(key="m101", speaker="甲", text="Claude 写代码挺强", source_reason="previous", message_id="101"),
            ],
        )
    )
    assert judged.kind == "SAME_PREDICATE"
    assert judged.inherit_from == "m101"
    assert judged.confidence == 0.91


def test_jev_discourse_first_pass_batches_independent_choices() -> None:
    from qq_social_agent.discourse_state import AddresseeCandidate
    from qq_social_agent.ellipsis_resolver import EllipsisSource
    from qq_social_agent.reference_resolver import ReferentCandidate, ReplyHint

    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        assert isinstance(kwargs["state"], dict)
        assert kwargs["state"]["message"]["current_text"] == "[@8] 她这个呢"
        assert set(kwargs["questions"]) == {
            "addressee_target",
            "reference_kind",
            "person_target",
            "ellipsis_kind",
            "inherit_from",
        }
        inherit_instructions = kwargs["questions"]["inherit_from"]["instructions"]
        assert "语义上被回答的最近消息" in inherit_instructions
        assert "same_speaker 只是来源证据" in inherit_instructions
        return {
            "answers": {
                "addressee_target": {"choice": "c_mention", "confidence": 0.94},
                "reference_kind": {"choice": "PERSON", "confidence": 0.85},
                "person_target": {"choice": "u7", "confidence": 0.78},
                "ellipsis_kind": {"choice": "ITEM_DEIXIS", "confidence": 0.88},
                "inherit_from": {"choice": "m1", "confidence": 0.83},
            }
        }

    client.evaluate = fake_evaluate
    result = asyncio.run(
        client.resolve_discourse_first_pass(
            current_text="[@8] 她这个呢",
            current_label="甲",
            addressee_candidates=[
                AddresseeCandidate("c_mention", 8, "乙", "at"),
                AddresseeCandidate("generic", None, "群里泛说", "generic"),
                AddresseeCandidate("other", None, "无法判断", "other"),
            ],
            referent_candidates=[ReferentCandidate("u7", 7, "小鸟")],
            ellipsis_sources=[EllipsisSource("m1", "甲", "方案A", "same_speaker")],
            reply=ReplyHint(),
            at_user_ids=(8,),
        )
    )
    assert result["addressee"].confidence == 0.94
    assert result["referent"].confidence == 0.78
    assert result["ellipsis"].confidence == 0.83


def test_jev_resolve_repair_and_ambiguity() -> None:
    from qq_social_agent.discourse_effects import RepairTarget

    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        names = set(kwargs["questions"])
        if "repair_kind" in names:
            assert "repair_target" in names
            return {
                "answers": {
                    "repair_kind": {"choice": "REFERENT", "confidence": 0.92},
                    "repair_target": {"choice": "t_referent", "confidence": 0.88},
                }
            }
        if "memory_action" in names:
            return {
                "answers": {
                    "memory_action": {"choice": "REFINE", "confidence": 0.8},
                    "memory_target": {"choice": "mem1", "confidence": 0.9},
                }
            }
        return {
            "answers": {
                "ambiguity_kind": {"choice": "PERSON", "confidence": 0.81},
            }
        }

    client.evaluate = fake_evaluate
    repair = asyncio.run(
        client.resolve_repair(
            current_text="不是小鸟，是小王",
            current_label="甲",
            targets=[RepairTarget(key="t_referent", kind="referent", summary="小鸟")],
        )
    )
    assert repair.kind == "REFERENT"
    memory_effect = asyncio.run(
        client.resolve_memory_effect(
            current_text="小鸟准备考计算机研究生",
            candidate=type("C", (), {"content": "小鸟准备考计算机研究生", "subject_user_id": 1, "source_message_id": ""})(),
            related=[type("A", (), {"id": 1, "content": "小鸟准备考研", "source": "t", "updated_at": 1})()],
        )
    )
    assert memory_effect.action == "REFINE"
    amb = asyncio.run(client.resolve_ambiguity(current_text="她考哪里？", candidate_labels=["小鸟", "小王"]))
    assert amb.kind == "PERSON"


def test_jev_critique_draft_uses_confidence_gated_choices() -> None:
    from qq_social_agent.reference_resolver import ReferenceResolution

    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        questions = kwargs["questions"]
        assert list(questions) == [
            "intent_covered",
            "referent_consistent",
            "context_consistent",
            "unsupported_claim",
        ]
        for question in questions.values():
            assert question["type"] == "choice"
            assert "other" in question["criteria"]
            assert "not_applicable" in question["criteria"]
        state = kwargs["state"]
        assert state.startswith("【待发送草稿】")
        assert "【当前消息】" in state
        assert state.rfind("【约束】") > state.find("【待发送草稿】")
        return {
            "answers": {
                "intent_covered": {"choice": "covered", "probabilities": {"missed": 0.10}},
                "referent_consistent": {"choice": "conflict", "probabilities": {"conflict": 0.90}},
                "context_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.10}},
                "unsupported_claim": {"choice": "supported", "probabilities": {"unsupported": 0.10}},
            }
        }

    client.evaluate = fake_evaluate
    judged = asyncio.run(
        client.critique_draft(
            draft="小鸟去了MIT",
            current_text="不是小鸟，是小王",
            action="answer",
            reference=ReferenceResolution((2001,), reason="repair_referent", kind="PERSON", status="RESOLVED"),
        )
    )
    assert judged.referent_consistent == "NO"
    assert judged.unsupported_claim == "NO"


def test_jev_prefers_official_api_when_typesafe_key_exists() -> None:
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "direct", "OPENROUTER_API_KEY": "router"}):
        client = JevClient()
    assert client.provider == "typesafe"
    assert client.base_url == "https://api.typesafe.ai/v1/systemone"
    assert client.model == "jev-1.13.0"
    assert client.api_key == "direct"


def test_jev_telemetry_records_full_choice_distribution() -> None:
    events = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer direct"
        assert "http-referer" not in request.headers
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"target": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.7,
                "probabilities": {"a": 0.7, "b": 0.2, "other": 0.1},
            }},
            "usage": {"input_tokens": 10, "output_tokens": 4},
        })

    async def run() -> None:
        client = JevClient(api_key="direct", base_url="https://api.typesafe.ai/v1/systemone")
        client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        set_jev_telemetry_recorder(events.append)
        try:
            await client.evaluate(
                state={"message": "x"},
                questions={"target": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B", "other": "Other"}}},
            )
        finally:
            set_jev_telemetry_recorder(None)
            await client.aclose()

    asyncio.run(run())
    answer = events[0]["answers"]["target"]
    assert answer["probabilities"] == {"a": 0.7, "b": 0.2, "other": 0.1}
    assert answer["top1"] == {"choice": "a", "probability": 0.7}
    assert answer["top2"] == {"choice": "b", "probability": 0.2}
    assert answer["margin"] == 0.5
    assert answer["escape_probability"] == 0.1


def test_jev_audit_allows_overlapping_answer() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        instructions = kwargs["questions"]["send"]["instructions"]
        assert "不确定是不是复读时给高分" in instructions
        assert "【是否点名/回复风雪】是" in kwargs["state"]
        return {"answers": {"send": {"noul": 0.49}}}

    client.evaluate = fake_evaluate
    send, reason = asyncio.run(
        client.audit_proactive_reply(
            persona=_persona(),
            recent_messages=[ChatMessage(1, 2, "风雪", "说到留学，我现在最焦虑的居然是租房", True, 1.0)],
            candidate="CMU 这边我还在看 CS，租房确实得提前抢。",
            chat_label="QQ 群聊",
            addressed=True,
            current_text="你打算申请哪",
        )
    )
    assert send is True
    assert "未复读" in reason


def test_jev_audit_blocks_only_when_sure_it_is_repeat() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"send": {"noul": 0.12}}}

    client.evaluate = fake_evaluate
    send, reason = asyncio.run(
        client.audit_proactive_reply(
            persona=_persona(),
            recent_messages=[ChatMessage(1, 2, "风雪", "说到留学，我现在最焦虑的居然是租房", True, 1.0)],
            candidate="说到留学，我现在最焦虑的居然是租房。",
            chat_label="QQ 群聊",
            addressed=False,
        )
    )
    assert send is False
    assert "复读" in reason


def test_jev_choose_proactive_topic_picks_from_bucket() -> None:
    client = JevClient(api_key="test-key")
    topics = [
        "游戏：最近玩过的游戏、单机和联机的乐趣",
        "游戏：氪金、抽卡、账号价格和时间成本",
        "游戏：剧情、角色、操作手感和最烦的机制",
    ]

    async def fake_evaluate(**kwargs):
        questions = kwargs["questions"]
        assert list(questions) == ["topic"]
        criteria = questions["topic"]["criteria"]
        assert criteria["t2"] == topics[1]
        assert criteria["OTHER"] == "这一块里没有适合这次开口的方向"
        assert kwargs["state"].startswith("【聊天场景】")
        assert "【已抽到的话题块】游戏" in kwargs["state"]
        assert kwargs["state"].rfind("【约束】") > kwargs["state"].find("【已抽到的话题块】")
        return {"answers": {"topic": {"choice": "t2"}}}

    client.evaluate = fake_evaluate
    chosen = asyncio.run(
        client.choose_proactive_topic(
            bucket="游戏",
            topics=topics,
            recent_messages=[ChatMessage(1, 2, "甲", "今晚开一把吗", False, 1.0)],
            chat_label="QQ 群聊",
        )
    )
    assert chosen == topics[1]


def test_jev_choose_proactive_topic_other_returns_none() -> None:
    client = JevClient(api_key="test-key")

    async def fake_evaluate(**kwargs):
        return {"answers": {"topic": {"choice": "OTHER"}}}

    client.evaluate = fake_evaluate
    chosen = asyncio.run(
        client.choose_proactive_topic(
            bucket="历史",
            topics=["历史：帝国兴衰、近代史、人物评价和历史假设"],
        )
    )
    assert chosen is None
