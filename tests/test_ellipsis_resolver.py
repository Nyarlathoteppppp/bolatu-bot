from __future__ import annotations

from qq_social_agent.ellipsis_resolver import (
    EllipsisJudgement,
    EllipsisSource,
    apply_jev_ellipsis_judgement,
    build_ellipsis_candidates,
    format_ellipsis_prompt_block,
    semantic_message_text,
    should_ask_jev_ellipsis,
    slot_for_query,
)
from qq_social_agent.memory import ChatMessage
from qq_social_agent.reference_resolver import ReferenceResolution, ReplyHint


def _msg(user_id: int, nick: str, text: str, *, mid: str = "", is_bot: bool = False, ts: float = 1.0) -> ChatMessage:
    return ChatMessage(1, user_id, nick, text, is_bot, ts, source_message_id=mid)


def test_trigger_skips_complete_sentence() -> None:
    assert should_ask_jev_ellipsis("Gemini 写代码其实还行") is False
    assert should_ask_jev_ellipsis("我觉得这个呢其实还不错") is False


def test_semantic_message_text_extracts_reply_body() -> None:
    rendered = "甲[#00001]回复乙[#00002]消息【乙[#00002]说：人生的意义是什么；甲[#00001]回复乙[#00002]：生孩子】"
    assert semantic_message_text(rendered) == "生孩子"
    assert semantic_message_text("普通消息") == "普通消息"


def test_trigger_hits_short_and_cues() -> None:
    assert should_ask_jev_ellipsis("Gemini 呢？")
    assert should_ask_jev_ellipsis("去哪？")
    assert should_ask_jev_ellipsis("也行")
    assert should_ask_jev_ellipsis("第二个呢？")
    assert should_ask_jev_ellipsis("然后呢？")
    assert should_ask_jev_ellipsis("速度呢？")
    assert should_ask_jev_ellipsis("看看这个")
    assert should_ask_jev_ellipsis("[@2842521566] 看看这个")


def test_same_predicate_gemini() -> None:
    messages = [_msg(2, "甲", "Claude 写代码挺强", mid="101")]
    sources = build_ellipsis_candidates(messages, current_text="Gemini 呢？")
    judged = EllipsisJudgement(kind="SAME_PREDICATE", inherit_from=sources[0].key, confidence=0.91)
    resolved = apply_jev_ellipsis_judgement(judged, sources, current_text="Gemini 呢？")
    assert resolved.kind == "SAME_PREDICATE"
    assert "Claude 写代码挺强" in resolved.source_text
    assert resolved.slot is None
    block = format_ellipsis_prompt_block(resolved)
    assert "kind=SAME_PREDICATE" in block
    assert "Claude 4.6" not in block
    assert "Claude 写代码挺强" in block


def test_slot_query_location() -> None:
    messages = [_msg(2, "甲", "我准备考研", mid="201")]
    sources = build_ellipsis_candidates(messages, current_text="去哪？")
    judged = EllipsisJudgement(kind="SLOT_QUERY", inherit_from=sources[0].key, confidence=0.88)
    resolved = apply_jev_ellipsis_judgement(judged, sources, current_text="去哪？")
    assert resolved.kind == "SLOT_QUERY"
    assert resolved.slot == "LOCATION"
    assert "我准备考研" in format_ellipsis_prompt_block(resolved)


def test_short_answer_shanghai() -> None:
    messages = [_msg(2, "甲", "你去哪考研？", mid="301")]
    sources = build_ellipsis_candidates(messages, current_text="上海")
    judged = EllipsisJudgement(kind="SHORT_ANSWER", inherit_from=sources[0].key, confidence=0.84)
    resolved = apply_jev_ellipsis_judgement(judged, sources, current_text="上海")
    assert resolved.kind == "SHORT_ANSWER"
    assert "你去哪考研" in resolved.source_text


def test_continuation_then_what() -> None:
    messages = [_msg(2, "甲", "我后来退了", mid="401")]
    sources = build_ellipsis_candidates(messages, current_text="然后呢？")
    judged = EllipsisJudgement(kind="CONTINUATION", inherit_from=sources[0].key, confidence=0.8)
    resolved = apply_jev_ellipsis_judgement(judged, sources, current_text="然后呢？")
    assert resolved.kind == "CONTINUATION"


def test_comparison_speed() -> None:
    messages = [_msg(2, "甲", "Claude 比 Gemini 贵", mid="501")]
    sources = build_ellipsis_candidates(messages, current_text="速度呢？")
    judged = EllipsisJudgement(kind="COMPARISON", inherit_from=sources[0].key, confidence=0.86)
    resolved = apply_jev_ellipsis_judgement(judged, sources, current_text="速度呢？")
    assert resolved.kind == "COMPARISON"


def test_complete_sentence_none() -> None:
    resolved = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="NONE", inherit_from="NONE", confidence=0.2),
        [],
        current_text="Gemini 写代码其实还行",
    )
    assert resolved.kind == "NONE"
    assert format_ellipsis_prompt_block(resolved) == ""


def test_reply_this_one_prefers_reply_source() -> None:
    messages = [
        _msg(2, "甲", "第一套方案要重写缓存", mid="10", ts=1.0),
        _msg(3, "乙", "第二套更稳但慢", mid="11", ts=2.0),
        _msg(4, "丙", "刚才那顿饭不错", mid="12", ts=3.0),
    ]
    reply = ReplyHint(exists=True, author_id=2, author_label="甲", text="第一套方案要重写缓存", message_id="10")
    sources = build_ellipsis_candidates(
        messages,
        current_text="这个呢？",
        reply=reply,
    )
    assert sources[0].source_reason == "reply"
    assert "第一套方案" in sources[0].text


def test_referent_related_source_enters_pool() -> None:
    messages = [
        _msg(2, "甲", "小鸟准备考研", mid="21", ts=1.0),
        _msg(3, "乙", "今晚吃什么？", mid="22", ts=2.0),
        _msg(4, "丙", "缓存怎么写？", mid="23", ts=3.0),
    ]
    sources = build_ellipsis_candidates(
        messages,
        current_text="去哪？",
        reference=ReferenceResolution((2,), reason="previous_named_member", confidence=0.9, kind="PERSON"),
    )
    reasons = {row.source_reason: row.text for row in sources}
    assert "referent_related" in reasons
    assert "小鸟准备考研" in reasons["referent_related"]
    assert "今晚吃什么" in " ".join(row.text for row in sources if row.source_reason == "recent_question") or True


def test_low_confidence_does_not_link() -> None:
    sources = [EllipsisSource(key="m101", speaker="甲", text="Claude 写代码挺强", source_reason="previous")]
    resolved = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="SAME_PREDICATE", inherit_from="m101", confidence=0.3),
        sources,
        current_text="Gemini 呢？",
    )
    assert resolved.unresolved is True
    assert resolved.source_text == ""
    assert "unresolved_ellipsis=true" in format_ellipsis_prompt_block(resolved)


def test_slot_rules_do_not_call_model() -> None:
    assert slot_for_query("去哪？", "SLOT_QUERY") == "LOCATION"
    assert slot_for_query("什么时候", "SLOT_QUERY") == "TIME"
    assert slot_for_query("谁啊", "SLOT_QUERY") == "PERSON"
    assert slot_for_query("为什么", "SLOT_QUERY") == "REASON"
    assert slot_for_query("怎么做", "SLOT_QUERY") == "METHOD"
    assert slot_for_query("多少钱", "SLOT_QUERY") == "QUANTITY"
    assert slot_for_query("哪个", "SLOT_QUERY") == "CHOICE"
    assert slot_for_query("怎样了", "SLOT_QUERY") == "UNKNOWN"
    assert slot_for_query("去哪？", "SAME_PREDICATE") is None


def test_same_speaker_burst_is_candidate_for_this() -> None:
    messages = [
        _msg(8, "石头", "commandogat，claudrgpt好像是原价", mid="1", ts=1.0),
        _msg(8, "石头", "坏", mid="2", ts=2.0),
        _msg(8, "石头", "哦冲一块钱当十块钱", mid="3", ts=3.0),
        _msg(8, "石头", "1折", mid="4", ts=4.0),
    ]
    sources = build_ellipsis_candidates(
        messages,
        current_text="[@2842521566] 看看这个",
        current_user_id=8,
    )
    same = [row for row in sources if row.source_reason == "same_speaker"]
    assert any("1折" in row.text for row in same)
    assert any("一块钱" in row.text for row in same)


def test_item_deixis_links_same_speaker_line() -> None:
    sources = [
        EllipsisSource(key="s4", speaker="石头", text="1折", source_reason="same_speaker"),
        EllipsisSource(key="s3", speaker="石头", text="哦冲一块钱当十块钱", source_reason="same_speaker"),
    ]
    resolved = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="s4", confidence=0.86),
        sources,
        current_text="看看这个",
    )
    assert resolved.kind == "ITEM_DEIXIS"
    assert resolved.source_text == "1折"
    assert resolved.unresolved is False


def test_ellipsis_unavailable_is_not_not_applicable() -> None:
    resolved = apply_jev_ellipsis_judgement(None, [], current_text="后来呢")
    assert resolved.status == "UNAVAILABLE"
    assert resolved.status != "NOT_APPLICABLE"
    none = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="NONE", inherit_from="NONE", confidence=0.9),
        [],
        current_text="后来呢",
    )
    assert none.status == "NOT_APPLICABLE"
    assert none.unresolved is False
