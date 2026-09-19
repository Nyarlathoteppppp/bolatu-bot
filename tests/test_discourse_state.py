from __future__ import annotations

import asyncio
from types import SimpleNamespace

from qq_social_agent.discourse_effects import (
    AmbiguityJudgement,
    RepairJudgement,
    RepairTarget,
    apply_jev_repair_judgement,
    apply_repair_invalidation,
)
from qq_social_agent.discourse_state import (
    AddresseeCandidate,
    AddresseeJudgement,
    DeixisCandidate,
    DiscourseAuditJudgement,
    apply_jev_addressee_judgement,
    apply_jev_discourse_audit,
    assemble_discourse_state,
    build_addressee_candidates,
    build_deixis_candidates,
    classify_ambiguity_effect,
    discourse_decision_trace,
    draft_violates_media_gate,
    format_discourse_prompt_block,
    logical_turns,
    message_has_real_media,
    resolve_group_discourse,
    should_audit_discourse_state,
    targeted_requery_fields,
)
from qq_social_agent.ellipsis_resolver import (
    EllipsisJudgement,
    apply_jev_ellipsis_judgement,
    should_ask_jev_ellipsis,
)
from qq_social_agent.memory import ChatMessage
from qq_social_agent.pre_send_critic import format_critic_jev_state
from qq_social_agent.reference_resolver import ReferenceResolution, ReplyHint
from qq_social_agent.resolver_result import AMBIGUOUS, NONE, NOT_APPLICABLE, RESOLVED, UNAVAILABLE


P1 = 1001
P2 = 2002
A = 3003
B = 4004


def _msg(
    user_id: int,
    nick: str,
    text: str,
    *,
    mid: str = "",
    ts: float = 1.0,
    is_bot: bool = False,
    segments: str = "",
) -> ChatMessage:
    return ChatMessage(
        1,
        user_id,
        nick,
        text,
        is_bot,
        ts,
        source_message_id=mid,
        message_segments_json=segments,
    )


def _label(user_id: int, nick: str) -> str:
    return f"{nick}[#{str(user_id)[-5:]}]"


class ScriptedJev:
    def __init__(self, **answers):
        self.answers = answers
        self.calls: list[str] = []

    async def resolve_referent(self, **kwargs):
        self.calls.append("referent")
        return self.answers.get("referent")

    async def resolve_addressee(self, **kwargs):
        self.calls.append("addressee")
        return self.answers.get("addressee")

    async def resolve_ellipsis(self, **kwargs):
        self.calls.append("ellipsis")
        return self.answers.get("ellipsis")

    async def resolve_repair(self, **kwargs):
        self.calls.append("repair")
        return self.answers.get("repair")

    async def resolve_ambiguity(self, **kwargs):
        self.calls.append("ambiguity")
        return self.answers.get("ambiguity")

    async def audit_discourse_state(self, **kwargs):
        self.calls.append("audit")
        return self.answers.get("audit")


def test_logical_turns_group_same_speaker_burst_without_mutating_messages() -> None:
    messages = [
        _msg(P1, "P1", "哦冲一块钱当十块钱", mid="1", ts=10.0),
        _msg(P1, "P1", "1折", mid="2", ts=11.0),
        _msg(P1, "P1", "[@2002] 看看这个", mid="3", ts=12.0),
    ]
    original = tuple((m.source_message_id, m.text) for m in messages)
    turns = logical_turns(messages, max_gap_seconds=90)
    assert len(turns) == 1
    assert [m.text for m in turns[0]] == ["哦冲一块钱当十块钱", "1折", "[@2002] 看看这个"]
    assert tuple((m.source_message_id, m.text) for m in messages) == original


def test_logical_turns_break_on_other_speaker_or_time_gap() -> None:
    messages = [
        _msg(P1, "P1", "1折", mid="1", ts=10.0),
        _msg(P2, "P2", "啥", mid="2", ts=11.0),
        _msg(P1, "P1", "看看这个", mid="3", ts=12.0),
        _msg(P1, "P1", "更早那条", mid="4", ts=200.0),
    ]
    turns = logical_turns(messages, max_gap_seconds=90)
    assert [tuple(m.text for m in turn) for turn in turns] == [
        ("1折",),
        ("啥",),
        ("看看这个",),
        ("更早那条",),
    ]


def test_deixis_candidates_prefer_same_speaker_burst_then_reply_mention_topic_media_other() -> None:
    messages = [
        _msg(P1, "P1", "哦冲一块钱当十块钱", mid="1", ts=10.0),
        _msg(P1, "P1", "1折", mid="2", ts=11.0),
    ]
    rows = build_deixis_candidates(
        messages,
        current_text="[@2002] 看看这个",
        current_user_id=P1,
        at_user_ids=(P2,),
        reply=ReplyHint(),
        media_present=False,
    )
    reasons = [row.source_reason for row in rows]
    assert "same_speaker" in reasons
    assert any(row.text == "1折" for row in rows if row.source_reason == "same_speaker")
    assert any("一块钱" in row.text for row in rows if row.source_reason == "same_speaker")
    assert "other" in {row.key for row in rows}


def test_scenario_look_at_this_discount_addressee_and_deixis() -> None:
    recent = [
        _msg(P1, "P1", "哦冲一块钱当十块钱", mid="1", ts=10.0),
        _msg(P1, "P1", "1折", mid="2", ts=11.0),
    ]
    addressee_rows = build_addressee_candidates(
        current_user_id=P1,
        current_label=_label(P1, "P1"),
        current_text="[@2002] 看看这个",
        reply=ReplyHint(),
        at_user_ids=(P2,),
        mention_labels=((_label(P2, "P2"), P2),),
    )
    addressee = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_mention", confidence=0.92, reason="jev_mention"),
        addressee_rows,
    )
    deixis_rows = build_deixis_candidates(
        recent,
        current_text="[@2002] 看看这个",
        current_user_id=P1,
        at_user_ids=(P2,),
        media_present=False,
    )
    inherit = next(row.key for row in deixis_rows if row.text == "1折")
    deixis = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from=inherit, confidence=0.9),
        deixis_rows,
        current_text="看看这个",
    )
    state = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="[@2002] 看看这个",
        addressee=addressee,
        deixis=deixis,
        media_present=False,
        ambiguity_effect="NON_BLOCKING",
        state_audit="PASS",
    )
    assert state.addressee.status == RESOLVED
    assert state.addressee.target_id == P2
    assert state.deixis.status == RESOLVED
    assert "1折" in state.deixis.target_text
    assert state.media_present is False
    prompt = format_discourse_prompt_block(state)
    assert "addressee" in prompt
    assert "media_present=false" in prompt
    assert "不要问缺图或链接" in prompt
    assert not draft_violates_media_gate("这折扣挺猛", state)
    assert draft_violates_media_gate("把图发一下", state)
    assert draft_violates_media_gate("截图呢", state)
    assert draft_violates_media_gate("链接呢", state)
    assert draft_violates_media_gate("图片没看到", state)
    trace = discourse_decision_trace(state, final_action="answer")
    assert trace["speaker"] == _label(P1, "P1")
    assert trace["addressee"]["target"] == _label(P2, "P2") or str(P2) in str(trace["addressee"])
    assert trace["addressee"]["status"] == RESOLVED
    assert trace["deixis"]["status"] == RESOLVED
    assert trace["deixis"]["target_text"] == "1折"
    assert trace["media_present"] is False
    assert trace["ambiguity"] == "NON_BLOCKING"
    assert trace["state_audit"] == "PASS"
    assert trace["final_action"] == "answer"


def test_scenario_reply_you_this_is_wrong_addressee_is_reply_target() -> None:
    reply = ReplyHint(exists=True, author_id=P2, author_label=_label(P2, "P2"), text="我觉得没问题", message_id="9")
    rows = build_addressee_candidates(
        current_user_id=P1,
        current_label=_label(P1, "P1"),
        current_text="你这个不对",
        reply=reply,
        at_user_ids=(),
    )
    keys = {row.key for row in rows}
    assert "c_reply" in keys
    judged = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_reply", confidence=0.88, reason="jev_reply"),
        rows,
    )
    assert judged.status == RESOLVED
    assert judged.target_id == P2


def test_scenario_reply_a_at_b_is_jev_choice_not_hardcoded() -> None:
    reply = ReplyHint(exists=True, author_id=A, author_label=_label(A, "A"), text="方案A", message_id="1")
    rows = build_addressee_candidates(
        current_user_id=P1,
        current_label=_label(P1, "P1"),
        current_text="[@4004] 你看这个",
        reply=reply,
        at_user_ids=(B,),
        mention_labels=((_label(B, "B"), B),),
    )
    keys = {row.key: row for row in rows}
    assert "c_reply" in keys and keys["c_reply"].user_id == A
    assert "c_mention" in keys and keys["c_mention"].user_id == B
    assert "generic" in keys
    assert "other" in keys
    picked_reply = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_reply", confidence=0.8, reason="jev_reply"),
        rows,
    )
    picked_mention = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_mention", confidence=0.8, reason="jev_mention"),
        rows,
    )
    assert picked_reply.target_id == A
    assert picked_mention.target_id == B
    unavailable = apply_jev_addressee_judgement(None, rows)
    assert unavailable.status == UNAVAILABLE
    assert unavailable.status != NONE
    assert unavailable.target_id is None


def test_scenario_image_then_look_at_this_can_bind_media() -> None:
    recent = [
        _msg(
            P1,
            "P1",
            "[图片]",
            mid="1",
            ts=10.0,
            segments='[{"type":"image","data":{"url":"https://x/a.png"}}]',
        ),
    ]
    assert message_has_real_media(recent[0]) is True
    rows = build_deixis_candidates(
        recent,
        current_text="看看这个",
        current_user_id=P1,
        media_present=True,
        current_has_media=True,
    )
    media_row = next(row for row in rows if row.source_reason == "media")
    resolved = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from=media_row.key, confidence=0.91),
        rows,
        current_text="看看这个",
    )
    state = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="看看这个",
        deixis=resolved,
        media_present=True,
    )
    assert state.media_present is True
    assert state.deixis.status == RESOLVED
    assert not draft_violates_media_gate("这图有点糊", state)


def test_scenario_text_only_look_at_this_forbids_missing_media_clarify() -> None:
    recent = [_msg(P1, "P1", "1折", mid="1", ts=10.0)]
    rows = build_deixis_candidates(
        recent,
        current_text="看看这个",
        current_user_id=P1,
        media_present=False,
    )
    inherit = next(row.key for row in rows if row.text == "1折")
    deixis = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from=inherit, confidence=0.86),
        rows,
        current_text="看看这个",
    )
    state = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="看看这个",
        deixis=deixis,
        media_present=False,
    )
    assert state.deixis.status == RESOLVED
    assert state.deixis.target_text == "1折"
    assert draft_violates_media_gate("图呢，发一下截图", state)
    prompt = format_discourse_prompt_block(state)
    assert "不要问缺图或链接" in prompt
    assert "unresolved" not in prompt.lower() or "UNAVAILABLE" not in prompt


def test_scenario_jev_unavailable_stays_unavailable_not_none() -> None:
    addressee = apply_jev_addressee_judgement(None, [])
    assert addressee.status == UNAVAILABLE
    assert addressee.status != NONE
    assert addressee.status != NOT_APPLICABLE
    deixis = apply_jev_ellipsis_judgement(None, [], current_text="看看这个")
    assert deixis.status == UNAVAILABLE
    assert deixis.status != NONE
    state = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="看看这个",
        addressee=addressee,
        deixis=deixis,
        media_present=False,
        state_audit="UNAVAILABLE",
    )
    assert state.addressee.status == UNAVAILABLE
    assert state.deixis.status == UNAVAILABLE
    trace = discourse_decision_trace(state, final_action="answer")
    assert trace["addressee"]["status"] == UNAVAILABLE
    assert trace["state_audit"] == "UNAVAILABLE"


def test_scenario_similar_discount_texts_are_non_blocking() -> None:
    effect = classify_ambiguity_effect(
        candidates=("1折", "充1块算10块"),
        kind="ITEM",
        resolved_deixis_text="1折",
        jev_effect="NON_BLOCKING",
    )
    assert effect == "NON_BLOCKING"
    blocking = classify_ambiguity_effect(
        candidates=("小鸟", "小王"),
        kind="PERSON",
        jev_effect="BLOCKING",
    )
    assert blocking == "BLOCKING"


def test_scenario_repair_invalidates_old_deixis_then_recompute() -> None:
    old_ellipsis = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="s1", confidence=0.9),
        [DeixisCandidate(key="s1", speaker="P1", text="现在这个套餐", source_reason="same_speaker")],
        current_text="看看这个",
    )
    assert old_ellipsis.source_text == "现在这个套餐"
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="ITEM", target_key="t_ellipsis", confidence=0.9),
        [RepairTarget(key="t_ellipsis", kind="item", summary="现在这个套餐")],
        current_text="不是那个，我说的是前面那个套餐",
        recent_messages=[_msg(P1, "P1", "前面那个套餐更便宜", mid="0", ts=1.0)],
    )
    reference = ReferenceResolution()
    _, reset_ellipsis, _, _, invalidated = apply_repair_invalidation(
        repair=repair,
        reference=reference,
        ellipsis=old_ellipsis,
    )
    assert "ellipsis" in invalidated
    assert reset_ellipsis.status == NOT_APPLICABLE
    assert reset_ellipsis.source_text == ""
    recomputed = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="s0", confidence=0.88),
        [DeixisCandidate(key="s0", speaker="P1", text="前面那个套餐更便宜", source_reason="same_speaker")],
        current_text="不是那个，我说的是前面那个套餐",
    )
    assert recomputed.source_text == "前面那个套餐更便宜"


def test_scenario_state_audit_conflict_triggers_targeted_requery() -> None:
    addressee_rows = build_addressee_candidates(
        current_user_id=P1,
        current_label=_label(P1, "P1"),
        current_text="[@4004] 你看这个",
        reply=ReplyHint(exists=True, author_id=A, author_label=_label(A, "A"), text="方案A", message_id="1"),
        at_user_ids=(B,),
        mention_labels=((_label(B, "B"), B),),
    )
    first = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_reply", confidence=0.7, reason="jev_reply"),
        addressee_rows,
    )
    assert first.target_id == A
    fields = targeted_requery_fields(
        DiscourseAuditJudgement(conflict_field="addressee", confidence=0.8, reason="reply_vs_mention")
    )
    assert fields == ("addressee",)
    second = apply_jev_addressee_judgement(
        AddresseeJudgement(target_key="c_mention", confidence=0.9, reason="jev_mention"),
        addressee_rows,
    )
    assert second.target_id == B
    audit = apply_jev_discourse_audit(
        DiscourseAuditJudgement(conflict_field="none", confidence=0.9, reason="ok")
    )
    assert audit == "PASS"


def test_deixis_triggers_cover_listed_surfaces() -> None:
    for text in (
        "这个",
        "那个",
        "这样",
        "这东西",
        "这个方案",
        "这个价格",
        "这个图",
        "看看这个",
        "你看这个",
        "[@2002] 看看这个",
    ):
        assert should_ask_jev_ellipsis(text), text


def test_resolved_deixis_is_not_reopened_by_ambiguity_or_critic_state() -> None:
    deixis = apply_jev_ellipsis_judgement(
        EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="s4", confidence=0.9),
        [DeixisCandidate(key="s4", speaker="P1", text="1折", source_reason="same_speaker")],
        current_text="看看这个",
    )
    state = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="看看这个",
        deixis=deixis,
        media_present=False,
        ambiguity_effect="NON_BLOCKING",
    )
    assert state.deixis.status == RESOLVED
    assert state.deixis.target_text == "1折"
    critic_state = format_critic_jev_state(
        draft="这1折可以冲",
        current_text="看看这个",
        action="answer",
        discourse=state,
    )
    assert "1折" in critic_state
    assert "已解析" in critic_state or "discourse" in critic_state.lower() or "deixis" in critic_state.lower()


def test_resolve_group_discourse_end_to_end_with_scripted_jev() -> None:
    recent = [
        _msg(P1, "P1", "哦冲一块钱当十块钱", mid="1", ts=10.0),
        _msg(P1, "P1", "1折", mid="2", ts=11.0),
    ]
    jev = ScriptedJev(
        addressee=AddresseeJudgement(target_key="c_mention", confidence=0.93, reason="jev_mention"),
        ellipsis=EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="AUTO", confidence=0.9),
        audit=DiscourseAuditJudgement(conflict_field="none", confidence=0.9, reason="ok"),
        ambiguity=AmbiguityJudgement(kind="NONE", confidence=0.8),
    )

    async def _run():
        return await resolve_group_discourse(
            current_text="[@2002] 看看这个",
            current_user_id=P1,
            current_nickname="P1",
            self_id=1801507496,
            recent_messages=recent,
            reply=ReplyHint(),
            at_user_ids=(P2,),
            named_resolver=lambda text: (P2,) if "2002" in text or "P2" in text else (),
            jev=jev,
            current_has_media=False,
            reply_has_media=False,
        )

    state = asyncio.run(_run())
    assert "addressee" not in jev.calls
    assert "ellipsis" in jev.calls
    assert "audit" not in jev.calls
    assert state.addressee.target_id == P2
    assert state.deixis.status == RESOLVED
    assert "1折" in state.deixis.target_text or "一块钱" in state.deixis.target_text
    assert state.media_present is False
    assert state.ambiguity in {"NONE", "NON_BLOCKING"}
    assert state.state_audit == "PASS"
    assert not draft_violates_media_gate("这折扣能冲", state)


def test_resolve_group_discourse_audit_conflict_requeries_only_addressee() -> None:
    recent = [_msg(P1, "P1", "1折", mid="1", ts=10.0)]
    answers = {
        "addressee": [
            AddresseeJudgement(target_key="c_reply", confidence=0.7, reason="first"),
            AddresseeJudgement(target_key="c_mention", confidence=0.91, reason="second"),
        ],
        "ellipsis": EllipsisJudgement(kind="ITEM_DEIXIS", inherit_from="AUTO", confidence=0.9),
        "audit": DiscourseAuditJudgement(conflict_field="addressee", confidence=0.85, reason="reply_vs_at"),
        "ambiguity": AmbiguityJudgement(kind="NONE", confidence=0.7),
    }

    class RequeryJev(ScriptedJev):
        async def resolve_addressee(self, **kwargs):
            self.calls.append("addressee")
            queued = self.answers["addressee"]
            if isinstance(queued, list):
                return queued.pop(0)
            return queued

    jev = RequeryJev(**answers)
    reply = ReplyHint(exists=True, author_id=A, author_label=_label(A, "A"), text="方案A", message_id="9")

    async def _run():
        return await resolve_group_discourse(
            current_text="[@4004] 你看这个",
            current_user_id=P1,
            current_nickname="P1",
            self_id=1801507496,
            recent_messages=recent,
            reply=reply,
            at_user_ids=(B,),
            named_resolver=lambda text: (B,) if "4004" in text else (),
            jev=jev,
            current_has_media=False,
            reply_has_media=False,
        )

    state = asyncio.run(_run())
    assert jev.calls.count("addressee") == 2
    assert jev.calls.count("ellipsis") == 1
    assert state.addressee.target_id == B
    assert state.state_audit in {"PASS", "REQUERY"}


def test_addressee_choice_criteria_keep_other() -> None:
    from qq_social_agent.discourse_state import addressee_choice_criteria

    rows = [
        AddresseeCandidate(key="c_reply", user_id=A, label=_label(A, "A"), source="reply"),
        AddresseeCandidate(key="c_mention", user_id=B, label=_label(B, "B"), source="at"),
        AddresseeCandidate(key="generic", user_id=None, label="群里泛说", source="generic"),
        AddresseeCandidate(key="other", user_id=None, label="其他", source="other"),
    ]
    criteria = addressee_choice_criteria(rows)
    assert "other" in criteria
    assert "generic" in criteria
    assert "c_reply" in criteria
    assert "c_mention" in criteria


def test_discourse_audit_requires_comparable_evidence() -> None:
    single = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="你怎么看",
        addressee=apply_jev_addressee_judgement(
            AddresseeJudgement("c_mention", 0.9),
            [AddresseeCandidate("c_mention", P2, _label(P2, "P2"), "at")],
        ),
    )
    assert should_audit_discourse_state(single) is False

    competing = assemble_discourse_state(
        speaker_id=P1,
        speaker_label=_label(P1, "P1"),
        current_text="[@4004] 你看",
        addressee=single.addressee,
        reply_target=single.addressee,
        mentions=(single.addressee,),
    )
    assert should_audit_discourse_state(competing) is True


def test_resolve_group_discourse_uses_one_first_pass_batch() -> None:
    class BatchJev(ScriptedJev):
        async def resolve_discourse_first_pass(self, **kwargs):
            self.calls.append("batch")
            assert kwargs["addressee_candidates"]
            assert kwargs["ellipsis_sources"]
            return {
                "addressee": AddresseeJudgement("c_mention", 0.91),
                "ellipsis": EllipsisJudgement("ITEM_DEIXIS", "AUTO", 0.88),
            }

    jev = BatchJev(audit=DiscourseAuditJudgement("none", 0.9))
    state = asyncio.run(
        resolve_group_discourse(
            current_text="[@2002] 看看这个",
            current_user_id=P1,
            current_nickname="P1",
            self_id=1801507496,
            recent_messages=[_msg(P1, "P1", "1折", mid="1", ts=10.0)],
            reply=ReplyHint(exists=True, author_id=A, author_label=_label(A, "A"), text="方案A"),
            at_user_ids=(P2,),
            named_resolver=lambda _text: (P2,),
            jev=jev,
        )
    )
    assert jev.calls.count("batch") == 1
    assert "addressee" not in jev.calls
    assert "ellipsis" not in jev.calls
    assert state.addressee.target_id == P2
    assert state.deixis.status == RESOLVED
