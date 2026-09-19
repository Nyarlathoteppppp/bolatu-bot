from __future__ import annotations

from qq_social_agent.memory import ChatMessage
from qq_social_agent.reference_resolver import (
    ReferentJudgement,
    ReferenceResolution,
    ReplyHint,
    apply_jev_referent_judgement,
    build_referent_candidates,
    has_strong_person_reference,
    parse_jev_referent_answers,
    resolve_context_reference,
    should_ask_jev_referent,
)


BIRD = 184589072
JIA = 7
YI = 8
BOT = 1801507496
NV2 = 9


def _named(text: str) -> tuple[int, ...]:
    ids = []
    if "小鸟" in text:
        ids.append(BIRD)
    if "甲" in text:
        ids.append(JIA)
    if "乙" in text:
        ids.append(YI)
    return tuple(ids)


def test_pronoun_inherits_previous_named_member() -> None:
    messages = [
        ChatMessage(1, JIA, "甲", "小鸟以前准备考研", False, 10.0),
        ChatMessage(1, 99, "张风雪", "好像聊过", True, 11.0),
    ]
    resolution = resolve_context_reference(
        "他现在还考吗",
        messages,
        current_user_id=YI,
        resolve_named_users=_named,
    )
    assert resolution.user_ids == (BIRD,)
    assert resolution.reason == "previous_named_member"
    assert "承接前文" in resolution.expanded_query
    assert should_ask_jev_referent("他现在还考吗", resolution) is True


def test_named_in_current_skips_jev() -> None:
    resolution = resolve_context_reference(
        "小鸟她考研吗？",
        [ChatMessage(1, JIA, "甲", "随便聊聊", False, 10.0)],
        current_user_id=YI,
        resolve_named_users=_named,
    )
    assert resolution.user_ids == (BIRD,)
    assert resolution.reason == "named_in_current"
    assert resolution.confidence >= 0.85
    assert should_ask_jev_referent("小鸟她考研吗？", resolution) is False


def test_ordinary_message_does_not_force_reference() -> None:
    messages = [ChatMessage(1, JIA, "甲", "随便聊聊", False, 10.0)]
    resolution = resolve_context_reference(
        "今天吃什么",
        messages,
        current_user_id=YI,
    )
    assert resolution.user_ids == ()
    assert resolution.reason == "none"
    assert should_ask_jev_referent("今天吃什么", resolution) is False


def test_elliptical_followup_does_not_bind_latest_speaker() -> None:
    messages = [ChatMessage(1, JIA, "甲", "我以前准备考研", False, 10.0)]
    resolution = resolve_context_reference(
        "那后来呢",
        messages,
        current_user_id=YI,
    )
    assert resolution.user_ids == ()
    assert resolution.kind == "NONE"
    assert resolution.reason == "none"


def test_mentioned_but_silent_person_enters_candidate_pool() -> None:
    messages = [
        ChatMessage(1, JIA, "甲", "小鸟以前准备考研", False, 10.0),
        ChatMessage(1, YI, "乙", "今天吃什么", False, 11.0),
    ]
    rows = build_referent_candidates(
        messages,
        current_user_id=YI,
        current_text="她考哪里？",
        self_id=BOT,
        resolve_named_users=_named,
    )
    keys = {row.key: row for row in rows}
    assert f"u{BIRD}" in keys
    assert f"u{BOT}" in keys
    assert "mentioned" in keys[f"u{BIRD}"].sources
    evidence = "\n".join(keys[f"u{BIRD}"].evidence)
    assert "小鸟以前准备考研" in evidence
    assert keys[f"u{BOT}"].is_bot


def test_reply_author_is_candidate_but_mentioned_person_can_win() -> None:
    messages = [ChatMessage(1, JIA, "甲", "小鸟以前准备考研", False, 10.0)]
    reply = ReplyHint(
        exists=True,
        author_id=JIA,
        author_label="甲[#00007]",
        text="小鸟以前准备考研",
        message_id="42",
    )
    rows = build_referent_candidates(
        messages,
        current_user_id=YI,
        current_text="她呢？",
        self_id=BOT,
        reply=reply,
        resolve_named_users=_named,
        rule_guess=ReferenceResolution((JIA,), reason="latest_other_speaker", confidence=0.72, kind="PERSON"),
    )
    keys = {row.key: row for row in rows}
    assert keys[f"u{JIA}"].sources[0] == "reply" or "reply" in keys[f"u{JIA}"].sources
    assert "reply_mention" in keys[f"u{BIRD}"].sources or "mentioned" in keys[f"u{BIRD}"].sources
    judged = ReferentJudgement(kind="PERSON", person_key=f"u{BIRD}", confidence=0.81, reason="jev_person")
    resolved = apply_jev_referent_judgement(
        judged,
        rows,
        current_text="她呢？",
        fallback=ReferenceResolution((JIA,), reason="latest_other_speaker", confidence=0.72, kind="PERSON"),
    )
    assert resolved.user_ids == (BIRD,)
    assert resolved.unresolved is False


def test_non_person_exam_does_not_bind_member() -> None:
    fallback = ReferenceResolution((JIA,), reason="latest_other_speaker", confidence=0.72, kind="PERSON")
    rows = build_referent_candidates(
        [ChatMessage(1, JIA, "甲", "考研好难", False, 10.0)],
        current_user_id=YI,
        current_text="那个考试难吗？",
        self_id=BOT,
    )
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="NON_PERSON", person_key=f"u{JIA}", confidence=0.9, reason="exam"),
        rows,
        current_text="那个考试难吗？",
        fallback=fallback,
    )
    assert resolved.user_ids == ()
    assert resolved.kind == "NON_PERSON"
    assert resolved.unresolved is False


def test_non_person_plugin() -> None:
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="NON_PERSON", person_key="NONE", confidence=0.8, reason="plugin"),
        [],
        current_text="那个插件还用吗？",
        fallback=ReferenceResolution(),
    )
    assert resolved.kind == "NON_PERSON"
    assert resolved.user_ids == ()


def test_bot_can_be_selected_for_she_just_said() -> None:
    messages = [ChatMessage(1, BOT, "张风雪", "我觉得这个题不难", True, 10.0)]
    rows = build_referent_candidates(
        messages,
        current_user_id=JIA,
        current_text="她刚才说什么？",
        self_id=BOT,
    )
    bot_row = next(row for row in rows if row.is_bot)
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="PERSON", person_key=bot_row.key, confidence=0.7, reason="bot"),
        rows,
        current_text="她刚才说什么？",
        fallback=ReferenceResolution(),
    )
    assert resolved.user_ids == (BOT,)


def test_two_plausible_women_stay_ambiguous() -> None:
    rows = [
        type("C", (), {"key": "u3", "user_id": 3, "evidence": ("甲：小鸟来了",), "label": "小鸟"})(),
        type("C", (), {"key": "u9", "user_id": 9, "evidence": ("乙：安钰也在考",), "label": "安钰"})(),
    ]
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="PERSON", person_key="NONE", confidence=0.4, reason="both"),
        rows,
        current_text="她到底考哪？",
        fallback=ReferenceResolution((3,), reason="latest_other_speaker", confidence=0.72, kind="PERSON"),
    )
    assert resolved.user_ids == ()
    assert resolved.unresolved is True
    assert resolved.kind == "PERSON"


def test_jev_can_override_weak_latest_speaker_prior() -> None:
    fallback = ReferenceResolution((7,), reason="latest_other_speaker", confidence=0.72, kind="PERSON")
    rows = [
        type("C", (), {"key": "u7", "user_id": 7, "evidence": ("甲：随便",), "label": "甲"})(),
        type("C", (), {"key": "u3", "user_id": 3, "evidence": ("甲：小鸟以前准备考研",), "label": "小鸟"})(),
    ]
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="PERSON", person_key="u3", confidence=0.77, reason="mentioned"),
        rows,
        current_text="她现在还考吗",
        fallback=fallback,
    )
    assert resolved.user_ids == (3,)


def test_low_confidence_does_not_bind() -> None:
    rows = [type("C", (), {"key": "u7", "user_id": 7, "evidence": ("甲：嗨",), "label": "甲"})()]
    resolved = apply_jev_referent_judgement(
        ReferentJudgement(kind="PERSON", person_key="u7", confidence=0.4, reason="weak"),
        rows,
        current_text="她呢？",
        fallback=ReferenceResolution((7,), reason="latest_other_speaker", confidence=0.72, kind="PERSON"),
    )
    assert resolved.user_ids == ()
    assert resolved.unresolved is True
    assert resolved.reason == "jev_low_confidence"


def test_weak_that_does_not_preset_person() -> None:
    assert has_strong_person_reference("那个考试难吗？") is False
    assert should_ask_jev_referent("那个考试难吗？", ReferenceResolution()) is True


def test_stale_named_member_does_not_skip_jev() -> None:
    messages = [
        ChatMessage(1, JIA, "甲", "小鸟以前准备考研", False, 10.0),
        ChatMessage(1, NV2, "乙", "今天食堂好挤", False, 11.0),
    ]
    resolution = resolve_context_reference(
        "他现在还考吗",
        messages,
        current_user_id=YI,
        resolve_named_users=_named,
    )
    assert resolution.reason != "previous_named_member"
    assert resolution.confidence < 0.85
    assert should_ask_jev_referent("他现在还考吗", resolution) is True


def test_referent_unavailable_is_not_not_applicable() -> None:
    resolved = apply_jev_referent_judgement(
        None,
        [],
        current_text="她呢？",
        fallback=ReferenceResolution(),
    )
    assert resolved.status == "UNAVAILABLE"
    assert resolved.kind == "PERSON"
    none = apply_jev_referent_judgement(
        ReferentJudgement(kind="NONE", person_key="NONE", confidence=0.8),
        [],
        current_text="那个插件还用吗？",
        fallback=ReferenceResolution(),
    )
    assert none.status == "NOT_APPLICABLE"
    assert none.unresolved is False


def test_previous_named_member_is_prior_not_lock() -> None:
    messages = [
        ChatMessage(1, JIA, "甲", "小鸟以前准备考研", False, 10.0),
    ]
    resolution = resolve_context_reference(
        "他现在还考吗",
        messages,
        current_user_id=YI,
        resolve_named_users=_named,
    )
    assert resolution.reason == "previous_named_member"
    assert resolution.user_ids == (BIRD,)
    assert should_ask_jev_referent("他现在还考吗", resolution) is True


def test_missing_referent_answers_are_error_not_none() -> None:
    judged = parse_jev_referent_answers({"answers": {}})
    assert judged.reason == "error"
    resolved = apply_jev_referent_judgement(
        judged,
        [],
        current_text="她呢？",
        fallback=ReferenceResolution(),
    )
    assert resolved.status == "ERROR"
    assert resolved.kind == "PERSON"
    assert resolved.unresolved is True
