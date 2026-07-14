from __future__ import annotations

from qq_social_agent.memory import ChatMessage
from qq_social_agent.reference_resolver import resolve_context_reference


def test_pronoun_inherits_previous_named_member() -> None:
    messages = [
        ChatMessage(1, 7, "甲", "小鸟以前准备考研", False, 10.0),
        ChatMessage(1, 99, "张风雪", "好像聊过", True, 11.0),
    ]

    resolution = resolve_context_reference(
        "他现在还考吗",
        messages,
        current_user_id=8,
        resolve_named_users=lambda text: (184589072,) if "小鸟" in text else (),
    )

    assert resolution.user_ids == (184589072,)
    assert resolution.reason == "previous_named_member"
    assert "承接前文" in resolution.expanded_query


def test_ordinary_message_does_not_force_reference() -> None:
    messages = [ChatMessage(1, 7, "甲", "随便聊聊", False, 10.0)]

    resolution = resolve_context_reference(
        "今天吃什么",
        messages,
        current_user_id=8,
    )

    assert resolution.user_ids == ()
    assert resolution.reason == "none"


def test_elliptical_followup_uses_latest_other_speaker_conservatively() -> None:
    messages = [ChatMessage(1, 7, "甲", "我以前准备考研", False, 10.0)]

    resolution = resolve_context_reference(
        "那后来呢",
        messages,
        current_user_id=8,
    )

    assert resolution.user_ids == (7,)
    assert resolution.reason == "latest_other_speaker"
