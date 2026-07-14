import random
import time

from qq_social_agent.memory import ChatMessage
from qq_social_agent.persona import Persona
from qq_social_agent.scorer import score_message


def _persona() -> Persona:
    return Persona(
        id="test",
        name="Test",
        description="",
        prompt="",
        decision_prompt="",
        max_reply_chars=120,
        passive_reply_probability=1.0,
    )


def test_mention_always_passes() -> None:
    result = score_message(
        text="老师我该不该考研",
        recent_messages=[],
        persona=_persona(),
        mentioned=True,
        replied_to_bot=False,
        passive_threshold=80,
        passive_probability=0.0,
    )
    assert result.should_reply
    assert result.score >= 100


def test_low_value_text_is_suppressed() -> None:
    result = score_message(
        text="6",
        recent_messages=[],
        persona=_persona(),
        mentioned=False,
        replied_to_bot=False,
        passive_threshold=10,
        passive_probability=1.0,
        rng=random.Random(1),
    )
    assert not result.should_reply


def test_substantial_question_can_pass_passive_gate() -> None:
    result = score_message(
        text="计算机就业现在到底还能不能冲？",
        recent_messages=[],
        persona=_persona(),
        mentioned=False,
        replied_to_bot=False,
        passive_threshold=45,
        passive_probability=1.0,
        rng=random.Random(1),
    )
    assert result.should_reply


def test_casual_chat_can_pass_low_passive_gate() -> None:
    result = score_message(
        text="绝区零是好游戏",
        recent_messages=[],
        persona=_persona(),
        mentioned=False,
        replied_to_bot=False,
        passive_threshold=15,
        passive_probability=1.0,
        rng=random.Random(1),
    )
    assert result.should_reply


def test_recent_bot_reply_dampens_but_does_not_block_strong_topic() -> None:
    recent = [
        ChatMessage(1, 2, "bot", "刚说过", True, time.time()),
    ]
    result = score_message(
        text="计算机就业现在到底还能不能冲？",
        recent_messages=recent,
        persona=_persona(),
        mentioned=False,
        replied_to_bot=False,
        passive_threshold=35,
        passive_probability=1.0,
        rng=random.Random(1),
    )
    assert result.should_reply
    assert "recent_bot_reply" in result.reasons
