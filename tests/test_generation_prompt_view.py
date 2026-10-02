from qq_social_agent.deepseek_client import _format_message, _render_utterance
from qq_social_agent.discourse_state import Binding, DiscourseState
from qq_social_agent.ellipsis_resolver import parse_reply_envelope
from qq_social_agent.memory import ChatMessage
from qq_social_agent.reference_resolver import ReferenceResolution
from qq_social_agent.resolver_result import RESOLVED
from qq_social_agent.self_interaction_context import format_self_interaction_context
from qq_social_agent.speaker_context import _message_relation_facts, format_generation_relation

BOT = 1801507496
ENVELOPE = (
    "小鸟[#21852]回复shit♂van（自强）[#68258]消息【shit♂van（自强）[#68258]说：柏拉图的大家怎么变得这么可爱了；"
    "小鸟[#21852]回复shit♂van（自强）[#68258]：大家一直都很可爱呀】"
)


def test_reply_envelope_parses_and_renders_once_per_label() -> None:
    envelope = parse_reply_envelope(ENVELOPE)
    assert envelope is not None
    assert envelope.target == "shit♂van（自强）[#68258]"
    assert envelope.quoted == "柏拉图的大家怎么变得这么可爱了"
    assert envelope.reply == "大家一直都很可爱呀"
    line = _format_message(ChatMessage(1, 21852, "小鸟", ENVELOPE, False, 1.0))
    assert line == "小鸟[#21852]（回复 shit♂van（自强）[#68258]「柏拉图的大家怎么变得这么可爱了」）: 大家一直都很可爱呀"


def test_reply_to_bot_and_unknown_original_render_plainly() -> None:
    text = "A[#11111]回复张风雪（开启）[#07496]消息【张风雪（开启）[#07496]原消息内容未知，消息ID：9；A[#11111]回复张风雪（开启）[#07496]：你知道他在说什么吗】"
    assert _render_utterance("A[#11111]", text) == ("A[#11111]（回复 风雪）", "你知道他在说什么吗")
    assert _render_utterance("A[#11111]", "普通消息") == ("A[#11111]", "普通消息")


def _facts(**kwargs):
    defaults = dict(
        current_user_id=1001,
        current_nickname="P1",
        current_text="偷偷玩不带我",
        reference_resolution=ReferenceResolution(),
        mentioned=False,
        replied_to_bot=False,
        addressed_bot=False,
        followup_addressed=False,
        self_id=BOT,
    )
    defaults.update(kwargs)
    return _message_relation_facts(**defaults)


def test_generation_relation_is_plain_words_without_internal_fields() -> None:
    state = DiscourseState(
        addressee=Binding(status=RESOLVED, target="A[#03003]", target_id=3003, reason="speaker_continuation"),
    )
    text = format_generation_relation(
        current_user_id=1001,
        current_nickname="P1",
        recent_messages=[],
        facts=_facts(),
        discourse_state=state,
        followup_soft=False,
        self_id=BOT,
    )
    assert "在对 A[#03003] 说，不是对你" in text
    for internal in ("status=", "target=", "mentioned_bot", "state_audit", "[discourse]", "message_id"):
        assert internal not in text


def test_generation_relation_for_reply_to_bot() -> None:
    text = format_generation_relation(
        current_user_id=1001,
        current_nickname="P1",
        recent_messages=[],
        facts=_facts(replied_to_bot=True, addressed_bot=True),
        discourse_state=DiscourseState(),
        followup_soft=False,
        self_id=BOT,
    )
    assert "在回复你之前说的话" in text
    assert "没人在问你" not in text


def test_self_interaction_context_is_readable() -> None:
    text = format_self_interaction_context([{
        "said": "别冒充新生",
        "trigger_speaker": "世界毁灭了",
        "trigger_said": "我可以假装我是大一新生",
        "subsequent_feedback": [{"speaker": "世界毁灭了", "kind": "closed", "said": "是这么个理"}],
    }])
    assert text.startswith("<self_interaction_context>")
    assert "回应 世界毁灭了「我可以假装我是大一新生」，你说：「别冒充新生」" in text
    assert "表示这个话题结束了" in text
    assert "{" not in text
    assert format_self_interaction_context([]) == ""
