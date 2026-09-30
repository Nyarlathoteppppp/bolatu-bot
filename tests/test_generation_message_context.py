from qq_social_agent.generation_message_context import (
    select_generation_messages,
    shortlist_older_messages,
)
from qq_social_agent.memory import ChatMessage


def _messages(count: int) -> list[ChatMessage]:
    return [
        ChatMessage(1, 2, "A", f"line-{index}", False, float(index), id=index + 1,
                    source_message_id=f"qq-{index}")
        for index in range(count)
    ]


def test_weak_history_does_not_fill_context() -> None:
    messages = _messages(40)
    candidates = messages[:-6]
    selected = select_generation_messages(
        messages, candidates=candidates, scores=[0.1] * len(candidates),
        pinned_source_ids=set(), pinned_db_ids=set(),
    )
    assert selected == messages[-6:]


def test_exact_evidence_survives_jev_failure_and_db_id_collision() -> None:
    messages = _messages(40)
    # QQ source ID "12" belongs to an old message. DB id 12 belongs to a different one.
    messages[1] = ChatMessage(1, 2, "A", "quoted", False, 1.0, id=2, source_message_id="12")
    selected = select_generation_messages(
        messages, candidates=[], scores=None,
        pinned_source_ids={"12"}, pinned_db_ids=set(),
    )
    assert [message.text for message in selected] == ["quoted", *[message.text for message in messages[-11:]]]
    assert messages[11] not in selected


def test_shortlist_finds_topic_beyond_jev_first_32() -> None:
    messages = _messages(66)
    messages[2] = ChatMessage(1, 2, "A", "反向代理部署配置", False, 2.0, id=3,
                              source_message_id="evidence")
    older = messages[:-6]
    shortlist = shortlist_older_messages(
        older, current_text="反向代理怎么配置", pinned_source_ids={"evidence"}, pinned_db_ids=set(),
    )
    assert len(shortlist) == 32
    assert messages[2] in shortlist
    assert messages[59] in shortlist
    scores = [0.9 if message is messages[2] else 0.05 for message in shortlist]
    selected = select_generation_messages(
        messages, candidates=shortlist, scores=scores,
        pinned_source_ids={"evidence"}, pinned_db_ids=set(),
    )
    assert selected == [messages[2], *messages[-6:]]


def test_scored_history_uses_only_available_slots_in_time_order() -> None:
    messages = _messages(20)
    candidates = messages[:-6]
    scores = [0.0] * len(candidates)
    scores[2], scores[8] = 0.8, 0.9
    selected = select_generation_messages(
        messages, candidates=candidates, scores=scores,
        pinned_source_ids=set(), pinned_db_ids=set(), max_total=8,
    )
    assert selected == [messages[2], messages[8], *messages[-6:]]


def test_many_exact_sources_are_not_dropped_to_meet_soft_budget() -> None:
    messages = _messages(20)
    selected = select_generation_messages(
        messages, candidates=[], scores=None,
        pinned_source_ids={message.source_message_id for message in messages[:8]},
        pinned_db_ids=set(), max_total=12,
    )
    assert selected == [*messages[:8], *messages[-6:]]


def test_jev_failure_keeps_recent_history_and_exact_evidence() -> None:
    messages = _messages(30)
    selected = select_generation_messages(
        messages, candidates=messages[:24], scores=None,
        pinned_source_ids={"qq-1"}, pinned_db_ids=set(),
    )
    assert selected == [messages[1], *messages[-11:]]
