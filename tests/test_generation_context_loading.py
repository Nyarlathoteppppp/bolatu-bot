from qq_social_agent.discourse_state import DiscourseState
from qq_social_agent.ellipsis_resolver import EllipsisResolution
from qq_social_agent.group_generation_context import load_group_generation_messages
from qq_social_agent.memory import MemoryStore
from qq_social_agent.private_generation_context import load_private_generation_messages
from qq_social_agent.private_message_types import PrivateTurn
from qq_social_agent.reference_resolver import ReplyHint
from qq_social_agent.resolver_result import RESOLVED


def test_generation_reads_busy_group_window_without_expanding_decision_window(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(51):
        memory.add_message(
            1, 2, "A", f"line-{index}",
            created_at=1000 + index,
            source_message_id=f"source-{index}",
        )
    recent = memory.recent_messages(1, 40)[:-1]
    assert "line-2" not in [message.text for message in recent]

    candidates, pinned_sources, _ = load_group_generation_messages(
        memory=memory,
        group_id=1,
        recent_messages=recent,
        discourse=DiscourseState(),
        reply_hint=ReplyHint(),
        excluded_source_ids={"source-50"},
        lookback_seconds=900,
        now=1050,
    )

    assert "line-2" in [message.text for message in candidates]
    assert "line-50" not in [message.text for message in candidates]
    assert len(recent) == 39
    assert len(candidates) == 50
    assert not pinned_sources


def test_generation_pins_exact_quoted_and_resolved_sources_outside_window(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 2, "A", "old quoted image", created_at=100, source_message_id="quote")
    memory.add_message(1, 2, "A", "old resolved source", created_at=200, source_message_id="resolved")
    memory.add_message(1, 2, "A", "current", created_at=2000, source_message_id="current")

    candidates, pinned_sources, pinned_ids = load_group_generation_messages(
        memory=memory,
        group_id=1,
        recent_messages=[],
        discourse=DiscourseState(ellipsis=EllipsisResolution(status=RESOLVED, source_message_id="resolved")),
        reply_hint=ReplyHint(exists=True, message_id="quote"),
        excluded_source_ids={"current"},
        lookback_seconds=900,
        now=2000,
    )

    assert [message.text for message in candidates] == ["old quoted image", "old resolved source"]
    assert pinned_sources == {"quote", "resolved"}
    assert len(pinned_ids) == 2


def test_private_session_reads_beyond_40_and_stops_at_last_12_hour_gap(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    chat_id = 10_000_000_000_001
    memory.add_message(chat_id, 2, "A", "old session", created_at=100, source_message_id="old")
    for index in range(45):
        memory.add_message(
            chat_id, 2, "A", f"current-{index}",
            created_at=50_000 + index,
            source_message_id=f"current-{index}",
        )
    session = memory.current_session_messages(chat_id, gap_seconds=12 * 60 * 60)
    assert len(session) == 45
    assert session[0].text == "current-0"
    assert session[-1].text == "current-44"


def test_private_generation_excludes_every_message_in_current_buffered_turn(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    chat_id = 10_000_000_000_001
    for index, source_id in enumerate(("previous", "burst-1", "burst-2")):
        memory.add_message(
            chat_id, 2, "A", source_id,
            created_at=50_000 + index,
            source_message_id=source_id,
        )
    turn = PrivateTurn(
        bot=None, user_id=2, chat_id=chat_id, self_id=1, nickname="A",
        source_message_id="burst-2", correlation_id="", text="burst-1\nburst-2",
        forced_once_context="", received_message_count=2,
        current_source_message_ids=("burst-1", "burst-2"),
    )
    selected = load_private_generation_messages(memory, turn, ())
    assert [message.text for message in selected] == ["previous"]
