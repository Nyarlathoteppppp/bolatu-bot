from qq_social_agent.memory import ChatMessage
from qq_social_agent.private_context_window import current_private_session_messages


def _message(text: str, created_at: float, *, is_bot: bool = False) -> ChatMessage:
    return ChatMessage(
        group_id=10003115344487,
        user_id=1801507496 if is_bot else 3115344487,
        nickname="风雪" if is_bot else "对方",
        text=text,
        is_bot=is_bot,
        created_at=created_at,
    )


def test_old_private_joke_does_not_enter_new_session() -> None:
    messages = [
        _message("想你就先给钱", 100, is_bot=True),
        _message("别说这个了", 110),
        _message("想你", 110 + 30 * 24 * 60 * 60),
    ]

    assert [item.text for item in current_private_session_messages(messages)] == ["想你"]


def test_short_pause_keeps_direct_chat_context() -> None:
    messages = [_message("帮我看看", 100), _message("发我看看", 101, is_bot=True), _message("这个", 1000)]

    assert current_private_session_messages(messages) == messages
