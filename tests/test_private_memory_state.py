from qq_social_agent.memory import MemoryStore


def test_private_conversation_state_respects_manual_locks(tmp_path):
    store = MemoryStore(tmp_path / "bot.sqlite3")
    chat_id = 10_001_903_297_906
    state = store.update_private_conversation_state(
        chat_id=chat_id,
        user_id=1_903_297_906,
        display_name="伪物",
        relationship_note="熟人朋友",
        current_topic="动捕项目",
        open_threads=["比较动捕方案", "选一个可做的 demo"],
        frozen_fields=["relationship_note", "current_topic"],
    )
    assert state.current_topic == "动捕项目"
    assert len(state.open_threads) == 2

    refreshed = store.refresh_private_conversation_learning(
        chat_id=chat_id,
        user_id=1_903_297_906,
        display_name="新昵称",
        open_threads=["自动学习到的新话题"],
    )
    assert refreshed.display_name == "新昵称"
    assert refreshed.relationship_note == "熟人朋友"
    assert refreshed.current_topic == "动捕项目"
    assert refreshed.open_threads == ("自动学习到的新话题",)
