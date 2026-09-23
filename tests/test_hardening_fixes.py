import asyncio
from types import SimpleNamespace

import nonebot

nonebot.init()

from qq_social_agent.memory import ChatMessage, MemoryStore
from qq_social_agent.plugin import (
    BASIC_APPROVAL_DENIED_MESSAGE,
    _compact_reply_text,
    _handle_memory_atom_command_text,
    _is_local_admin_request,
    _is_near_duplicate_bot_reply,
)


def test_local_admin_allows_loopback_and_docker_gateway_only() -> None:
    def req(host: str):
        return SimpleNamespace(client=SimpleNamespace(host=host))

    assert _is_local_admin_request(req("127.0.0.1"))
    assert _is_local_admin_request(req("::1"))
    assert _is_local_admin_request(req("172.18.0.1"))
    assert not _is_local_admin_request(req("172.18.0.3"))
    assert not _is_local_admin_request(req("172.32.0.1"))
    assert not _is_local_admin_request(req("8.8.8.8"))


def test_near_duplicate_blocks_same_punchline() -> None:
    recent = [
        ChatMessage(
            1, 1801507496, "张风雪",
            "那咋了，泡面汤都算轻的，没直接拿他内裤擦地就不错了",
            True, 1.0,
        )
    ]
    assert _is_near_duplicate_bot_reply(
        "泡面汤都算轻的，没直接拿他内裤擦地就不错了",
        recent,
    )
    assert not _is_near_duplicate_bot_reply(
        "共产主义社会应该给每个人用洗衣机洗内裤的权利",
        recent,
    )
    assert _compact_reply_text("  a  b ") == "ab"


def test_memory_atom_commands_are_group_scoped(tmp_path, monkeypatch) -> None:
    import qq_social_agent.plugin as plugin

    store = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr(plugin, "memory", store)
    own = store.upsert_memory_atom(
        atom_type="note",
        group_id=1026813421,
        content="本群记忆",
        source="test",
        subject_user_id=1,
    )
    other = store.upsert_memory_atom(
        atom_type="note",
        group_id=999,
        content="外群记忆",
        source="test",
        subject_user_id=1,
    )
    denied = _handle_memory_atom_command_text(1, 1026813421, f"删记忆：{other}")
    assert denied == BASIC_APPROVAL_DENIED_MESSAGE or "没找到" in (denied or "")
    scoped = _handle_memory_atom_command_text(1535071184, 1026813421, f"删记忆：{other}")
    assert scoped == "没找到这个记忆单元。"
    assert store.memory_atom(other).status == "active"
    ok = _handle_memory_atom_command_text(1535071184, 1026813421, f"删记忆：{own}")
    assert "软过期" in ok
    assert store.memory_atom(own).status == "expired"
