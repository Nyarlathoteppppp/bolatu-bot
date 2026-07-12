import asyncio
from types import SimpleNamespace

from qq_social_agent.notice_events import notice_snapshot
from qq_social_agent.social_actions import PokeResult


def test_notice_snapshot_keeps_safe_structured_event_metadata() -> None:
    event = SimpleNamespace(
        notice_type="group_upload",
        sub_type="upload",
        group_id="100",
        user_id="200",
        operator_id="300",
        target_id=None,
        message_id="message-1",
        duration="60",
        card_old="旧名片",
        card_new="新名片",
        file={
            "name": "群资料.pdf",
            "size": "2048",
            "url": "https://example.com/download?token=secret",
        },
    )

    snapshot = notice_snapshot(event)
    metadata = snapshot.metric_metadata()

    assert snapshot.group_id == 100
    assert snapshot.user_id == 200
    assert metadata["file_name"] == "群资料.pdf"
    assert metadata["file_size"] == 2048
    assert metadata["old_card"] == "旧名片"
    assert metadata["new_card"] == "新名片"
    assert "url" not in metadata
    assert "secret" not in str(metadata)


def test_directory_refresh_notice_classification() -> None:
    import nonebot

    nonebot.init()
    import qq_social_agent.plugin as plugin

    assert plugin._notice_needs_directory_refresh("group_increase", "approve")
    assert plugin._notice_needs_directory_refresh("notify", "group_card")
    assert not plugin._notice_needs_directory_refresh("group_recall", "recall")


def test_poke_notice_only_reciprocates_when_bot_is_target(monkeypatch) -> None:
    import nonebot

    nonebot.init()
    import qq_social_agent.plugin as plugin

    calls: list[tuple[int, int, object]] = []

    class FakeService:
        async def poke_user(self, bot, *, group_id: int, user_id: int, context: object) -> PokeResult:
            calls.append((group_id, user_id, context))
            return PokeResult(True, "sent", "reciprocal_poke")

    monkeypatch.setattr(plugin, "social_action_service", FakeService())
    bot = SimpleNamespace(self_id=555)
    target_bot = SimpleNamespace(
        notice_type="notify", sub_type="poke", group_id=100, user_id=200, target_id=555
    )
    target_other = SimpleNamespace(
        notice_type="notify", sub_type="poke", group_id=100, user_id=201, target_id=999
    )

    asyncio.run(plugin._handle_notice_social_action(bot, target_bot))
    asyncio.run(plugin._handle_notice_social_action(bot, target_other))

    assert len(calls) == 1
    assert calls[0][:2] == (100, 200)
