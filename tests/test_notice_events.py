from types import SimpleNamespace

from qq_social_agent.notice_events import notice_snapshot


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
