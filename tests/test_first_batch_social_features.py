import asyncio
from types import SimpleNamespace

import nonebot

nonebot.init()

import qq_social_agent.plugin as plugin
from qq_social_agent.deepseek_client import _parse_reply_decision
from qq_social_agent.history_sync import ReplyReference, backfill_group_history, resolve_reply_reference
from qq_social_agent.media_context import ImageOcrService, parse_ocr_text
from qq_social_agent.memory import MemoryStore
from qq_social_agent.social_actions import SocialActionService


def test_memory_source_message_deduplicates_and_claims(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    assert memory.claim_inbound_message(1, 42, correlation_id="group:1:42")
    assert not memory.claim_inbound_message(1, 42, correlation_id="group:1:42")
    assert memory.add_message(1, 100, "A", "第一句", source_message_id=42)
    assert not memory.add_message(1, 100, "A", "重复句", source_message_id=42)

    messages = memory.recent_messages(1, 5)
    assert [message.text for message in messages] == ["第一句"]


def test_group_directory_overrides_profile_display_name(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 100, "旧昵称", "第一句", created_at=100)

    memory.replace_group_members(
        1,
        [
            {
                "user_id": 100,
                "nickname": "QQ昵称",
                "card": "新群名片",
                "role": "member",
                "title": "",
                "joined_at": 0,
                "last_sent_at": 0,
            }
        ],
        synced_at=200,
    )

    profiles = memory.member_profiles_for_context(1, [100], limit=3)
    assert profiles[0].display_name == "新群名片"
    assert profiles[0].aliases == ("旧昵称",)


def test_history_backfill_inserts_messages_without_triggering_duplicates(tmp_path) -> None:
    class FakeHistoryBot:
        self_id = 999

        async def call_api(self, api: str, **data):
            assert api == "get_group_msg_history"
            assert data["group_id"] == 1
            return {
                "messages": [
                    {
                        "message_id": 10,
                        "group_id": 1,
                        "user_id": 100,
                        "time": 1000,
                        "sender": {"user_id": 100, "nickname": "A"},
                        "message": [{"type": "text", "data": {"text": "历史消息"}}],
                    },
                    {
                        "message_id": 10,
                        "group_id": 1,
                        "user_id": 100,
                        "time": 1000,
                        "sender": {"user_id": 100, "nickname": "A"},
                        "message": [{"type": "text", "data": {"text": "重复历史"}}],
                    },
                ]
            }

    memory = MemoryStore(tmp_path / "bot.sqlite3")
    inserted = asyncio.run(backfill_group_history(FakeHistoryBot(), memory, 1, count=20, self_id=999))

    assert inserted == 1
    assert [message.text for message in memory.recent_messages(1, 3)] == ["历史消息"]
    assert not memory.claim_inbound_message(1, 10)


def test_resolve_reply_reference_uses_get_msg_when_event_reply_has_no_text() -> None:
    class FakeReplyBot:
        async def call_api(self, api: str, **data):
            assert api == "get_msg"
            assert data == {"message_id": 42}
            return {
                "message_id": 42,
                "user_id": 1801507496,
                "sender": {"user_id": 1801507496, "nickname": "张风雪"},
                "message": [{"type": "text", "data": {"text": "风雪觉得这个可以"}}],
            }

    event = SimpleNamespace(
        reply=SimpleNamespace(message_id=42, message=None, user_id=None, sender=None),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "你自己说的"}),
        ],
    )

    reference = asyncio.run(resolve_reply_reference(FakeReplyBot(), event))
    assert reference == ReplyReference("42", 1801507496, "张风雪", "风雪觉得这个可以")

    text = plugin._message_context_text(event, bot_id=1801507496, resolved_reply=reference)
    assert "张风雪和风雪都是你自己" in text
    assert "张风雪[#07496]说：风雪觉得这个可以" in text


def test_social_action_service_reacts_once_and_rate_limits() -> None:
    class FakeReactBot:
        def __init__(self) -> None:
            self.calls = []

        async def call_api(self, api: str, **data):
            self.calls.append((api, data))
            return {}

    bot = FakeReactBot()
    service = SocialActionService(per_user_cooldown_seconds=120, per_group_cooldown_seconds=0)

    first = asyncio.run(
        service.react_to_message(bot, group_id=1, user_id=100, message_id=42, reaction="laugh", now=1000)
    )
    second = asyncio.run(
        service.react_to_message(bot, group_id=1, user_id=100, message_id=43, reaction="laugh", now=1010)
    )

    assert first.sent
    assert second.reason == "user_cooldown"
    assert bot.calls == [("set_msg_emoji_like", {"message_id": 42, "emoji_id": "28"})]


def test_social_action_service_rotates_configured_emoji_ids() -> None:
    class FakeReactBot:
        def __init__(self) -> None:
            self.calls = []

        async def call_api(self, api: str, **data):
            self.calls.append((api, data))
            return {}

    bot = FakeReactBot()
    service = SocialActionService(
        emoji_ids={"laugh": ["28", "101"]},
        per_user_cooldown_seconds=0,
        per_group_cooldown_seconds=0,
    )

    first = asyncio.run(
        service.react_to_message(bot, group_id=1, user_id=100, message_id=42, reaction="laugh", now=1000)
    )
    second = asyncio.run(
        service.react_to_message(bot, group_id=1, user_id=101, message_id=43, reaction="laugh", now=1010)
    )

    assert first.sent
    assert second.sent
    assert [call[1]["emoji_id"] for call in bot.calls] == ["28", "101"]
    assert "点了 laugh 表情" in service.recent_reaction_context(1)


def test_reply_decision_parser_accepts_react_action() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.8, "action": "react", '
        '"reaction": "laugh", "mode": "chat", "reason": "轻轻笑一下"}'
    )

    assert decision.should_reply
    assert decision.action == "react"
    assert decision.reaction == "laugh"
    assert decision.side_reaction == ""


def test_reply_decision_parser_keeps_text_action_for_side_reaction() -> None:
    decision = _parse_reply_decision(
        '{"should_reply": true, "confidence": 0.8, "action": "tease", '
        '"reaction": "laugh", "mode": "chat", "reason": "能接一句"}'
    )

    assert decision.should_reply
    assert decision.action == "tease"
    assert decision.reaction == ""
    assert decision.side_reaction == "laugh"


def test_parse_ocr_text_handles_common_payload_shapes() -> None:
    assert parse_ocr_text({"texts": [{"text": "第一行"}, {"text": "第二行"}]}) == "第一行 第二行"
    assert parse_ocr_text({"data": {"words_result": [{"words": "第三行"}]}}) == "第三行"
    assert parse_ocr_text({"result": ["第四行", {"content": "第五行"}]}) == "第四行 第五行"


def test_image_ocr_service_uses_ocr_image_and_builds_context() -> None:
    class FakeOcrBot:
        def __init__(self) -> None:
            self.calls = []

        async def call_api(self, api: str, **data):
            self.calls.append((api, data))
            assert api == "ocr_image"
            return {"texts": [{"text": "截图里的文字"}]}

    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="image", data={"url": "https://example.com/a.png", "file": "a.png"}),
        ],
    )
    service = ImageOcrService(max_images_per_message=2, max_calls_per_minute=10)
    context = asyncio.run(service.context_for_event(FakeOcrBot(), event))

    assert context.image_count == 1
    assert context.ocr_count == 1
    assert context.text == "第1张图：截图里的文字"


def test_image_ocr_service_times_out_slow_ocr_api() -> None:
    class SlowOcrBot:
        async def call_api(self, api: str, **data):
            await asyncio.sleep(0.2)
            return {"texts": [{"text": "太慢了不该等到"}]}

    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="image", data={"url": "https://example.com/slow.png"}),
        ],
    )
    service = ImageOcrService(
        max_images_per_message=1,
        max_calls_per_minute=10,
        api_timeout_seconds=0.01,
    )

    context = asyncio.run(service.context_for_event(SlowOcrBot(), event))

    assert context.image_count == 1
    assert context.ocr_count == 0
    assert context.skipped_reason == "empty_ocr"


def test_image_ocr_service_can_use_fallback_without_napcat_ocr() -> None:
    class RejectNapcatOcrBot:
        async def call_api(self, api: str, **data):
            raise AssertionError(f"NapCat OCR should not be called: {api}")

    class FakeFallbackOcr:
        def __init__(self) -> None:
            self.targets = []

        async def recognize(self, target: str) -> str:
            self.targets.append(target)
            return "fallback 识别文字"

    fallback = FakeFallbackOcr()
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="image", data={"url": "https://example.com/fallback.png"}),
        ],
    )
    service = ImageOcrService(
        max_images_per_message=1,
        max_calls_per_minute=10,
        napcat_ocr_enabled=False,
        fallback_ocr=fallback,
    )

    context = asyncio.run(service.context_for_event(RejectNapcatOcrBot(), event))

    assert fallback.targets == ["https://example.com/fallback.png"]
    assert context.image_count == 1
    assert context.ocr_count == 1
    assert context.text == "第1张图：fallback 识别文字"


def test_unreadable_image_with_ocr_context_can_enter_buffer() -> None:
    event = SimpleNamespace(
        message=[SimpleNamespace(type="image", data={"summary": "截图"})],
        get_plaintext=lambda: "",
    )

    assert not plugin._should_ignore_unreadable_media_event(
        event,
        forward_context="",
        readable_media_context="截图里的文字",
    )


def test_live_file_segment_becomes_readable_context() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(
                type="file",
                data={"name": "群资料.pdf", "file_id": "file-1", "file_size": "2048"},
            )
        ],
        reply=None,
    )

    assert plugin._message_context_text(event) == "[文件:群资料.pdf，2 KB]"
    assert plugin._message_has_context_media(event)


def test_inline_forward_nodes_are_read_without_onebot_fetch() -> None:
    class RejectForwardFetchBot:
        async def call_api(self, api: str, **data):
            raise AssertionError(f"inline forward must not call {api}: {data}")

    event = SimpleNamespace(
        message=[
            SimpleNamespace(
                type="forward",
                data={
                    "content": [
                        {
                            "type": "node",
                            "data": {
                                "user_id": "10001",
                                "nickname": "甲",
                                "content": [{"type": "text", "data": {"text": "第一句"}}],
                            },
                        },
                        {
                            "type": "node",
                            "data": {
                                "user_id": "10002",
                                "nickname": "乙",
                                "content": [{"type": "text", "data": {"text": "第二句"}}],
                            },
                        },
                    ]
                },
            )
        ]
    )

    context = asyncio.run(plugin._forward_context_text(RejectForwardFetchBot(), event, nickname="转发人"))

    assert "转发人传了聊天记录" in context
    assert "甲" in context and "第一句" in context
    assert "乙" in context and "第二句" in context
