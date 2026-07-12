import asyncio
from types import SimpleNamespace

import pytest

from qq_social_agent import onebot_gateway
from qq_social_agent.media_context import (
    ImageOcrService,
    file_metadata_context_for_event,
    image_cache_key,
    image_segments_from_event,
    ocr_image_segments_from_event,
)
from qq_social_agent.message_segments import (
    file_metadata,
    format_file_size,
    is_marketface_segment,
    message_text_from_payload,
    segment_placeholder,
)


def test_message_segment_parser_distinguishes_marketface_from_image() -> None:
    marketface = {
        "file": "marketface",
        "file_id": "marketface-a",
        "summary": "捂脸",
    }

    assert is_marketface_segment("image", marketface)
    assert is_marketface_segment("mface", {"summary": "坏笑"})
    assert is_marketface_segment(
        "image",
        {"file": "actual-sticker.jpg", "sub_type": "1", "summary": "[动画表情]"},
    )
    assert not is_marketface_segment("image", {"file": "photo.jpg", "sub_type": "0"})
    assert segment_placeholder("image", marketface) == "[mface:捂脸]"
    assert segment_placeholder("image", marketface, language="zh") == "[表情包:捂脸]"
    assert segment_placeholder("image", {"summary": "聊天截图"}) == "[image:聊天截图]"


def test_marketface_cache_key_uses_unique_identifier() -> None:
    first = image_cache_key({"file": "marketface", "file_id": "marketface-a"})
    second = image_cache_key({"file": "marketface", "file_id": "marketface-b"})

    assert first == "file_id:marketface-a"
    assert second == "file_id:marketface-b"
    assert first != second
    assert image_cache_key({"file": "marketface"}) == ""


def test_marketface_segments_are_visible_but_not_sent_to_ocr() -> None:
    class RejectBot:
        async def call_api(self, api: str, **data):
            raise AssertionError(f"marketface must not call OneBot OCR: {api} {data}")

    class RejectFallback:
        async def recognize(self, target: str) -> str:
            raise AssertionError(f"marketface must not call fallback OCR: {target}")

    event = SimpleNamespace(
        message=[
            SimpleNamespace(
                type="image",
                data={"file": "marketface", "file_id": "a", "url": "https://example.com/a"},
            ),
            SimpleNamespace(type="mface", data={"summary": "捂脸"}),
        ]
    )
    service = ImageOcrService(
        max_calls_per_minute=10,
        napcat_ocr_enabled=False,
        fallback_ocr=RejectFallback(),
    )

    assert len(image_segments_from_event(event)) == 2
    assert ocr_image_segments_from_event(event) == []
    context = asyncio.run(service.context_for_event(RejectBot(), event))
    assert context.image_count == 0
    assert context.ocr_count == 0
    assert context.text == ""


def test_file_and_rich_message_placeholders_preserve_safe_metadata() -> None:
    file_data = {
        "name": "群友整理.pdf",
        "file_id": "file-1",
        "file_size": "2097152",
        "url": "https://example.com/private-download-token",
    }

    assert format_file_size(file_data["file_size"]) == "2 MB"
    assert file_metadata(file_data) == {"name": "群友整理.pdf", "size": "2 MB", "file_id": "file-1"}
    assert segment_placeholder("file", file_data) == "[file:群友整理.pdf，2 MB]"
    assert "private-download-token" not in segment_placeholder("file", file_data)
    assert segment_placeholder("record", {"name": "语音.amr", "file_size": 1024}) == "[voice:语音.amr，1 KB]"
    assert segment_placeholder("video", {"name": "现场.mp4", "file_size": 1024}) == "[video:现场.mp4，1 KB]"
    assert segment_placeholder("music", {"title": "晚风", "singer": "群友"}) == "[music:晚风 - 群友]"
    assert segment_placeholder("share", {"title": "一篇文章", "url": "https://example.com"}) == "[share:一篇文章]"
    assert segment_placeholder("location", {"title": "南京南站"}) == "[location:南京南站]"
    assert segment_placeholder("contact", {"type": "qq", "id": "10001"}) == "[contact:qq:10001]"


def test_json_card_summary_is_semantic_and_bounded() -> None:
    payload = (
        '{"app":"music","prompt":"群友分享了一首歌",'
        '"meta":{"music":{"title":"晚风","desc":"来自 QQ 音乐"}},'
        '"jumpUrl":"https://example.com/private-token"}'
    )

    placeholder = segment_placeholder("json", {"data": payload})

    assert placeholder == "[music:群友分享了一首歌 / 晚风]"
    assert "private-token" not in placeholder


def test_message_payload_uses_content_when_message_list_is_empty() -> None:
    payload = {
        "message": [],
        "content": [
            {"type": "text", "data": {"text": "文件在这里"}},
            {"type": "file", "data": {"name": "说明.txt", "file_size": "12"}},
        ],
    }

    assert message_text_from_payload(payload) == "文件在这里 [file:说明.txt，12 B]"


def test_get_file_uses_unified_gateway_and_updates_status() -> None:
    class FakeBot:
        def __init__(self) -> None:
            self.calls = []

        async def call_api(self, api: str, **data):
            self.calls.append((api, data))
            return {
                "data": {
                    "file_name": "群资料.pdf",
                    "file_size": "2048",
                    "url": "https://example.com/file",
                    "base64": "not-retained-in-status",
                }
            }

    bot = FakeBot()
    before = onebot_gateway.status_snapshot()["apis"].get("get_file", {}).get("calls", 0)

    result = asyncio.run(onebot_gateway.get_file(bot, "file-42"))
    snapshot = onebot_gateway.status_snapshot()

    assert bot.calls == [("get_file", {"file_id": "file-42"})]
    assert result["file_name"] == "群资料.pdf"
    assert result["file_size"] == "2048"
    assert snapshot["apis"]["get_file"]["calls"] == before + 1
    assert snapshot["apis"]["get_file"]["successes"] >= 1
    assert snapshot["apis"]["get_file"]["in_flight"] == 0
    assert "not-retained-in-status" not in str(snapshot)


def test_missing_live_file_metadata_can_be_completed_without_downloading() -> None:
    class FakeBot:
        async def call_api(self, api: str, **data):
            assert api == "get_file"
            assert data == {"file_id": "file-99"}
            return {"data": {"file_name": "补全资料.txt", "file_size": "1024", "url": "https://secret"}}

    event = SimpleNamespace(
        message=[SimpleNamespace(type="file", data={"file_id": "file-99"})],
    )

    context = asyncio.run(file_metadata_context_for_event(FakeBot(), event))

    assert context == "[文件:补全资料.txt，1 KB]"
    assert "secret" not in context


def test_unified_gateway_times_out_and_records_failure() -> None:
    class SlowBot:
        async def call_api(self, api: str, **data):
            await asyncio.sleep(0.05)
            return {"ok": True}

    api = "test_slow_gateway_call"
    before = onebot_gateway.status_snapshot()["apis"].get(api, {}).get("timeouts", 0)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(onebot_gateway.call_api(SlowBot(), api, timeout_seconds=0.001))

    snapshot = onebot_gateway.status_snapshot()
    assert snapshot["apis"][api]["timeouts"] == before + 1
    assert snapshot["apis"][api]["failures"] >= 1
    assert snapshot["apis"][api]["in_flight"] == 0
    assert snapshot["last_api"] == api
