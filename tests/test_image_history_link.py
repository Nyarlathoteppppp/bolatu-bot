from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import nonebot

nonebot.init()

from qq_social_agent import plugin
from qq_social_agent.discourse_state import resolve_group_discourse
from qq_social_agent.ellipsis_resolver import EllipsisJudgement
from qq_social_agent.history_sync import backfill_group_history
from qq_social_agent.media_context import ImageOcrService
from qq_social_agent.memory import MemoryStore


class HistoryBot:
    self_id = 999

    async def call_api(self, api: str, **data):
        assert api == "get_group_msg_history"
        assert data["group_id"] == 1
        return {
            "messages": [
                {
                    "message_id": 42,
                    "group_id": 1,
                    "user_id": 100,
                    "time": 1000,
                    "sender": {"user_id": 100, "nickname": "A"},
                    "message": [
                        {
                            "type": "image",
                            "data": {
                                "url": "https://example.com/cat.png",
                                "file": "cat.png",
                            },
                        }
                    ],
                }
            ]
        }


class NoApiBot:
    self_id = 999

    async def call_api(self, api: str, **data):
        raise AssertionError(f"Image URL should not call {api}")


class Vision:
    async def recognize(self, target: str) -> str:
        assert target == "https://example.com/cat.png"
        return "一只猫在窗边"


class Jev:
    async def resolve_ellipsis(self, **kwargs):
        return EllipsisJudgement(
            kind="ITEM_DEIXIS",
            inherit_from="s42",
            confidence=0.99,
        )


def test_backfilled_image_keeps_source_payload_and_resolves_for_followup(monkeypatch, tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    original_segments = [
        {
            "type": "image",
            "data": {
                "url": "https://example.com/cat.png",
                "file": "cat.png",
            },
        }
    ]

    assert asyncio.run(backfill_group_history(HistoryBot(), memory, 1, count=20, self_id=999)) == 1
    row = memory.admin_message_by_source(1, "42")
    assert row is not None
    assert row["source_message_id"] == "42"
    assert row["created_at"] == 1000
    assert json.loads(row["message_segments_json"]) == original_segments
    assert not memory.claim_inbound_message(1, "42")

    recent = memory.images.enrich(memory.recent_messages(1, 10))
    discourse = asyncio.run(
        resolve_group_discourse(
            current_text="这个图什么意思",
            current_user_id=200,
            current_nickname="B",
            self_id=999,
            recent_messages=recent,
            jev=Jev(),
        )
    )
    assert discourse.ellipsis.source_message_id == "42"

    monkeypatch.setattr(plugin, "memory", memory)
    monkeypatch.setattr(
        plugin,
        "image_ocr_service",
        ImageOcrService(napcat_ocr_enabled=False, primary_ocr=Vision()),
    )
    context = asyncio.run(
        plugin._resolved_image_context(NoApiBot(), discourse, recent, group_id=1)
    )
    assert "一只猫在窗边" in context
    assert "ready" in memory.images.enrich(memory.recent_messages(1, 10))[0].text


def test_backfill_fills_segments_on_existing_text_only_history_row(tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    assert memory.add_message(
        1,
        100,
        "A",
        "[图片]",
        created_at=900,
        source_message_id="42",
        source_kind="history",
        correlation_id="old-history-row",
    )
    previous = memory.admin_message_by_source(1, "42")
    assert previous is not None

    assert asyncio.run(backfill_group_history(HistoryBot(), memory, 1, count=20, self_id=999)) == 0

    refreshed = memory.admin_message_by_source(1, "42")
    assert refreshed is not None
    assert refreshed["id"] == previous["id"]
    assert refreshed["text"] == "[图片]"
    assert refreshed["created_at"] == 900
    assert refreshed["source_kind"] == "history"
    assert refreshed["correlation_id"] == "old-history-row"
    assert json.loads(refreshed["message_segments_json"]) == [
        {
            "type": "image",
            "data": {
                "url": "https://example.com/cat.png",
                "file": "cat.png",
            },
        }
    ]
    assert not memory.claim_inbound_message(1, "42")
