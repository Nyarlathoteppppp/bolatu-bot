from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from qq_social_agent.daily_review_scheduler_service import DailyReviewSchedulerService
from qq_social_agent.daily_review_service import (
    DailyReviewDeliveryServices,
    DailyReviewPolicy,
    DailyReviewService,
    DailyReviewServices,
)
from qq_social_agent.memory import MemoryStore


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass


class _BatchService:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict, str]] = []
        self.result = SimpleNamespace(status="queued", content=None, error="")
        self.acknowledged: list[str] = []

    async def submit(self, key: str, body: dict, *, task: str) -> str:
        self.submissions.append((key, body, task))
        return "batch-1"

    async def poll(self, _key: str):
        return self.result

    def acknowledge(self, key: str) -> None:
        self.acknowledged.append(key)


class _Client:
    def __init__(self) -> None:
        self.draft_calls = 0
        self.parse_calls: list[tuple[str, list, int]] = []

    def build_daily_review_request(self, **kwargs) -> dict:
        return {"messages": [{"role": "user", "content": kwargs["today_label"]}]}

    def parse_daily_review_response(self, content: str, *, messages: list, max_chars: int):
        self.parse_calls.append((content, messages, max_chars))
        return SimpleNamespace(
            public_reply="批处理复盘。",
            events=[],
            member_changes=[],
            jargon_candidates=[],
            feedback_lessons=[],
            style_observations=[],
        )

    async def daily_review_draft(self, **_kwargs):
        self.draft_calls += 1
        return SimpleNamespace(
            public_reply="同步兜底复盘。",
            events=[],
            member_changes=[],
            jargon_candidates=[],
            feedback_lessons=[],
            style_observations=[],
        )


class _Config:
    allowed_groups = {7}
    groups = {"7": {"enabled": True}}
    default_persona = "p"

    @staticmethod
    def group_allowed(group_id: int) -> bool:
        return group_id == 7

    @staticmethod
    def group_config(_group_id: int) -> dict:
        return {"enabled": True}


def _services(memory: MemoryStore, batch: _BatchService, client: _Client, sent: list[str]) -> DailyReviewServices:
    async def prepare(text: str, *, context: str):
        return text, text, []

    async def send_group(_bot, _group_id: int, message) -> int:
        sent.append(str(message))
        return 456

    async def notify(**_kwargs) -> None:
        pass

    return DailyReviewServices(
        memory=memory,
        app_config=_Config(),
        get_deepseek_client=lambda: client,
        get_persona=lambda _persona_id: SimpleNamespace(name="风雪"),
        refresh_self_mute_state_if_stale=lambda *_args: asyncio.sleep(0, result=0.0),
        persist_learning=lambda *_args, **_kwargs: [],
        sanitize_generated_text=lambda text: text,
        split_reply_messages=lambda text, **_kwargs: [text] if text else [],
        delivery=DailyReviewDeliveryServices(
            prepare_political_send_texts=prepare,
            send_group_message=send_group,
            notify_owner_political_gag=notify,
            record_bot_sent_message=lambda *_args, **_kwargs: None,
            record_metric_event=lambda *_args, **_kwargs: None,
            action_failed_summary=lambda exc: str(exc),
            short_notice_text=lambda text, limit: text[:limit],
            logger=_Logger(),
        ),
        review_batch_enabled=lambda: True,
        review_batch_service=batch,
    )


def _policy() -> DailyReviewPolicy:
    return DailyReviewPolicy(
        timezone=ZoneInfo("UTC"),
        hour=0,
        minute=0,
        message_limit=100,
        respect_mute=False,
    )


def test_scheduled_review_persists_batch_and_resumes_after_restart(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "review.sqlite3")
    batch = _BatchService()
    client = _Client()
    sent: list[str] = []
    services = _services(memory, batch, client, sent)
    now = datetime(2026, 7, 14, 1, tzinfo=timezone.utc).timestamp()
    bot = SimpleNamespace(self_id=1801507496)
    service = DailyReviewService()

    assert not asyncio.run(service.send_due_reviews(bot, now=now, policy=_policy(), services=services))
    assert len(batch.submissions) == 1
    key, _body, task = batch.submissions[0]
    assert key == "daily_review:7:2026-07-13"
    assert task == "daily_review"
    assert sent == []

    batch.result = SimpleNamespace(status="completed", content='{"public_reply":"批处理复盘。"}', error="")
    restarted_service = DailyReviewService()
    assert not asyncio.run(
        restarted_service.poll_pending_reviews(
            bot,
            now=now + 8 * 3600,
            policy=_policy(),
            services=services,
        )
    )

    assert sent == ["批处理复盘。"]
    assert client.parse_calls[0][0].startswith("{")
    assert batch.acknowledged == [key]
    assert memory.app_kv_get(DailyReviewService.sent_key(7, "2026-07-13")) == "sent"
    assert memory.app_kv_get(DailyReviewService.batch_state_key(7, "2026-07-13")) == ""


def test_failed_batch_uses_sync_review_fallback_outside_catch_up(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "review.sqlite3")
    batch = _BatchService()
    client = _Client()
    sent: list[str] = []
    services = _services(memory, batch, client, sent)
    now = datetime(2026, 7, 14, 1, tzinfo=timezone.utc).timestamp()
    bot = SimpleNamespace(self_id=1801507496)
    service = DailyReviewService()

    assert not asyncio.run(service.send_due_reviews(bot, now=now, policy=_policy(), services=services))
    batch.result = SimpleNamespace(status="failed", content=None, error="provider error")

    assert not asyncio.run(
        service.poll_pending_reviews(
            bot,
            now=now + 8 * 3600,
            policy=_policy(),
            services=services,
        )
    )

    assert sent == ["同步兜底复盘。"]
    assert client.draft_calls == 1
    assert batch.acknowledged == ["daily_review:7:2026-07-13"]
    assert memory.app_kv_get(DailyReviewService.sent_key(7, "2026-07-13")) == "sent"


def test_rejected_batch_submit_uses_sync_fallback(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "review.sqlite3")
    batch = _BatchService()
    client = _Client()
    sent: list[str] = []
    services = _services(memory, batch, client, sent)
    now = datetime(2026, 7, 14, 1, tzinfo=timezone.utc).timestamp()
    bot = SimpleNamespace(self_id=1801507496)
    service = DailyReviewService()

    async def reject(_key: str, _body: dict, *, task: str) -> str:
        raise RuntimeError("HTTP 400")

    batch.submit = reject
    assert not asyncio.run(service.send_due_reviews(bot, now=now, policy=_policy(), services=services))
    assert not asyncio.run(service.poll_pending_reviews(bot, now=now + 8 * 3600, policy=_policy(), services=services))
    assert sent == ["同步兜底复盘。"]
    assert memory.app_kv_get(DailyReviewService.sent_key(7, "2026-07-13")) == "sent"


def test_uncertain_submission_falls_back_without_resubmitting(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "review.sqlite3")
    batch = _BatchService()
    client = _Client()
    sent: list[str] = []
    services = _services(memory, batch, client, sent)
    now = datetime(2026, 7, 14, 1, tzinfo=timezone.utc).timestamp()
    bot = SimpleNamespace(self_id=1801507496)
    service = DailyReviewService()

    assert not asyncio.run(service.send_due_reviews(bot, now=now, policy=_policy(), services=services))
    batch.result = SimpleNamespace(status="submission_uncertain", content=None, error="lost response")
    assert not asyncio.run(service.poll_pending_reviews(bot, now=now + 8 * 3600, policy=_policy(), services=services))
    assert len(batch.submissions) == 1
    assert sent == ["同步兜底复盘。"]


def test_daily_review_scheduler_polls_pending_jobs_outside_catch_up(monkeypatch) -> None:
    calls: list[str] = []

    async def poll(_bot, _now: float) -> bool:
        calls.append("poll")
        return True

    async def send(_bot, _now: float) -> bool:
        calls.append("send")
        return False

    async def stop_after_tick(_delay: float) -> None:
        raise asyncio.CancelledError

    scheduler = DailyReviewSchedulerService(
        timezone=ZoneInfo("UTC"),
        hour=0,
        minute=0,
        catch_up_seconds=60,
        retry_seconds=5,
        poll_seconds=300,
        send_due_reviews=send,
        poll_pending_reviews=poll,
        record_metric_event=lambda *_args, **_kwargs: None,
        logger=_Logger(),
        summarize_error=lambda text, limit: text[:limit],
    )
    monkeypatch.setattr("qq_social_agent.daily_review_scheduler_service.time.time", lambda: 1_800_000_000.0)
    monkeypatch.setattr("qq_social_agent.daily_review_scheduler_service.asyncio.sleep", stop_after_tick)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scheduler._run(SimpleNamespace(self_id="test"), "test"))

    assert calls == ["poll"]
