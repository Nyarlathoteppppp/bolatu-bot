"""Transport health and partial-delivery recovery without real QQ traffic."""
import asyncio
import time
from types import SimpleNamespace

import nonebot
import pytest
nonebot.init()

from qq_social_agent import plugin, observability
from qq_social_agent.approval_models import PendingApprovalCandidate, PendingGroupApproval
from qq_social_agent.delivery import DeliveryPlan
from qq_social_agent.memory import MemoryStore
from qq_social_agent.pipeline_types import PipelineState


@pytest.fixture
def isolated_health(monkeypatch):
    monkeypatch.setattr(observability, "_connected_bots", set())
    monkeypatch.setattr(observability, "_bot_connections", {})
    observability.mark_bot_connected("123")
    monkeypatch.setattr(plugin, "_status_db_health", lambda: (True, ""))
    monkeypatch.setattr(plugin, "deepseek_client", SimpleNamespace(clients={"p": object()}))


def test_reads_and_reconnect_do_not_heal_failed_send(isolated_health):
    observability.mark_onebot_api_error("123", "send_group_msg", error="1006514")
    observability.mark_onebot_api_success("123", "get_group_msg_history")
    observability.mark_bot_connected("123")
    result = plugin._http_ready_payload()
    assert result["onebot_ready"]
    assert not result["ok"]
    assert result["delivery"]["status"] == "failed"
    observability.mark_onebot_api_success("123", "send_private_msg")
    assert not plugin._http_ready_payload()["ok"]
    observability.mark_onebot_api_success("123", "send_group_msg")
    assert plugin._http_ready_payload()["ok"]


def test_private_success_does_not_verify_group_channel(isolated_health):
    observability.mark_onebot_api_success("123", "send_private_msg")
    result = plugin._http_ready_payload()
    assert result["ok"]
    assert result["delivery"]["status"] == "unverified"
    assert result["delivery"]["channels"]["123:send_group_msg"]["status"] == "unverified"
    assert result["delivery"]["channels"]["123:send_private_msg"]["status"] == "verified"


def test_send_timeout_is_unknown_and_idle_is_unverified(isolated_health):
    assert plugin._http_ready_payload()["delivery"]["status"] == "unverified"
    observability.mark_onebot_api_error("123", "send_private_msg", timeout=True)
    assert plugin._http_ready_payload()["delivery"]["status"] == "unknown"
    assert not plugin._http_ready_payload()["ok"]


@pytest.fixture
def delivery(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr(plugin, "memory", store)
    monkeypatch.setattr(plugin, "_record_metric_event", lambda *a, **kw: None)
    monkeypatch.setattr(plugin, "_record_user_reply", lambda *a, **kw: None)
    monkeypatch.setattr(plugin, "_record_bot_sent_message", lambda *a, **kw: None)
    monkeypatch.setattr(plugin, "_record_post_reply_followup_window", lambda *a, **kw: None)
    async def nothing(*a, **kw):
        pass
    async def prepare(text, **kw):
        return text, text, ()
    monkeypatch.setattr(plugin, "_prepare_group_political_send_texts", prepare)
    for name in ["_send_private_text", "_maybe_send_group_meme", "_execute_approved_side_reaction"]:
        monkeypatch.setattr(plugin, name, nothing)
    monkeypatch.setattr(plugin, "build_delivery_plan", lambda **kw: DeliveryPlan(
        parts=("第一段", "第二段"), mention_targets={}, sequence_lag=0, forced_trigger_mention=False))
    candidate = PendingApprovalCandidate(1, "第一段\n第二段", "reply", "")
    state = PipelineState(correlation_id="test", group_id=100, user_id=200,
                          nickname="A", text="测试", addressed=True)
    approval = PendingGroupApproval(
        approval_id="test", group_id=100, trigger_user_id=200, trigger_nickname="A",
        trigger_text="测试", persona_name="bot", self_id=123, candidates=(candidate,),
        mention_targets={}, created_at=time.time(), pipeline_state=state)
    return approval, candidate


def test_partial_reapproval_resumes_without_duplicate(monkeypatch, delivery):
    approval, candidate = delivery
    calls = []
    async def send(bot, group_id, message):
        calls.append(str(message))
        if len(calls) == 2:
            raise plugin.ActionFailed(status="failed", retcode=120, message="blocked")
        return len(calls)
    monkeypatch.setattr(plugin, "_send_group_message", send)
    async def run():
        for _ in range(3):
            await plugin._send_approved_group_reply_scoped(
                object(), approval, candidate, approver_id=None, high_quality=False, notify_success=False)
    asyncio.run(run())
    assert calls == ["第一段", "第二段", "第二段"]
    assert approval.pipeline_state.stage.value == "completed"
    assert len(approval.delivery_progress[candidate.text].sent_message_ids) == 2


@pytest.mark.parametrize("error", [asyncio.TimeoutError(), ConnectionResetError(),
    plugin.ActionFailed(status="failed", retcode=1200, message="Timeout: sendMsg")])
def test_uncertain_send_is_not_retried(monkeypatch, delivery, error):
    approval, candidate = delivery
    calls = []
    async def send(*args):
        calls.append(1)
        raise error
    monkeypatch.setattr(plugin, "_send_group_message", send)
    async def run():
        for _ in range(2):
            await plugin._send_approved_group_reply_scoped(
                object(), approval, candidate, approver_id=None, high_quality=False, notify_success=False)
    asyncio.run(run())
    assert len(calls) == 1
    assert approval.pipeline_state.stage.value == "failed"
    assert approval.delivery_progress[candidate.text].uncertain_index == 0


def test_connect_registers_tasks_even_if_reconciliation_fails(monkeypatch):
    calls = []
    names = ["_ensure_daily_review_task", "_ensure_weekly_usage_report_task", "_ensure_proactive_chat_task",
             "_ensure_private_guided_chat_task", "_ensure_private_hourly_chat_task",
             "_ensure_group_directory_task", "_ensure_history_backfill_task"]
    for name in names:
        monkeypatch.setattr(plugin, name, lambda bot, name=name: calls.append(name))
    monkeypatch.setattr(plugin, "_record_metric_event", lambda *a, **kw: None)
    monkeypatch.setattr(plugin, "mark_bot_connected", lambda *a: None)
    monkeypatch.setattr(plugin, "connected_onebot_bots", {})
    async def fail(*a, **kw):
        calls.append("network")
        raise asyncio.TimeoutError()
    async def ok(*a, **kw):
        calls.append("notice")
    for name in ["_reconcile_group_mutes", "_sync_group_status_cards"]:
        monkeypatch.setattr(plugin, name, fail)
    for name in ["_send_approval_rules_to_approvers", "_send_changelog_notice_to_approvers", "_notify_active_group_mutes"]:
        monkeypatch.setattr(plugin, name, ok)
    asyncio.run(plugin._send_approval_rules_on_connect(SimpleNamespace(self_id="123")))
    assert calls[:7] == names
    assert calls[-3:] == ["notice"] * 3


def test_reconnect_notice_has_cooldown_but_manual_switch_does_not(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "memory", MemoryStore(tmp_path / "bot.sqlite3"))
    monkeypatch.setattr(plugin, "_approval_user_ids", lambda: [200])
    calls = []
    async def send(*args, **kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(plugin, "_send_private_message", send)
    async def run():
        for reason in ["bot_connect", "bot_connect", "review_switch"]:
            await plugin._send_approval_rules_to_approvers(SimpleNamespace(self_id=123), reason=reason)
    asyncio.run(run())
    assert len(calls) == 2


def test_duplicate_approvals_are_serialized(monkeypatch, delivery):
    approval, candidate = delivery
    calls = []
    async def send(*args):
        calls.append(str(args[-1]))
        return len(calls)
    monkeypatch.setattr(plugin, "_send_group_message", send)
    async def run():
        await asyncio.gather(*(
            plugin._send_approved_group_reply_scoped(
                object(), approval, candidate, approver_id=None, high_quality=False, notify_success=False)
            for _ in range(2)
        ))
    asyncio.run(run())
    assert calls == ["第一段", "第二段"]


def test_cancelled_send_requires_verification_before_retry(monkeypatch, delivery):
    approval, candidate = delivery
    calls = []
    async def run():
        entered = asyncio.Event()
        async def send(*args):
            calls.append(1)
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(plugin, "_send_group_message", send)
        task = asyncio.create_task(plugin._send_approved_group_reply_scoped(
            object(), approval, candidate, approver_id=None, high_quality=False, notify_success=False))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await plugin._send_approved_group_reply_scoped(
            object(), approval, candidate, approver_id=None, high_quality=False, notify_success=False)
    asyncio.run(run())
    assert len(calls) == 1
    assert approval.delivery_progress[candidate.text].uncertain_index == 0
