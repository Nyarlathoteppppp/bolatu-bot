"""Approval arbitration and module boundaries retain serialized selection."""

import asyncio
from types import SimpleNamespace

import nonebot

nonebot.init()

from qq_social_agent import plugin
from qq_social_agent.memory import MemoryStore


def _approval() -> plugin.PendingGroupApproval:
    return plugin.PendingGroupApproval(
        approval_id="concurrent-approval",
        group_id=1026813421,
        trigger_user_id=184589072,
        trigger_nickname="小鸟",
        trigger_text="没人理我",
        persona_name="张风雪",
        self_id=1801507496,
        candidates=(
            plugin.PendingApprovalCandidate(1, "第一条", "tease", "短句"),
            plugin.PendingApprovalCandidate(2, "第二条", "answer", "回答"),
        ),
        mention_targets={},
        created_at=1000.0,
    )


def test_concurrent_approval_choices_claim_one_pending_item(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "memory", MemoryStore(tmp_path / "bot.sqlite3"))
    plugin.pending_group_approvals.clear()
    plugin.approval_choice_cooldowns.clear()
    approval = _approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    entered_delivery = asyncio.Event()
    release_delivery = asyncio.Event()
    selected = []
    private_feedback = []

    async def deliver(bot, selected_approval, candidate, **kwargs):
        selected.append((selected_approval.approval_id, candidate.index, kwargs["high_quality"]))
        entered_delivery.set()
        await release_delivery.wait()

    async def private_text(bot, user_id, text):
        private_feedback.append((user_id, text))

    monkeypatch.setattr(plugin, "_send_approved_group_reply", deliver)
    monkeypatch.setattr(plugin, "_send_private_text", private_text)
    bot = SimpleNamespace(self_id=1801507496)

    async def run():
        first = asyncio.create_task(plugin._handle_group_approval_private(bot, 1535071184, "A!"))
        await entered_delivery.wait()
        second = await plugin._handle_group_approval_private(bot, 1535071184, "B!")
        release_delivery.set()
        assert await first
        return second

    second_handled = asyncio.run(run())
    assert second_handled
    assert selected == [(approval.approval_id, 1, True)]
    assert private_feedback == [(1535071184, "当前没有待审批候选。")]
    assert plugin.pending_group_approvals == {}
    plugin.approval_choice_cooldowns.clear()
