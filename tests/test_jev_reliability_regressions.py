"""Regression cases: malformed decisions and partial critic failures."""
import asyncio
from types import SimpleNamespace

import pytest

from qq_social_agent import jev_client
from qq_social_agent.jev_client import JevClient
from qq_social_agent.deepseek_client import DeepSeekClient
from qq_social_agent.ellipsis_resolver import parse_jev_ellipsis_answers, apply_jev_ellipsis_judgement
from qq_social_agent.reference_resolver import (
    ReferentJudgement, ReferenceResolution, apply_jev_referent_judgement,
)
from qq_social_agent.pre_send_critic import apply_jev_critic_judgement, next_critic_action


@pytest.mark.parametrize("answer", [None, {}, {"noul": None}, {"noul": float("nan")},
                                    {"noul": float("inf")}, {"noul": False}, {"noul": 2}])
def test_bad_live_timing_observations_trigger_fallback(answer):
    client = JevClient(api_key="test")
    async def fake(**kwargs):
        return {"answers": {key: answer for key in kwargs["questions"]}}
    client.evaluate = fake
    with pytest.raises(ValueError):
        asyncio.run(client.timing_gate(
            persona=SimpleNamespace(decision_prompt=""), recent_messages=[],
            current_text="怎么修复网络？", current_nickname="A"))


@pytest.mark.parametrize("payload", [{}, {"answers": {}}, {"answers": {
    "ellipsis_kind": {"choice": "INVALID", "confidence": 0.9}}}])
def test_ellipsis_parse_error_remains_error(payload):
    result = apply_jev_ellipsis_judgement(parse_jev_ellipsis_answers(payload), [], current_text="那个呢")
    assert result.status == "ERROR"
    assert result.unresolved


@pytest.mark.parametrize("kind,confidence", [("NONE", 0.01), ("OTHER", 0.95)])
def test_negative_or_other_referent_stays_uncertain(kind, confidence):
    result = apply_jev_referent_judgement(
        ReferentJudgement(kind=kind, confidence=confidence), [],
        current_text="他在哪", fallback=ReferenceResolution())
    assert result.status == "AMBIGUOUS"
    assert not result.user_ids
    result = apply_jev_ellipsis_judgement(parse_jev_ellipsis_answers({"answers": {
        "ellipsis_kind": {"choice": kind, "confidence": confidence}}}), [], current_text="那个呢")
    assert result.status == "AMBIGUOUS"


@pytest.mark.parametrize("broken_branch", ["intent", "shared"])
@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_review_retains_other_branch_failure(monkeypatch, broken_branch, failure):
    monkeypatch.setattr(jev_client, "DRAFT_REVIEW_TIMEOUT_SECONDS", 0.03)
    client = JevClient(api_key="test")
    cancelled = []
    async def fake(**kwargs):
        branch = "intent" if "intent_covered" in kwargs["questions"] else "shared"
        if branch == broken_branch:
            if failure == "error":
                raise RuntimeError("offline")
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.append(branch)
        return {"answers": {
            key: {"choice": "YES" if key == "unsupported_claim" else "NO"}
            for key in kwargs["questions"]
        }}
    client.evaluate = fake
    wrapper = DeepSeekClient.__new__(DeepSeekClient)
    wrapper.jev_client = client
    _, judgement = asyncio.run(wrapper.review_draft(
        draft="测试草稿", current_text="测试问题", current_label="A", action="answer"))
    result = apply_jev_critic_judgement(judgement)
    assert result.failed
    assert result.unavailable
    assert next_critic_action(result, attempt=0) == "regenerate"
    assert next_critic_action(result, attempt=1) == "block"
    if failure == "timeout":
        assert cancelled == [broken_branch]


@pytest.mark.parametrize("method,key", [("should_reply", "should_reply"), ("route_tool", "need_tool")])
def test_missing_choice_with_valid_noul_is_not_a_silent_negative(method, key):
    client = JevClient(api_key="test")
    async def fake(**kwargs):
        return {"answers": {key: {"noul": 0.99}}}
    client.evaluate = fake
    with pytest.raises(ValueError):
        asyncio.run(getattr(client, method)(
            persona=SimpleNamespace(decision_prompt="", name="test"), recent_messages=[],
            current_text="查一下天气", current_nickname="A"))


def test_review_cancellation_propagates_and_cleans_tasks():
    client = JevClient(api_key="test")
    cancelled = []
    async def fake(**kwargs):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.append(True)
    client.evaluate = fake
    async def run():
        task = asyncio.create_task(client.review_draft(
            draft="草稿", current_text="问题", current_label="A", action="answer"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert len(cancelled) == 2
