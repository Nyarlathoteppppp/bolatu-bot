from qq_social_agent.jev_client import JevClient, OPENROUTER_JEV_MODEL, TYPESAFE_JEV_MODEL
from qq_social_agent.jev_policy import (
    GENERATION_RESERVE_SECONDS,
    GROUP_REPLY_BUDGET_SECONDS,
    GroupReplyBudget,
    JEV_MODEL_VERSION,
    OPENROUTER_JEV_MODEL as PINNED_OPENROUTER,
    TYPESAFE_JEV_MODEL as PINNED_TYPESAFE,
)


def test_pinned_models_are_versioned_not_latest():
    assert JEV_MODEL_VERSION == "1.13.0"
    assert PINNED_TYPESAFE == "jev-1.13.0"
    assert PINNED_OPENROUTER == "typesafe/jev-1.13"
    assert "latest" not in PINNED_TYPESAFE
    assert "latest" not in PINNED_OPENROUTER
    assert TYPESAFE_JEV_MODEL == PINNED_TYPESAFE
    assert OPENROUTER_JEV_MODEL == PINNED_OPENROUTER


def test_client_defaults_to_pinned_typesafe_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "direct")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert JevClient().model == "jev-1.13.0"


def test_budget_skips_optional_stages_but_keeps_generation_reserve():
    now = 1000.0
    budget = GroupReplyBudget.start(now, seconds=GROUP_REPLY_BUDGET_SECONDS)
    assert not budget.skip("ask_back", now)
    late = budget.deadline - GENERATION_RESERVE_SECONDS - 1.0
    assert budget.skip("ask_back", late)
    assert budget.skip("speaking_action", late)
    assert budget.skip("optional_rag", late)
    assert budget.skip("critic_retry", late)
    assert budget.remaining(late) >= 0
    assert "ask_back" in budget.skipped
