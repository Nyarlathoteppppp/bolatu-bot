"""Unknown Jev observations must not become confident negative decisions."""
import asyncio
from types import SimpleNamespace

import pytest

from qq_social_agent.jev_client import JevClient
from qq_social_agent.deepseek_client import DeepSeekClient
from qq_social_agent.pronoun_guard import apply_jev_pronoun_judgement, parse_jev_pronoun_answers


BAD_ANSWERS = [None, {}, {"noul": None}, {"noul": "oops"}, {"noul": float("nan")},
               {"noul": float("inf")}, {"noul": -1}, {"noul": 2}, {"noul": False}]


@pytest.mark.parametrize("answer", BAD_ANSWERS)
def test_unknown_audit_and_search_pass(answer):
    client = JevClient(api_key="test")

    async def evaluate(**kwargs):
        return {"answers": {key: answer for key in kwargs["questions"]}}

    client.evaluate = evaluate
    assert asyncio.run(client.audit_proactive_reply(
        persona=None, recent_messages=[], candidate="候选"))[0] is True
    assert asyncio.run(client.judge_search_useful(query="问题", evidence="证据"))[0] is True


@pytest.mark.parametrize("answer", BAD_ANSWERS)
def test_invalid_pronoun_score_is_error_not_incorrect(answer):
    judged = parse_jev_pronoun_answers({"answers": {
        "pronoun_accurate": answer, "pronoun_issue": {"choice": "wrong_you"},
    }}, has_pronoun=True)
    result = apply_jev_pronoun_judgement(judged, has_pronoun=True)
    assert result.status == "ERROR"
    assert not result.needs_fix


@pytest.mark.parametrize("score,calls,fix", [(0.28, 2, True), (0.281, 1, False), (0.9, 1, False)])
def test_pronoun_two_step_threshold(score, calls, fix):
    client = JevClient(api_key="test")
    seen = []

    async def evaluate(**kwargs):
        seen.append(kwargs)
        key = next(iter(kwargs["questions"]))
        assert len(kwargs["questions"]) == 1
        return {"answers": {key: {"noul": score} if key == "pronoun_accurate" else {"choice": "wrong_you"}}}

    client.evaluate = evaluate
    judged = asyncio.run(client.check_pronoun(draft="你刚说过", current_text="啥", current_label="A"))
    assert len(seen) == calls
    if calls == 2:
        assert seen[0]["state"] == seen[1]["state"]
        assert list(seen[1]["questions"]) == ["pronoun_issue"]
    assert apply_jev_pronoun_judgement(judged, has_pronoun=True).needs_fix is fix


def test_no_pronoun_no_call():
    client = JevClient(api_key="test")

    async def evaluate(**kwargs):
        pytest.fail("no pronoun must skip Jev")

    client.evaluate = evaluate
    asyncio.run(client.check_pronoun(draft="可以试试重启", current_text="咋办", current_label="A"))


@pytest.mark.parametrize("issue", [None, {}, {"choice": "other"}, {"choice": "unknown"}])
def test_uncertain_second_pronoun_step_passes(issue):
    client = JevClient(api_key="test")

    async def evaluate(**kwargs):
        if "pronoun_accurate" in kwargs["questions"]:
            return {"answers": {"pronoun_accurate": {"noul": 0.1}}}
        return {"answers": {"pronoun_issue": issue}}

    client.evaluate = evaluate
    judged = asyncio.run(client.check_pronoun(draft="我刚说过", current_text="啥", current_label="A"))
    assert not apply_jev_pronoun_judgement(judged, has_pronoun=True).needs_fix


def test_pronoun_wrapper_reaches_jev_and_failure_passes():
    client = DeepSeekClient.__new__(DeepSeekClient)
    calls = []

    async def check(**kwargs):
        calls.append(kwargs)
        return "observed"

    client.jev_client = SimpleNamespace(available=True, check_pronoun=check)
    assert asyncio.run(client.check_pronoun(draft="你")) == "observed"
    assert calls == [{"draft": "你"}]

    async def broken(**kwargs):
        raise RuntimeError("offline")

    client.jev_client.check_pronoun = broken
    assert asyncio.run(client.check_pronoun(draft="你")) is None


@pytest.mark.parametrize("action", ["answer", "reply", "ask_back"])
def test_addressed_ask_back_cannot_bypass_high_threshold(action):
    client = JevClient(api_key="test")

    async def evaluate(**kwargs):
        return {"answers": {"ask_back": {"noul": 0.89}}}

    client.evaluate = evaluate
    assert not asyncio.run(client.should_ask_back(current_text="怎么修？", action=action, addressed=True))


def test_review_draft_isolates_intent_then_drills_pronoun():
    client = JevClient(api_key="test")
    seen = []

    async def evaluate(**kwargs):
        seen.append((list(kwargs["questions"]), kwargs["state"]))
        if list(kwargs["questions"]) == ["intent_covered"]:
            return {"answers": {
                "intent_covered": {"choice": "YES"},
            }}
        if "referent_consistent" in kwargs["questions"]:
            return {"answers": {
                "referent_consistent": {"choice": "YES"},
                "context_consistent": {"choice": "YES"},
                "unsupported_claim": {"choice": "NO"},
                "pronoun_accurate": {"noul": 0.12},
            }}
        return {"answers": {"pronoun_issue": {"choice": "wrong_you"}}}

    client.evaluate = evaluate
    pronoun, critic = asyncio.run(client.review_draft(
        draft="你刚说过", current_text="啥", current_label="A", action="reply"))
    assert seen[0][0] == ["intent_covered"]
    assert "【近期聊天】" not in seen[0][1]
    assert seen[1][0][:3] == ["referent_consistent", "context_consistent", "unsupported_claim"]
    assert "pronoun_accurate" in seen[1][0]
    assert seen[2][0] == ["pronoun_issue"]
    assert critic.intent_covered == "YES"
    assert apply_jev_pronoun_judgement(pronoun, has_pronoun=True).needs_fix is True


def test_audit_unavailable_does_not_call_llm():
    client = DeepSeekClient.__new__(DeepSeekClient)

    async def unavailable(*args, **kwargs):
        return None

    async def completion(**kwargs):
        pytest.fail("unavailable Jev audit must not launch another judge")

    client._try_jev = unavailable
    client._chat_completion = completion
    assert asyncio.run(client.audit_proactive_reply(
        persona=None, recent_messages=[], candidate="候选")) == (True, "jev_unavailable_audit_pass")


@pytest.mark.parametrize("text", ["去哪", "谁", "为啥", "咋办", "？"])
def test_short_questions_are_not_hard_silenced(text):
    client = JevClient(api_key="test")
    calls = []

    async def evaluate(**kwargs):
        calls.append(kwargs)
        return {"answers": {
            "wants_answer": {"noul": 0.95}, "needs_care": {"noul": 0.0},
            "to_other": {"noul": 0.0},
            "timing_route": {"probabilities": {"silent": 0.1, "answer": 0.8, "continue_bot": 0.0, "social_join": 0.1}},
        }}

    client.evaluate = evaluate
    timing = asyncio.run(client.timing_gate(
        persona=SimpleNamespace(decision_prompt=""), recent_messages=[],
        current_text=text, current_nickname="A"))
    assert len(calls) == 1
    assert timing.channel.value == "text"
