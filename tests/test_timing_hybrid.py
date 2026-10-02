import asyncio
import json
from types import SimpleNamespace

import pytest

from qq_social_agent.deepseek_client import DeepSeekClient
from qq_social_agent.pipeline_types import OutputChannel
from qq_social_agent.prompts import PromptRegistry
from qq_social_agent.timing_gate import TimingDecision, choose_jev_group_timing


def choose(**values):
    defaults = dict(looks_like_question=False, followup_addressed=False,
                    wants_answer=0.1, to_other=0.1, silent=0.3,
                    answer=0.1, continue_bot=0.1, social=0.5, other=0.05)
    return choose_jev_group_timing(**(defaults | values))


def test_only_ambiguous_openings_need_llm_review():
    assert choose().review_required
    assert not choose(silent=0.7, social=0.1).review_required
    assert not choose(other=0.6).review_required
    strong = choose(looks_like_question=True, wants_answer=0.95, answer=0.9,
                    silent=0.05, social=0.01)
    assert strong.channel == OutputChannel.TEXT and not strong.review_required


def test_other_addressee_can_have_observer_opening():
    assert choose(to_other=0.9).review_required
    private = choose(to_other=0.9, silent=0.8, social=0.05)
    assert private.channel == OutputChannel.SILENT and not private.review_required


@pytest.mark.parametrize('to_other', [0.1, 0.9])
def test_weak_top_choice_is_not_worth_an_llm_call(to_other):
    weak = choose(to_other=to_other, social=0.4, silent=0.3)
    assert weak.channel == OutputChannel.SILENT
    assert not weak.review_required
    worthwhile = choose(to_other=to_other, social=0.5, silent=0.3)
    assert worthwhile.review_required


def test_confident_observer_opening_goes_straight_to_generation():
    strong = choose(to_other=0.95, social=0.9, silent=0.05)
    assert strong.channel == OutputChannel.TEXT
    assert not strong.review_required
    assert strong.to_reply_decision().action == 'reply'


def test_bound_continuation_does_not_need_another_vote():
    continuation = choose(followup_addressed=True, continue_bot=0.55, silent=0.3,
                          social=0.1, other=0.025, answer=0.025)
    assert continuation.channel == OutputChannel.TEXT
    assert not continuation.review_required
    assert choose(followup_addressed=False, continue_bot=0.55, silent=0.3,
                  social=0.1).channel == OutputChannel.SILENT


@pytest.mark.parametrize('sample,review', [(0.29, True), (0.30, False)])
def test_sampling_happens_before_llm_and_preserves_reply_angle(monkeypatch, sample, review):
    client = object.__new__(DeepSeekClient)
    client.config = SimpleNamespace(interjection_review_probability=0.3)
    client.prompts = PromptRegistry()
    client._try_jev = lambda *_args, **_kwargs: asyncio.sleep(0, result=choose())
    monkeypatch.setattr('qq_social_agent.deepseek_client.random.random', lambda: sample)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            dict(channel='text', intent='chat', confidence=0.8, reason='有具体接话点',
                 reply_angle='接住编译器报错的吐槽'))))])

    client._chat_completion = completion
    result = asyncio.run(client.timing_gate(persona=SimpleNamespace(name='风雪', decision_prompt=''),
        recent_messages=[], current_text='编译器又和我过不去', current_nickname='A'))
    assert bool(calls) == review
    if review:
        assert calls[0]['task'] == 'timing_review'
        assert result.reason.startswith('jev_llm_')
        assert result.to_reply_decision().reply_angle == '接住编译器报错的吐槽'
    else:
        assert result.channel == OutputChannel.SILENT


@pytest.mark.parametrize('decision', [choose(silent=0.8, social=0.1),
    choose(to_other=0.95, social=0.9, silent=0.05),
    choose(followup_addressed=True, continue_bot=0.7, silent=0.2)])
def test_clear_jev_decisions_bypass_sampling_and_llm(monkeypatch, decision):
    client = object.__new__(DeepSeekClient)
    client.config = SimpleNamespace(interjection_review_probability=0)
    client.prompts = PromptRegistry()
    client._try_jev = lambda *_args, **_kwargs: asyncio.sleep(0, result=decision)
    def unexpected(*_args, **_kwargs):
        raise AssertionError('clear decisions must bypass sampling and LLM review')
    monkeypatch.setattr('qq_social_agent.deepseek_client.random.random', unexpected)
    client._chat_completion = unexpected
    result = asyncio.run(client.timing_gate(persona=SimpleNamespace(name='风雪', decision_prompt=''),
        recent_messages=[], current_text='那换个编译器呢', current_nickname='A'))
    assert result is decision
