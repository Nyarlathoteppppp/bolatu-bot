from __future__ import annotations

from qq_social_agent.discourse_effects import RepairResolution
from qq_social_agent.ellipsis_resolver import EllipsisResolution
from qq_social_agent.pre_send_critic import (
    CriticJudgement,
    apply_jev_critic_judgement,
    critic_blocks_memory,
    critic_needs_clarify,
    critic_questions,
    format_critic_feedback,
    format_critic_jev_state,
    next_critic_action,
    parse_jev_critic_answers,
)
from qq_social_agent.reference_resolver import ReferenceResolution
from qq_social_agent.resolver_result import ERROR, NOT_APPLICABLE, RESOLVED, UNAVAILABLE


BIRD = 184589072
WANG = 2001


def test_critic_statuses_are_distinct() -> None:
    unavailable = apply_jev_critic_judgement(None)
    assert unavailable.status == UNAVAILABLE
    assert unavailable.status != NOT_APPLICABLE
    assert next_critic_action(unavailable, attempt=0) == "send"

    error = apply_jev_critic_judgement(CriticJudgement(intent_covered="MAYBE"))
    assert error.status == ERROR
    assert next_critic_action(error, attempt=0) == "send"

    passed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="YES",
            context_consistent="YES",
            unsupported_claim="NO",
        )
    )
    assert passed.status == RESOLVED
    assert passed.failed is False
    assert next_critic_action(passed, attempt=0) == "send"


def test_wrong_referent_fails_and_regenerates_once() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="NO",
            context_consistent="YES",
            unsupported_claim="NO",
        )
    )
    assert failed.failed is True
    assert "referent_consistent" in failed.failures
    assert next_critic_action(failed, attempt=0) == "regenerate"
    assert next_critic_action(failed, attempt=1) == "block"
    feedback = format_critic_feedback(failed)
    assert "referent_consistent=NO" in feedback
    assert "不要沿用被纠正前的对象" in feedback


def test_context_conflict_fails() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="YES",
            context_consistent="NO",
            unsupported_claim="NO",
        )
    )
    assert failed.failed is True
    assert "context_consistent" in failed.failures
    assert critic_blocks_memory(failed) is True


def test_unsupported_claim_only_uses_local_sources() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="YES",
            context_consistent="YES",
            unsupported_claim="YES",
        )
    )
    assert failed.failed is True
    assert failed.failures == ("unsupported_claim",)
    questions = critic_questions()
    assert "不要判断外部世界真假" in questions["unsupported_claim"]["instructions"]
    state = format_critic_jev_state(
        draft="小鸟去了MIT",
        current_text="不是小鸟，是小王",
        action="answer",
        memory_context="小王准备考研",
        tool_context="",
        reference=ReferenceResolution((WANG,), reason="repair_referent", kind="PERSON", status=RESOLVED),
        ellipsis=EllipsisResolution(status=NOT_APPLICABLE),
        repair=RepairResolution(kind="REFERENT", status=RESOLVED, replacement_user_ids=(WANG,)),
    )
    assert state.startswith("【待发送草稿】")
    assert "【当前消息】" in state
    assert "【约束】" in state.split("【待发送草稿】")[-1]
    assert "外部世界" in state


def test_first_fail_second_pass_sends_second() -> None:
    first = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="NO",
            context_consistent="YES",
            unsupported_claim="NO",
        )
    )
    second = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="YES",
            context_consistent="YES",
            unsupported_claim="NO",
        )
    )
    assert next_critic_action(first, attempt=0) == "regenerate"
    assert next_critic_action(second, attempt=1) == "send"
    assert second.failed is False


def test_retry_at_most_once() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="NO",
            referent_consistent="NO",
            context_consistent="NO",
            unsupported_claim="YES",
        )
    )
    assert next_critic_action(failed, attempt=0) == "regenerate"
    assert next_critic_action(failed, attempt=1) == "block"
    assert next_critic_action(failed, attempt=2) == "block"


def test_critic_questions_are_self_contained_and_atomic() -> None:
    questions = critic_questions()
    assert list(questions) == [
        "intent_covered",
        "referent_consistent",
        "context_consistent",
        "unsupported_claim",
    ]
    for key, question in questions.items():
        text = question["instructions"]
        assert key not in text
        assert "并且" not in text.replace("也不要", "")
        assert "但是" not in text
        assert "不要判断其他项" in text or key != "intent_covered"
        assert question["type"] == "choice"
        assert "other" in question["criteria"]
        assert "not_applicable" in question["criteria"]
    intent = questions["intent_covered"]
    assert "针对缺口追问算已处理" in intent["instructions"]
    assert "形成直接回应就算已处理" in intent["instructions"]
    assert "拒绝理由直接对应请求内容" in intent["instructions"]
    assert "只谈另一个对象算未处理" in intent["instructions"]
    assert "缺少的必要信息进行追问" in intent["criteria"]["covered"]
    assert "非核心近似数字" in questions["unsupported_claim"]["instructions"]


def test_critic_only_blocks_high_failure_probability() -> None:
    judged = parse_jev_critic_answers({"answers": {
        "intent_covered": {"choice": "missed", "probabilities": {"missed": 0.90}},
        "referent_consistent": {"choice": "conflict", "probabilities": {"conflict": 0.62}},
        "context_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.10}},
        "unsupported_claim": {"choice": "supported", "probabilities": {"unsupported": 0.30}},
    }})
    result = apply_jev_critic_judgement(judged)
    assert result.failures == ("intent_covered",)
    assert result.uncertain == ("referent_consistent",)
    assert dict(judged.failure_probabilities)["intent_covered"] == 0.90


def test_critic_keeps_borderline_intent_uncertain() -> None:
    judged = parse_jev_critic_answers({"answers": {
        "intent_covered": {"choice": "missed", "probabilities": {"missed": 0.82}},
        "referent_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.02}},
        "context_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.02}},
        "unsupported_claim": {"choice": "not_applicable", "probabilities": {"unsupported": 0.01}},
    }})
    result = apply_jev_critic_judgement(judged)
    assert result.failed is False
    assert result.uncertain == ("intent_covered",)


def test_answer_action_uses_stricter_intent_threshold() -> None:
    payload = {"answers": {
        "intent_covered": {"choice": "missed", "probabilities": {"missed": 0.82}},
        "referent_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.02}},
        "context_consistent": {"choice": "consistent", "probabilities": {"conflict": 0.02}},
        "unsupported_claim": {"choice": "not_applicable", "probabilities": {"unsupported": 0.01}},
    }}
    answer = apply_jev_critic_judgement(parse_jev_critic_answers(payload, action="answer"))
    tease = apply_jev_critic_judgement(parse_jev_critic_answers(payload, action="tease"))
    assert answer.failures == ("intent_covered",)
    assert tease.failures == ()
    assert tease.uncertain == ("intent_covered",)


def test_addressed_unsupported_claim_rewrites_once_then_sends() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="YES",
            context_consistent="YES",
            unsupported_claim="YES",
        )
    )
    assert next_critic_action(failed, attempt=0, addressed=True) == "regenerate"
    assert next_critic_action(failed, attempt=1, addressed=True) == "send"
    assert next_critic_action(failed, attempt=1, addressed=False) == "block"
    assert "点名也必须出声" in format_critic_feedback(failed)
    assert critic_needs_clarify(failed) is False


def test_addressed_referent_conflict_still_sends_after_retry() -> None:
    failed = apply_jev_critic_judgement(
        CriticJudgement(
            intent_covered="YES",
            referent_consistent="NO",
            context_consistent="NO",
            unsupported_claim="NO",
        )
    )
    assert next_critic_action(failed, attempt=0, addressed=True) == "regenerate"
    assert next_critic_action(failed, attempt=1, addressed=True) == "send"
    assert critic_needs_clarify(failed) is True
