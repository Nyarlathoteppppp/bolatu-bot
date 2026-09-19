from qq_social_agent.pronoun_guard import (
    apply_jev_pronoun_judgement,
    draft_has_person_pronoun,
    format_pronoun_feedback,
    parse_jev_pronoun_answers,
    pronoun_questions,
)
from qq_social_agent.resolver_result import NOT_APPLICABLE, RESOLVED, UNAVAILABLE


def test_no_pronoun_skips_jev() -> None:
    assert draft_has_person_pronoun("CMU 难申") is False
    result = apply_jev_pronoun_judgement(None, has_pronoun=False)
    assert result.status == NOT_APPLICABLE
    assert result.needs_fix is False


def test_uncertain_pronoun_passes() -> None:
    judged = parse_jev_pronoun_answers(
        {
            "answers": {
                "pronoun_accurate": {"noul": 0.46},
                "pronoun_issue": {"choice": "wrong_you"},
            }
        },
        has_pronoun=True,
    )
    result = apply_jev_pronoun_judgement(judged, has_pronoun=True)
    assert result.status == RESOLVED
    assert result.needs_fix is False
    assert format_pronoun_feedback(result) == ""


def test_certain_wrong_you_needs_fix() -> None:
    judged = parse_jev_pronoun_answers(
        {
            "answers": {
                "pronoun_accurate": {"noul": 0.12},
                "pronoun_issue": {"choice": "wrong_you"},
            }
        },
        has_pronoun=True,
    )
    result = apply_jev_pronoun_judgement(judged, has_pronoun=True)
    assert result.needs_fix is True
    feedback = format_pronoun_feedback(result)
    assert "wrong_you" in feedback
    assert "只改人称" in feedback


def test_other_or_unavailable_does_not_block() -> None:
    judged = parse_jev_pronoun_answers(
        {
            "answers": {
                "pronoun_accurate": {"noul": 0.10},
                "pronoun_issue": {"choice": "other"},
            }
        },
        has_pronoun=True,
    )
    result = apply_jev_pronoun_judgement(judged, has_pronoun=True)
    assert result.needs_fix is False
    unavailable = apply_jev_pronoun_judgement(None, has_pronoun=True)
    assert unavailable.status == UNAVAILABLE
    assert unavailable.needs_fix is False


def test_pronoun_questions_are_orthogonal() -> None:
    questions = pronoun_questions()
    assert set(questions) == {"pronoun_accurate", "pronoun_issue"}
    assert "other" in questions["pronoun_issue"]["criteria"]
