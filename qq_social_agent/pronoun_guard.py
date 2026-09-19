from __future__ import annotations

import re
import math
from dataclasses import dataclass

from .resolver_result import ERROR, NOT_APPLICABLE, RESOLVED, SOURCE_JEV, SOURCE_RULE, UNAVAILABLE, finalize_result

YES = "YES"
NO = "NO"
PRONOUN_ACCURATE_CHOICES = (YES, NO)
PRONOUN_ISSUE_CHOICES = (
    "none",
    "wrong_you",
    "wrong_he",
    "speaker_target_swap",
    "attributed_wrong_speech",
    "other",
)
_PRONOUN_RE = re.compile(r"(你们|他们|她们|咱们|我们|你|您|他|她|咱|我)")


@dataclass(frozen=True)
class PronounJudgement:
    has_pronoun: bool = False
    accurate: str = ""
    noul: float = 0.0
    issue: str = "none"
    reason: str = ""


@dataclass(frozen=True)
class PronounResult:
    has_pronoun: bool = False
    accurate: str = ""
    issue: str = "none"
    noul: float = 0.0
    status: str = ""
    source: str = ""
    value: str = ""
    reason: str = "none"
    needs_fix: bool = False

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if not self.has_pronoun:
                status = NOT_APPLICABLE
            elif "jev_unavailable" in reason:
                status = UNAVAILABLE
            elif reason == "error" or reason.endswith("_error"):
                status = ERROR
            else:
                status = RESOLVED
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(
                self,
                "source",
                SOURCE_JEV if status in {RESOLVED, UNAVAILABLE} else SOURCE_RULE,
            )
        object.__setattr__(self, "needs_fix", bool(self.has_pronoun and status == RESOLVED and self.accurate == NO and self.issue not in {"", "none", "other"} and self.noul <= 0.28))
        finalize_result(self, has_value=status == RESOLVED)


def draft_has_person_pronoun(text: str) -> bool:
    return bool(_PRONOUN_RE.search(text or ""))


def pronoun_issue_criteria() -> dict[str, str]:
    return {
        "none": "人称没有问题，或草稿里没有在指群里的人",
        "wrong_you": "草稿里的「你」指错人：当前触发人不是这个「你」，或把旁人的话当成当前触发人说的",
        "wrong_he": "草稿里的「他/她/他们」指错人",
        "speaker_target_swap": "把当前发言人和被回复对象搞反了",
        "attributed_wrong_speech": "把旁人刚说的内容安到当前触发人身上",
        "other": "不像上面几类，或吃不准",
    }


def format_pronoun_jev_state(
    *,
    draft: str,
    current_text: str,
    current_label: str,
    speaker_context: str = "",
    recent_messages: list | tuple | None = None,
) -> str:
    lines = [
        f"【待发送草稿】{(draft or '')[:400]}",
        f"【当前触发人】{current_label}",
        f"【当前消息】{(current_text or '')[:240]}",
    ]
    if speaker_context.strip():
        lines.append("【说话关系】")
        lines.append(speaker_context.strip()[:800])
    history = []
    for msg in list(recent_messages or [])[-8:]:
        nick = getattr(msg, "nickname", None) or str(getattr(msg, "user_id", ""))
        prefix = "风雪" if getattr(msg, "is_bot", False) else nick
        history.append(f"{prefix}: {getattr(msg, 'text', '')}")
    if history:
        lines.append("【近期聊天，也就是回复时看到的上下文】")
        lines.extend(history)
    lines.append("【约束】只判断草稿里的人称。不确定就当准确。不要改写草稿。")
    return "\n".join(lines)


def pronoun_questions() -> dict:
    return {
        "pronoun_accurate": {
            "type": "noul",
            "instructions": (
                "待发送草稿里的「我/我们/你/他/她/你们/他们」是否和说话关系、当前触发人、近期聊天一致。"
                "草稿由风雪说出，直接自称我通常指风雪，不是当前触发人。"
                "区分直接称呼、泛指用法和引用/转述内部的人称：泛指你不必绑定触发人，引语按原说话者判断。"
                "普通建议里的你不等于声称对方已经做过某件事。"
                "当前触发人没说过草稿里安给他的那句话，应为 false。"
                "把旁人的话当成当前触发人说的，应为 false。"
                "把发言人和被回复对象搞反，应为 false。"
                "没有人称、或吃不准，应为 true。"
                "不要判断好不好笑，不要改写草稿。"
            ),
        },
        "pronoun_issue": {
            "type": "choice",
            "instructions": (
                "如果人称准确或吃不准，选 none。"
                "只有很确定指错人时，才选具体错误类型。"
                "不要改写草稿。"
            ),
            "criteria": pronoun_issue_criteria(),
        },
    }


def parse_jev_pronoun_answers(data: dict, *, has_pronoun: bool) -> PronounJudgement:
    answers = data.get("answers") if isinstance(data, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    raw_score = answers.get("pronoun_accurate")
    raw_score = raw_score.get("noul") if isinstance(raw_score, dict) else None
    valid = not isinstance(raw_score, bool)
    try:
        noul = float(raw_score)
        valid = valid and math.isfinite(noul) and 0.0 <= noul <= 1.0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        return PronounJudgement(has_pronoun=has_pronoun, noul=0.5, issue="other", reason="error")
    raw_issue = answers.get("pronoun_issue")
    issue = str(raw_issue.get("choice") or "none").strip().lower() if isinstance(raw_issue, dict) else "other"
    if issue not in PRONOUN_ISSUE_CHOICES:
        issue = "other"
    accurate = YES if noul >= 0.45 else NO
    return PronounJudgement(
        has_pronoun=has_pronoun,
        accurate=accurate,
        noul=noul,
        issue=issue,
        reason="jev_pronoun",
    )


def apply_jev_pronoun_judgement(judgement: PronounJudgement | None, *, has_pronoun: bool) -> PronounResult:
    if not has_pronoun:
        return PronounResult(has_pronoun=False, reason="no_pronoun", status=NOT_APPLICABLE, source=SOURCE_RULE)
    if judgement is None:
        return PronounResult(has_pronoun=True, reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    return PronounResult(
        has_pronoun=True,
        accurate=judgement.accurate,
        issue=judgement.issue,
        noul=judgement.noul,
        reason=judgement.reason or "jev_pronoun",
        status=ERROR if judgement.reason == "error" else RESOLVED,
        source=SOURCE_JEV,
        value=f"{judgement.accurate}:{judgement.issue}:{judgement.noul:.2f}",
    )


def format_pronoun_feedback(result: PronounResult) -> str:
    if not result.needs_fix:
        return ""
    labels = pronoun_issue_criteria()
    issue_text = labels.get(result.issue, labels["other"])
    return "\n".join(
        [
            "[pronoun]",
            f"status={result.status}",
            f"issue={result.issue}",
            issue_text,
            "只改人称和归属，不要改话题，不要解释检查过程。不确定就不要点名。",
        ]
    )
