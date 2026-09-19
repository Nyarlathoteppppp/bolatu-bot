from __future__ import annotations

from dataclasses import dataclass

from .discourse_state import format_discourse_prompt_block
from .ellipsis_resolver import EllipsisResolution
from .reference_resolver import ReferenceResolution
from .resolver_result import (
    ERROR,
    NOT_APPLICABLE,
    RESOLVED,
    SOURCE_JEV,
    SOURCE_RULE,
    UNAVAILABLE,
    finalize_result,
)

YES = "YES"
NO = "NO"
CRITIC_CHOICES = (YES, NO)
MAX_CRITIC_RETRIES = 1

_FAIL_ON_NO = ("intent_covered", "referent_consistent", "context_consistent")
_FAIL_ON_YES = ("unsupported_claim",)


@dataclass(frozen=True)
class CriticJudgement:
    intent_covered: str = ""
    referent_consistent: str = ""
    context_consistent: str = ""
    unsupported_claim: str = ""
    reason: str = ""


@dataclass(frozen=True)
class CriticResult:
    intent_covered: str = NOT_APPLICABLE
    referent_consistent: str = NOT_APPLICABLE
    context_consistent: str = NOT_APPLICABLE
    unsupported_claim: str = NOT_APPLICABLE
    status: str = ""
    source: str = ""
    value: str = ""
    reason: str = "none"
    failed: bool = False
    failures: tuple[str, ...] = ()
    regenerated: bool = False

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if "jev_unavailable" in reason:
                status = UNAVAILABLE
            elif reason == "error" or reason.endswith("_error"):
                status = ERROR
            elif self.intent_covered in CRITIC_CHOICES:
                status = RESOLVED
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(
                self,
                "source",
                SOURCE_JEV if reason.startswith("jev_") or status in {RESOLVED, UNAVAILABLE} else SOURCE_RULE,
            )
        failures = self.failures
        if not failures and status == RESOLVED:
            failures = _failures_from_answers(
                intent_covered=self.intent_covered,
                referent_consistent=self.referent_consistent,
                context_consistent=self.context_consistent,
                unsupported_claim=self.unsupported_claim,
            )
            object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "failed", bool(failures) and status == RESOLVED)
        finalize_result(self, has_value=status == RESOLVED)


def _failures_from_answers(
    *,
    intent_covered: str,
    referent_consistent: str,
    context_consistent: str,
    unsupported_claim: str,
) -> tuple[str, ...]:
    answers = {
        "intent_covered": intent_covered,
        "referent_consistent": referent_consistent,
        "context_consistent": context_consistent,
        "unsupported_claim": unsupported_claim,
    }
    failed: list[str] = []
    for key in _FAIL_ON_NO:
        if answers.get(key) == NO:
            failed.append(key)
    for key in _FAIL_ON_YES:
        if answers.get(key) == YES:
            failed.append(key)
    return tuple(failed)


def critic_choice_criteria(key: str) -> dict[str, str]:
    if key == "intent_covered":
        return {
            YES: "待发送草稿回应了当前消息里的那个请求、纠正或问题本身",
            NO: "待发送草稿在谈别的事，或没处理当前消息真正在说的那一点",
        }
    if key == "referent_consistent":
        return {
            YES: "待发送草稿里的人/对象与已解析 DiscourseState 一致，或这条根本不需要人物对象",
            NO: "待发送草稿用了别人，或沿用了已解析修正之前的旧对象",
        }
    if key == "context_consistent":
        return {
            YES: "待发送草稿没有和 DiscourseState 的 addressee/deixis/repair 打架",
            NO: "待发送草稿把听话人、这个/那个、省略继承或纠正结果说反了",
        }
    return {
        YES: "待发送草稿写出了近期聊天、memory、tool result 里都没有的具体事实",
        NO: "待发送草稿没有写出这些来源里找不到的具体事实",
    }


def format_critic_jev_state(
    *,
    draft: str,
    current_text: str,
    action: str,
    current_label: str = "",
    speaker_context: str = "",
    recent_messages: list | tuple | None = None,
    memory_context: str = "",
    tool_context: str = "",
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
    repair=None,
    discourse=None,
) -> str:
    if discourse is not None:
        reference = reference or getattr(discourse, "reference", None)
        ellipsis = ellipsis or getattr(discourse, "ellipsis", None)
        repair = repair or getattr(discourse, "repair", None)
    lines = [
        f"【待发送草稿】{(draft or '')[:400]}",
        f"【当前消息】{(current_text or '')[:240]}" + (f"\n【当前触发人】{current_label}" if current_label else ""),
        f"【当前说话动作】{action or 'reply'}",
    ]
    if discourse is not None:
        lines.append(format_discourse_prompt_block(discourse))
    if reference is not None:
        lines.append(
            f"【已解析指代】status={reference.status} kind={reference.kind} "
            f"ids={reference.user_ids} value={reference.value} reason={reference.reason}"
        )
    if ellipsis is not None:
        lines.append(
            f"【已解析省略】status={ellipsis.status} kind={ellipsis.kind} "
            f"source={ellipsis.source_text[:80]}"
        )
    if repair is not None:
        lines.append(
            f"【已解析修正】status={repair.status} kind={repair.kind} "
            f"target={repair.target_key} replacement={repair.replacement_user_ids}"
        )
    if speaker_context.strip():
        lines.append("【说话关系】")
        lines.append(speaker_context.strip()[:400])
    history = []
    for msg in list(recent_messages or [])[-6:]:
        nick = getattr(msg, "nickname", None) or str(getattr(msg, "user_id", ""))
        prefix = "风雪" if getattr(msg, "is_bot", False) else nick
        history.append(f"{prefix}: {getattr(msg, 'text', '')}")
    if history:
        lines.append("【近期聊天】")
        lines.extend(history)
    if str(memory_context or "").strip():
        lines.append("【memory】" + str(memory_context).strip()[:300])
    if str(tool_context or "").strip():
        lines.append("【tool result】" + str(tool_context).strip()[:300])
    lines.append("【约束】只根据上面材料判断。不要改写待发送草稿。不要判断外部世界真假。")
    lines.append("当前消息和已解析状态优先于待发送草稿；已解析修正之前的旧对象一律作废。")
    lines.append("具体事实是否存在，只看近期聊天、memory、tool result，不查外部世界。")
    return "\n".join(lines)


def critic_questions() -> dict:
    return {
        "intent_covered": {
            "type": "choice",
            "instructions": (
                "只看待发送草稿和当前消息。"
                "待发送草稿有没有回应当前消息里那个请求、纠正或问题本身？"
                "不要改写草稿，也不要判断其他项。"
            ),
            "criteria": critic_choice_criteria("intent_covered"),
        },
        "referent_consistent": {
            "type": "choice",
            "instructions": (
                "只看待发送草稿和已解析 DiscourseState。"
                "待发送草稿里的人/对象是否与 DiscourseState 一致？"
                "不要重新解析群聊，只核对草稿有没有和已解析状态打架。"
                "如果已解析修正改过对象，必须以纠正后的对象为准。"
                "这条不需要人物对象时选 YES。不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("referent_consistent"),
        },
        "context_consistent": {
            "type": "choice",
            "instructions": (
                "只看待发送草稿和已解析 DiscourseState。"
                "待发送草稿有没有把听话人、这个/那个、省略继承或纠正结果说反？"
                "media_present=false 时问缺图或链接也算打架。"
                "不要重新解析群聊。没有打架选 YES，说反了选 NO。不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("context_consistent"),
        },
        "unsupported_claim": {
            "type": "choice",
            "instructions": (
                "只看待发送草稿、近期聊天、memory、tool result。"
                "待发送草稿有没有写出这三处都不存在的具体事实，例如人名、数字、地点、承诺、工具结论？"
                "有选 YES，没有选 NO。不要判断外部世界真假，也不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("unsupported_claim"),
        },
    }


def parse_jev_critic_answers(data: dict) -> CriticJudgement:
    payload = data if isinstance(data, dict) else {}
    maybe_answers = payload.get("answers")
    answers: dict = maybe_answers if isinstance(maybe_answers, dict) else {}

    def _choice(key: str) -> str:
        candidate = answers.get(key)
        choice = ""
        if isinstance(candidate, dict):
            choice = str(candidate.get("choice") or "").strip().upper()
        return choice if choice in CRITIC_CHOICES else ""

    return CriticJudgement(
        intent_covered=_choice("intent_covered"),
        referent_consistent=_choice("referent_consistent"),
        context_consistent=_choice("context_consistent"),
        unsupported_claim=_choice("unsupported_claim"),
        reason="jev_critic",
    )


def apply_jev_critic_judgement(judgement: CriticJudgement | None) -> CriticResult:
    if judgement is None:
        return CriticResult(reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    values = (
        judgement.intent_covered,
        judgement.referent_consistent,
        judgement.context_consistent,
        judgement.unsupported_claim,
    )
    if any(value not in CRITIC_CHOICES for value in values):
        return CriticResult(reason="error", status=ERROR, source=SOURCE_JEV)
    return CriticResult(
        intent_covered=judgement.intent_covered,
        referent_consistent=judgement.referent_consistent,
        context_consistent=judgement.context_consistent,
        unsupported_claim=judgement.unsupported_claim,
        reason=judgement.reason or "jev_critic",
        status=RESOLVED,
        source=SOURCE_JEV,
        value=",".join(values),
    )


def format_critic_feedback(result: CriticResult) -> str:
    if result.status == UNAVAILABLE:
        return ""
    if result.status == ERROR:
        return "[critic]\nstatus=ERROR\n不要把检查失败当成没有指代。"
    if not result.failed:
        return ""
    lines = [
        "[critic]",
        f"status={result.status}",
        f"intent_covered={result.intent_covered}",
        f"referent_consistent={result.referent_consistent}",
        f"context_consistent={result.context_consistent}",
        f"unsupported_claim={result.unsupported_claim}",
        "failures=" + ",".join(result.failures),
        "按失败项重写，不要沿用被纠正前的对象，不要编造上下文没有的信息。不要解释检查过程。",
    ]
    return "\n".join(lines)


def critic_blocks_memory(result: CriticResult | None) -> bool:
    if result is None or result.status != RESOLVED or not result.failed:
        return False
    return any(key in result.failures for key in ("referent_consistent", "context_consistent"))


def critic_prefers_clarify(result: CriticResult | None) -> bool:
    if result is None or not result.failed:
        return False
    return any(key in result.failures for key in ("referent_consistent", "context_consistent", "intent_covered"))


def next_critic_action(result: CriticResult | None, *, attempt: int) -> str:
    if result is None or result.status in {UNAVAILABLE, ERROR, NOT_APPLICABLE}:
        return "send"
    if not result.failed:
        return "send"
    if int(attempt) <= 0:
        return "regenerate"
    return "block"
