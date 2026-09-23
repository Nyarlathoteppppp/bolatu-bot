from __future__ import annotations

from dataclasses import dataclass

from .discourse_state import format_discourse_prompt_block
from .ellipsis_resolver import EllipsisResolution
from .jev_policy import (
    CRITIC_ANSWER_INTENT_FAIL_THRESHOLD,
    CRITIC_FAIL_THRESHOLD,
    CRITIC_INTENT_FAIL_THRESHOLD,
    CRITIC_PASS_THRESHOLD,
)
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
UNCERTAIN = "UNCERTAIN"
CRITIC_CHOICES = (YES, NO)
CRITIC_VALUES = (YES, NO, UNCERTAIN)
MAX_CRITIC_RETRIES = 1

_FAIL_ON_NO = ("intent_covered", "referent_consistent", "context_consistent")
_FAIL_ON_YES = ("unsupported_claim",)


@dataclass(frozen=True)
class CriticJudgement:
    intent_covered: str = ""
    referent_consistent: str = ""
    context_consistent: str = ""
    unsupported_claim: str = ""
    failure_probabilities: tuple[tuple[str, float], ...] = ()
    unavailable: tuple[str, ...] = ()
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
    uncertain: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    regenerated: bool = False

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if "jev_unavailable" in reason:
                status = UNAVAILABLE
            elif reason == "error" or reason.endswith("_error"):
                status = ERROR
            elif self.intent_covered in CRITIC_VALUES:
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
        if not self.uncertain and status == RESOLVED:
            answers = {
                "intent_covered": self.intent_covered,
                "referent_consistent": self.referent_consistent,
                "context_consistent": self.context_consistent,
                "unsupported_claim": self.unsupported_claim,
            }
            object.__setattr__(
                self,
                "uncertain",
                tuple(key for key, value in answers.items() if value == UNCERTAIN),
            )
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
            "covered": (
                "草稿直接回答或执行了当前请求，针对缺少的必要信息进行追问，"
                "或在 action=tease/care/agree/reply 时用与当前对象直接相关的社交回应处理了消息；"
                "明确说明不执行并给出与请求内容直接相关的拒绝理由也算处理；纠正被明确接受也算处理"
            ),
            "missed": (
                "草稿忽略当前消息、转移话题、用另一个对象替代当前明确询问的对象，"
                "或仍沿用被纠正掉的值"
            ),
            "not_applicable": "当前消息没有需要处理的请求、问题或纠正",
            "other": "证据不足或不符合以上情况",
        }
    if key == "referent_consistent":
        return {
            "consistent": "草稿中的人和对象都与已解析字段一致",
            "conflict": "草稿中至少一个人或对象与已解析字段冲突",
            "not_applicable": "草稿没有指向具体的人或对象",
            "other": "证据不足或不符合以上情况",
        }
    if key == "context_consistent":
        return {
            "consistent": "草稿不与任何已解析 discourse 字段冲突",
            "conflict": "草稿与至少一个已解析 discourse 字段冲突",
            "not_applicable": "没有已解析 discourse 字段适用于这份草稿",
            "other": "证据不足或不符合以上情况",
        }
    return {
        "supported": (
            "每个作为答案依据的具体事实都由近期聊天、memory 或 tool result 明确支持或直接推出；"
            "纯比喻或玩笑里的非核心近似例子不在此项核验"
        ),
        "unsupported": "至少一个具体事实增加了来源中不存在的人、数字、地点、事件、承诺或工具结论",
        "not_applicable": "草稿只有建议、观点、提问或确认，没有具体事实主张",
        "other": "证据不足或不符合以上情况",
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


def format_intent_critic_jev_state(*, draft: str, current_text: str, action: str, discourse=None) -> str:
    resolved = ""
    if discourse is not None:
        # Supply established bindings, never ask the intent critic to re-parse history.
        ellipsis = discourse.ellipsis
        if ellipsis.status == RESOLVED and ellipsis.source_text:
            resolved = f"【已解析的省略继承】{ellipsis.source_text[:240]}"
    return "\n".join((
        f"【待发送草稿】{(draft or '')[:400]}",
        f"【当前消息】{(current_text or '')[:400]}",
        f"【当前说话动作】{action or 'reply'}",
        resolved,
        "【约束】只比较当前消息、明确提供的已解析省略继承和草稿；不要猜测其他聊天内容。",
    ))


def critic_questions() -> dict:
    return {
        "intent_covered": {
            "type": "choice",
            "instructions": (
                "只比较【待发送草稿】和【当前消息】，判断草稿是否处理了当前请求、问题或纠正。"
                "若提供【已解析的省略继承】，用它补全当前请求的对象；不得重新解析历史。"
                "缺少必要信息时，针对缺口追问算已处理。"
                "action=tease/care/agree/reply 时，只要草稿围绕当前消息中的对象或话题形成直接回应就算已处理，"
                "不要求它完整回答事实问题。"
                "当前消息明确询问某个命名对象时，只谈另一个对象算未处理；同属一个类别不算回答。"
                "草稿明确说明不执行当前请求且拒绝理由直接对应请求内容时，也算已处理；空泛说不回答不算。"
                "对于纠正，明确承认或清楚接受纠正后的值算已处理。"
                "不要改写草稿。不要判断其他项。"
            ),
            "criteria": critic_choice_criteria("intent_covered"),
        },
        "referent_consistent": {
            "type": "choice",
            "instructions": (
                "只比较【待发送草稿】里的人和对象与【已解析指代】及【已解析修正】。"
                "不要重新解析群聊。纠正后的对象优先。不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("referent_consistent"),
        },
        "context_consistent": {
            "type": "choice",
            "instructions": (
                "只比较【待发送草稿】和已解析 discourse 的听话人、指示词、省略继承、纠正结果及 media_present。"
                "media_present=false 时问缺图或链接也算打架。"
                "不要重新解析群聊。不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("context_consistent"),
        },
        "unsupported_claim": {
            "type": "choice",
            "instructions": (
                "只核对【待发送草稿】中可外部核验的具体事实是否有【近期聊天】、【memory】或【tool result】支持。"
                "一般建议、观点、带不确定标记的推测及承认纠正都不是具体事实主张。"
                "纯比喻或玩笑中的非核心近似数字和泛化例子不作事实拦截；"
                "只有把新增事实作为答案依据、工具结论或现实事件陈述时才核验。"
                "不要使用外部知识，也不要判断外部世界真假。不要改写草稿。"
            ),
            "criteria": critic_choice_criteria("unsupported_claim"),
        },
    }


def parse_jev_critic_answers(data: dict, *, action: str = "") -> CriticJudgement:
    payload = data if isinstance(data, dict) else {}
    maybe_answers = payload.get("answers")
    answers: dict = maybe_answers if isinstance(maybe_answers, dict) else {}

    def _legacy_choice(key: str) -> str:
        candidate = answers.get(key)
        choice = ""
        if isinstance(candidate, dict):
            choice = str(candidate.get("choice") or "").strip().upper()
        return choice if choice in CRITIC_CHOICES else ""

    failure_labels = {
        "intent_covered": "missed",
        "referent_consistent": "conflict",
        "context_consistent": "conflict",
        "unsupported_claim": "unsupported",
    }

    def _failure_probability(key: str) -> float | None:
        candidate = answers.get(key)
        probabilities = candidate.get("probabilities") if isinstance(candidate, dict) else None
        value = probabilities.get(failure_labels[key]) if isinstance(probabilities, dict) else None
        if value is None or isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return score if 0.0 <= score <= 1.0 else None

    probabilities = {
        key: score
        for key in ("intent_covered", "referent_consistent", "context_consistent", "unsupported_claim")
        if (score := _failure_probability(key)) is not None
    }

    def _verdict(key: str, *, fail_value: str, pass_value: str) -> str:
        legacy = _legacy_choice(key)
        if legacy:
            return legacy
        score = probabilities.get(key)
        if score is None:
            return ""
        if key == "intent_covered":
            fail_threshold = (
                CRITIC_ANSWER_INTENT_FAIL_THRESHOLD
                if str(action or "").strip().lower() == "answer"
                else CRITIC_INTENT_FAIL_THRESHOLD
            )
        else:
            fail_threshold = CRITIC_FAIL_THRESHOLD
        if score >= fail_threshold:
            return fail_value
        if score <= CRITIC_PASS_THRESHOLD:
            candidate = answers.get(key)
            choice = str(candidate.get("choice") or "").strip().casefold() if isinstance(candidate, dict) else ""
            return UNCERTAIN if choice == "other" else pass_value
        return UNCERTAIN

    verdicts = {
        "intent_covered": _verdict("intent_covered", fail_value=NO, pass_value=YES),
        "referent_consistent": _verdict("referent_consistent", fail_value=NO, pass_value=YES),
        "context_consistent": _verdict("context_consistent", fail_value=NO, pass_value=YES),
        "unsupported_claim": _verdict("unsupported_claim", fail_value=YES, pass_value=NO),
    }
    return CriticJudgement(
        **verdicts,
        unavailable=tuple(key for key, value in verdicts.items() if not value),
        failure_probabilities=tuple(probabilities.items()),
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
    names = ("intent_covered", "referent_consistent", "context_consistent", "unsupported_claim")
    unavailable = tuple(key for key, value in zip(names, values) if value not in CRITIC_VALUES)
    if len(unavailable) == len(names):
        return CriticResult(reason="error", status=ERROR, source=SOURCE_JEV, unavailable=unavailable)
    clean = tuple(value if value in CRITIC_VALUES else UNCERTAIN for value in values)
    return CriticResult(
        intent_covered=clean[0],
        referent_consistent=clean[1],
        context_consistent=clean[2],
        unsupported_claim=clean[3],
        unavailable=unavailable,
        reason="jev_critic_partial" if unavailable else judgement.reason or "jev_critic",
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
    if "unsupported_claim" in result.failures:
        lines.append(
            "点名也必须出声：查不到的事实改成不确定或点出这是群梗，不要编，不要空回复。"
        )
    return "\n".join(lines)


def critic_blocks_memory(result: CriticResult | None) -> bool:
    if result is None or result.status != RESOLVED or not result.failed:
        return False
    return any(key in result.failures for key in ("referent_consistent", "context_consistent"))


def critic_prefers_clarify(result: CriticResult | None) -> bool:
    if result is None or not result.failed:
        return False
    return any(key in result.failures for key in ("referent_consistent", "context_consistent", "intent_covered"))


def next_critic_action(
    result: CriticResult | None,
    *,
    attempt: int,
    addressed: bool = False,
) -> str:
    if result is None or result.status in {UNAVAILABLE, ERROR, NOT_APPLICABLE}:
        return "send"
    if not result.failed:
        return "send"
    if int(attempt) <= 0:
        return "regenerate"
    # Addressed messages must still go out. Silence looks like the bot ignored @/reply.
    if addressed:
        return "send"
    return "block"


def critic_needs_clarify(result: CriticResult | None) -> bool:
    if result is None or result.status != RESOLVED or not result.failed:
        return False
    return any(key in result.failures for key in ("referent_consistent", "context_consistent"))
