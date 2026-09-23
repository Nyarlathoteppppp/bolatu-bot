from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .jev_policy import JEV_ELLIPSIS_CONFIDENCE_MIN
from .reference_resolver import ReferenceResolution, ReplyHint
from .resolver_result import (
    AMBIGUOUS,
    ERROR,
    NOT_APPLICABLE,
    RESOLVED,
    SOURCE_JEV,
    SOURCE_RULE,
    UNAVAILABLE,
    choice_confidence,
    finalize_result,
    joint_choice_confidence,
)


MAX_ELLIPSIS_SOURCES = 6
SHORT_TEXT_CHARS = 12
TRIGGER_MAX_CHARS = 24
SAME_SPEAKER_BURST = 3
MENTION_NOISE_RE = re.compile(r"\[@\d+\]|\[CQ:at,[^\]]+\]|@\d+")
DEIXIS_RE = re.compile(
    r"(看看这个|看这个|你看这个|看看那个|看那个|这个呢|那个呢|这玩意|那玩意|这东西|这样|"
    r"这个方案|这个价格|这个图|(?<![人])这个(?!人)|(?<![人])那个(?!人))"
)

ELLIPSIS_KINDS = (
    "NONE",
    "SAME_PREDICATE",
    "SLOT_QUERY",
    "SHORT_ANSWER",
    "CONTINUATION",
    "COMPARISON",
    "ITEM_DEIXIS",
)

SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("LOCATION", re.compile(r"(去哪|哪儿|哪里|哪[儿里]?$|在哪)")),
    ("TIME", re.compile(r"(什么时候|几点)")),
    ("PERSON", re.compile(r"(谁)")),
    ("REASON", re.compile(r"(为什么|为啥)")),
    ("METHOD", re.compile(r"(怎么做|怎么弄|怎么)")),
    ("QUANTITY", re.compile(r"(多少钱|多少|几个)")),
    ("CHOICE", re.compile(r"(哪个)")),
)

CUE_RE = re.compile(
    r"(然后呢|后来呢|接着呢|去哪|哪儿|哪里|多少|什么时候|为什么|怎么样|"
    r"也行|不行|差不多|第二个呢|这个呢|那个呢|看看这个|看这个|看看那个|看那个|呢[？?]?$)"
)
SHORT_ANSWER_RE = re.compile(r"^(也行|可以|不行|差不多|好|行|嗯|哦|啊|对|不是|上海|北京|广州|深圳).{0,6}$")
COMPLETE_CLAUSE_RE = re.compile(
    r"(觉得|准备|其实|还行|不错|挺强|很强|比.+贵|写代码|考研|是|有)"
)
QUESTION_TAIL_RE = re.compile(r"[？?吗呢]$")


def semantic_message_text(text: str) -> str:
    """Extract the actual utterance from a normalized QQ reply envelope."""
    value = str(text or "").strip()
    if "消息【" not in value or "回复" not in value or not value.endswith("】"):
        return value
    reply_at = value.rfind("回复")
    colon_positions = [pos for token in ("：", ":") if (pos := value.find(token, reply_at)) >= 0]
    if not colon_positions:
        return value
    current = value[min(colon_positions) + 1 : -1].strip()
    return current or value


@dataclass(frozen=True)
class EllipsisSource:
    key: str
    speaker: str
    text: str
    source_reason: str
    message_id: str = ""
    user_id: int = 0
    is_question: bool = False


@dataclass(frozen=True)
class EllipsisJudgement:
    kind: str = "NONE"
    inherit_from: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class EllipsisResolution:
    kind: str = "NONE"
    source_message_id: str = ""
    source_key: str = ""
    source_text: str = ""
    slot: str | None = None
    confidence: float = 0.0
    unresolved: bool = False
    reason: str = "none"
    status: str = ""
    source: str = ""
    value: str = ""

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if "jev_unavailable" in reason:
                status = UNAVAILABLE
            elif self.unresolved or reason in {
                "jev_no_source",
                "jev_low_confidence",
                "jev_unknown_source",
            }:
                status = AMBIGUOUS
            elif self.kind not in {"", "NONE"} and self.source_text:
                status = RESOLVED
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(self, "source", SOURCE_JEV if reason.startswith("jev_") else SOURCE_RULE)
        finalize_result(self, has_value=bool(self.source_text) or self.kind not in {"", "NONE"})


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def utterance_for_ellipsis(text: str) -> str:
    return compact_text(MENTION_NOISE_RE.sub("", str(text or "")))


def should_ask_jev_ellipsis(text: str) -> bool:
    compact = utterance_for_ellipsis(text)
    if not compact:
        return False
    if COMPLETE_CLAUSE_RE.search(compact) and len(compact) >= 10 and not compact.endswith(("呢", "呢？", "呢?")):
        if "呢其实" in compact or ( "呢" in compact and not compact.endswith("呢") and len(compact) >= 12):
            return False
        if not CUE_RE.search(compact) and len(compact) >= SHORT_TEXT_CHARS:
            return False
    if DEIXIS_RE.search(compact):
        return True
    if len(compact) <= SHORT_TEXT_CHARS:
        return True
    if CUE_RE.search(compact) and len(compact) <= TRIGGER_MAX_CHARS:
        return True
    return bool(SHORT_ANSWER_RE.match(compact))


def slot_for_query(text: str, kind: str) -> str | None:
    if kind != "SLOT_QUERY":
        return None
    compact = compact_text(text)
    for slot, pattern in SLOT_PATTERNS:
        if pattern.search(compact) or pattern.search(str(text or "")):
            return slot
    return "UNKNOWN"


def source_key_for_message(message: object, *, index: int) -> str:
    source_id = str(getattr(message, "source_message_id", "") or "").strip()
    if source_id:
        return f"s{source_id}"
    msg_id = getattr(message, "id", 0) or 0
    if int(msg_id) > 0:
        return f"m{int(msg_id)}"
    return f"i{index}"


def looks_like_question(text: str) -> bool:
    compact = compact_text(text)
    if not compact:
        return False
    if compact.endswith(("？", "?", "吗", "呢")):
        return True
    return bool(QUESTION_TAIL_RE.search(compact))


def looks_like_statement(text: str) -> bool:
    compact = compact_text(text)
    if not compact or looks_like_question(text):
        return False
    return len(compact) >= 4


def build_ellipsis_candidates(
    recent_messages: Iterable[object],
    *,
    current_text: str,
    current_user_id: int = 0,
    reply: ReplyHint | None = None,
    reference: ReferenceResolution | None = None,
    max_candidates: int = MAX_ELLIPSIS_SOURCES,
) -> list[EllipsisSource]:
    messages = tuple(recent_messages)
    rows: list[EllipsisSource] = []
    seen: set[str] = set()

    def add(message: object | None, reason: str, *, synthetic: EllipsisSource | None = None) -> bool:
        if len(rows) >= max_candidates:
            return False
        if synthetic is not None:
            if synthetic.key in seen or not synthetic.text.strip():
                return False
            seen.add(synthetic.key)
            rows.append(synthetic)
            return True
        if message is None:
            return False
        index = -1
        for idx, item in enumerate(messages):
            if item is message:
                index = idx
                break
        key = source_key_for_message(message, index=index if index >= 0 else len(rows))
        if key in seen:
            return False
        text = semantic_message_text(str(getattr(message, "text", "") or ""))
        if not text:
            return False
        speaker = str(getattr(message, "nickname", "") or getattr(message, "user_id", "") or "")
        if bool(getattr(message, "is_bot", False)):
            speaker = "风雪"
        source = EllipsisSource(
            key=key,
            speaker=speaker,
            text=text,
            source_reason=reason,
            message_id=str(getattr(message, "source_message_id", "") or getattr(message, "id", "") or ""),
            user_id=int(getattr(message, "user_id", 0) or 0),
            is_question=looks_like_question(text),
        )
        seen.add(key)
        rows.append(source)
        return True

    if reply is not None and reply.exists:
        matched = None
        reply_id = str(reply.message_id or "").strip()
        reply_text = str(reply.text or "").strip()
        for msg in messages:
            source_id = str(getattr(msg, "source_message_id", "") or "").strip()
            if reply_id and source_id == reply_id:
                matched = msg
                break
            if reply_text and str(getattr(msg, "text", "") or "").strip() == reply_text:
                matched = msg
                break
        if matched is not None:
            add(matched, "reply")
        elif reply_text:
            add(
                None,
                "reply",
                synthetic=EllipsisSource(
                    key=f"s{reply_id}" if reply_id else "sreply",
                    speaker=reply.author_label or str(reply.author_id or "被回复消息"),
                    text=reply_text,
                    source_reason="reply",
                    message_id=reply_id,
                    user_id=int(reply.author_id or 0),
                    is_question=looks_like_question(reply_text),
                ),
            )

    speaker_id = int(current_user_id or 0)
    if speaker_id > 0:
        burst = 0
        for msg in reversed(messages):
            uid = int(getattr(msg, "user_id", 0) or 0)
            if uid != speaker_id or bool(getattr(msg, "is_bot", False)):
                if burst:
                    break
                continue
            if add(msg, "same_speaker"):
                burst += 1
            if burst >= SAME_SPEAKER_BURST:
                break

    if messages:
        add(messages[-1], "previous")

    for msg in reversed(messages):
        if looks_like_question(semantic_message_text(str(getattr(msg, "text", "") or ""))) and add(msg, "recent_question"):
            break

    referent_ids = set(reference.user_ids) if reference is not None else set()
    if referent_ids:
        names = {
            int(getattr(msg, "user_id", 0) or 0): str(getattr(msg, "nickname", "") or "").strip()
            for msg in messages
            if int(getattr(msg, "user_id", 0) or 0) in referent_ids
        }
        aliases = {name for name in names.values() if name}
        for msg in reversed(messages):
            uid = int(getattr(msg, "user_id", 0) or 0)
            text = semantic_message_text(str(getattr(msg, "text", "") or ""))
            if (uid in referent_ids or any(alias and alias in text for alias in aliases)) and add(
                msg, "referent_related"
            ):
                break

    for msg in reversed(messages):
        text = semantic_message_text(str(getattr(msg, "text", "") or ""))
        if looks_like_statement(text) and add(msg, "recent_statement"):
            break

    return rows[:max_candidates]


def format_ellipsis_jev_state(
    *,
    current_text: str,
    current_label: str,
    sources: Iterable[EllipsisSource],
    reply: ReplyHint | None = None,
    reference: ReferenceResolution | None = None,
) -> str:
    lines = [
        f"【当前发言人】{current_label}",
        f"【当前消息】{(current_text or '')[:240]}",
    ]
    if reply is not None and reply.exists:
        lines.append(
            f"【QQ回复】exists=true author={reply.author_label or reply.author_id} "
            f"id={reply.message_id} text={(reply.text or '')[:120]}"
        )
    else:
        lines.append("【QQ回复】exists=false")
    if reference is not None and reference.user_ids:
        refs = ",".join(str(uid) for uid in reference.user_ids)
        lines.append(f"【已解析指代】{refs} reason={reference.reason} kind={reference.kind}")
    lines.append("【可继承的前文】")
    for source in list(sources)[:MAX_ELLIPSIS_SOURCES]:
        lines.append(
            f"- {source.key} | speaker={source.speaker} | reason={source.source_reason} | "
            f"{source.speaker}：{source.text[:80]}"
        )
    return "\n".join(lines)


def ellipsis_inherit_criteria(sources: Iterable[EllipsisSource]) -> dict[str, str]:
    criteria = {"NONE": "当前句语义自足，或不该继承任何一条"}
    for source in list(sources)[:MAX_ELLIPSIS_SOURCES]:
        criteria[source.key] = (
            f"继承 {source.speaker} 的「{source.text[:40]}」({source.source_reason})"
        )
    criteria["other"] = "以上均不符合或无法判断"
    return criteria


def parse_jev_ellipsis_answers(data: dict) -> EllipsisJudgement:
    answers = data.get("answers") if isinstance(data, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    kind_raw = answers.get("ellipsis_kind") if isinstance(answers.get("ellipsis_kind"), dict) else {}
    inherit_raw = answers.get("inherit_from") if isinstance(answers.get("inherit_from"), dict) else {}
    if not isinstance(kind_raw, dict):
        kind_raw = {}
    if not isinstance(inherit_raw, dict):
        inherit_raw = {}
    kind = str(kind_raw.get("choice") or "").strip().upper()
    if kind not in (*ELLIPSIS_KINDS, "OTHER"):
        return EllipsisJudgement(reason="error")
    inherit_from = str(inherit_raw.get("choice") or "NONE").strip()
    if kind == "NONE":
        inherit_from = "NONE"
    confidence = (
        choice_confidence(answers, "ellipsis_kind")
        if kind in {"NONE", "OTHER"}
        else joint_choice_confidence(answers, "ellipsis_kind", "inherit_from")
    )
    if confidence is None:
        return EllipsisJudgement(
            kind=kind,
            inherit_from=inherit_from,
            confidence=0.0,
            reason="error",
        )
    return EllipsisJudgement(
        kind=kind,
        inherit_from=inherit_from,
        confidence=confidence,
        reason=f"jev_{kind.lower()}_{inherit_from}",
    )


def apply_jev_ellipsis_judgement(
    judgement: EllipsisJudgement | None,
    sources: Iterable[EllipsisSource],
    *,
    current_text: str,
) -> EllipsisResolution:
    rows = list(sources)
    if judgement is None:
        return EllipsisResolution(reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    if judgement.reason == "error" or judgement.kind not in (*ELLIPSIS_KINDS, "OTHER"):
        return EllipsisResolution(reason="error", status=ERROR, source=SOURCE_JEV)
    kind = judgement.kind
    if kind == "OTHER" or float(judgement.confidence or 0.0) < JEV_ELLIPSIS_CONFIDENCE_MIN:
        return EllipsisResolution(
            kind="NONE" if kind == "OTHER" else kind,
            reason="jev_other" if kind == "OTHER" else "jev_low_confidence",
            confidence=float(judgement.confidence or 0.0),
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if kind == "NONE":
        return EllipsisResolution(
            kind="NONE",
            reason=judgement.reason or "jev_none",
            status=NOT_APPLICABLE,
            source=SOURCE_JEV,
        )
    inherit_from = str(judgement.inherit_from or "NONE").strip()
    if inherit_from.upper() == "NONE" or inherit_from == "":
        return EllipsisResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_no_source",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if float(judgement.confidence or 0.0) < JEV_ELLIPSIS_CONFIDENCE_MIN:
        return EllipsisResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_low_confidence",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    for source in rows:
        if source.key == inherit_from:
            return EllipsisResolution(
                kind=kind,
                source_message_id=source.message_id or source.key,
                source_key=source.key,
                source_text=source.text,
                slot=slot_for_query(current_text, kind),
                confidence=float(judgement.confidence),
                reason=judgement.reason or "jev_ellipsis",
                status=RESOLVED,
                source=SOURCE_JEV,
                value=source.key,
            )
    return EllipsisResolution(
        kind=kind,
        unresolved=True,
        confidence=float(judgement.confidence or 0.0),
        reason="jev_unknown_source",
        status=AMBIGUOUS,
        source=SOURCE_JEV,
    )


def format_ellipsis_prompt_block(resolution: EllipsisResolution) -> str:
    if resolution.status == UNAVAILABLE:
        return (
            "[ellipsis]\n"
            "status=UNAVAILABLE\n"
            "不要编造被省略的前文。"
        )
    if resolution.unresolved or resolution.status == AMBIGUOUS:
        return (
            "[ellipsis]\n"
            f"kind={resolution.kind or 'UNKNOWN'}\n"
            "unresolved_ellipsis=true\n"
            "不要编造被省略的前文。"
        )
    if resolution.kind in {"", "NONE"} or not resolution.source_text:
        return ""
    lines = [
        "[ellipsis]",
        f"kind={resolution.kind}",
        f"inherits_from={resolution.source_key or resolution.source_message_id}",
    ]
    if resolution.slot:
        lines.append(f"slot={resolution.slot}")
    lines.append(f'source_text="{resolution.source_text[:120]}"')
    return "\n".join(lines)
