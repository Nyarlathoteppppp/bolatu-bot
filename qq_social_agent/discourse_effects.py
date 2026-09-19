from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from .ellipsis_resolver import EllipsisResolution
from .memory import MemoryAtom, MemoryStore
from .reference_resolver import ReferenceResolution, ReplyHint
from .resolver_result import (
    AMBIGUOUS,
    ERROR,
    NOT_APPLICABLE,
    RESOLVED,
    SOURCE_JEV,
    SOURCE_REPAIR,
    SOURCE_RULE,
    UNAVAILABLE,
    choice_confidence,
    finalize_result,
    has_usable_value,
    is_unresolved,
    joint_choice_confidence,
)


JEV_REPAIR_CONFIDENCE_MIN = 0.42
JEV_MEMORY_CONFIDENCE_MIN = 0.60
JEV_AMBIGUITY_CONFIDENCE_MIN = 0.60
MAX_REPAIR_TARGETS = 5
MAX_MEMORY_RELATED = 5

REPAIR_KINDS = ("NONE", "REFERENT", "ITEM", "FACT", "INTENT", "RETRACTION")
MEMORY_ACTIONS = ("IGNORE", "CREATE", "CONFIRM", "REFINE", "REPLACE", "NEGATE", "RETRACT")
AMBIGUITY_KINDS = (
    "NONE",
    "PERSON",
    "ITEM",
    "THREAD",
    "ELLIPSIS",
    "TIME",
    "SCOPE",
    "INTENT",
    "MEMORY_REFERENCE",
    "OTHER",
)

REPAIR_CUE_RE = re.compile(
    r"(不是.{0,12}是|我说的是|不是这个意思|前面那个|不是刚才|当我没说|刚才那句|撤回|不是贵，是|不是.+是)"
)
FACT_CUE_RE = re.compile(r"(准备|不考|改去|改成|后来|其实是|当我没说|不.+了)")
ORDINAL_RE = re.compile(r"(第[一二三四五六七八九十\d]+(?:个|套|条|项)?|最后一个)")
ITEM_LIST_RE = re.compile(r"(第[一二三四五六七八九十\d]+(?:个|套|条|项)?[^\s，。！？]{0,24})")
NOT_A_BUT_B_RE = re.compile(r"不是(?!这个意思)(?P<old>.{1,12}?)(?:，|,)?是(?!思)(?P<new>.{1,16})")

NamedUserResolver = Callable[[str], Iterable[int]]


@dataclass(frozen=True)
class RepairTarget:
    key: str
    kind: str
    summary: str
    message_id: str = ""
    user_id: int = 0
    memory_id: int = 0


@dataclass(frozen=True)
class RepairJudgement:
    kind: str = "NONE"
    target_key: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class RepairResolution:
    kind: str = "NONE"
    target_key: str = ""
    replacement_user_ids: tuple[int, ...] = ()
    replacement_item: str = ""
    unresolved: bool = False
    confidence: float = 0.0
    reason: str = "none"
    status: str = ""
    source: str = ""
    value: str = ""
    invalidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if "jev_unavailable" in reason:
                status = UNAVAILABLE
            elif self.unresolved or reason in {
                "jev_low_confidence",
                "jev_unknown_target",
                "jev_no_target",
                "missing_replacement",
                "intent_unspecified",
            }:
                status = AMBIGUOUS
            elif self.kind not in {"", "NONE"}:
                status = RESOLVED
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(self, "source", SOURCE_JEV if reason.startswith("jev_") or self.kind not in {"", "NONE"} else SOURCE_RULE)
        invalidates = self.invalidates
        if not invalidates and status == RESOLVED:
            if self.kind == "REFERENT":
                invalidates = ("referent", "ellipsis", "memory", "ambiguity")
            elif self.kind in {"ITEM", "INTENT"}:
                invalidates = ("ellipsis", "ambiguity")
            elif self.kind in {"FACT", "RETRACTION"}:
                invalidates = ("memory", "ambiguity")
            object.__setattr__(self, "invalidates", invalidates)
        finalize_result(self, has_value=self.kind not in {"", "NONE"})


@dataclass(frozen=True)
class MemoryCandidate:
    subject_user_id: int | None
    content: str
    source_message_id: str = ""
    speaker: str = ""
    predicate: str = ""
    value: str = ""
    confidence: float = 0.7


@dataclass(frozen=True)
class MemoryEffectJudgement:
    action: str = "IGNORE"
    target_id: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class MemoryEffectResolution:
    action: str = "IGNORE"
    target_atom_id: int | None = None
    applied: bool = False
    unresolved: bool = False
    confidence: float = 0.0
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
                "jev_low_confidence",
                "jev_unknown_target",
                "missing_target",
            }:
                status = AMBIGUOUS
            elif self.action in {"", "IGNORE"}:
                status = NOT_APPLICABLE
            else:
                status = RESOLVED
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(self, "source", SOURCE_JEV if reason.startswith("jev_") else SOURCE_RULE)
        finalize_result(self, has_value=self.action not in {"", "IGNORE"})


@dataclass(frozen=True)
class AmbiguityJudgement:
    kind: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class AmbiguityResolution:
    kind: str = "NONE"
    confidence: float = 0.0
    unresolved: bool = False
    candidates: tuple[str, ...] = ()
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
            elif self.unresolved or (self.kind not in {"", "NONE"}):
                status = AMBIGUOUS if self.kind not in {"", "NONE"} else NOT_APPLICABLE
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(self, "source", SOURCE_JEV if reason.startswith("jev_") else SOURCE_RULE)
        finalize_result(self, has_value=self.kind not in {"", "NONE"})


def should_ask_jev_repair(
    text: str,
    *,
    reference: ReferenceResolution | None = None,
    named_in_text: Iterable[int] = (),
) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if REPAIR_CUE_RE.search(str(text or "")) or REPAIR_CUE_RE.search(compact):
        return True
    named = tuple(int(uid) for uid in named_in_text if int(uid) > 0)
    if reference is not None and reference.user_ids and named and set(named).isdisjoint(reference.user_ids):
        return True
    return False


def build_repair_targets(
    *,
    recent_messages: Iterable[object] = (),
    reply: ReplyHint | None = None,
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
    related_memories: Iterable[MemoryAtom] = (),
    recent_bot_text: str = "",
    current_intent: str = "",
    current_text: str = "",
) -> list[RepairTarget]:
    rows: list[RepairTarget] = []
    seen: set[str] = set()

    def add(target: RepairTarget) -> None:
        if len(rows) >= MAX_REPAIR_TARGETS or target.key in seen:
            return
        if not str(target.summary or "").strip():
            return
        seen.add(target.key)
        rows.append(target)

    if reply is not None and reply.exists:
        add(
            RepairTarget(
                key="t_reply",
                kind="message",
                summary=(reply.text or "")[:80],
                message_id=reply.message_id,
                user_id=int(reply.author_id or 0),
            )
        )
    if recent_bot_text.strip():
        add(RepairTarget(key="t_bot", kind="intent", summary=recent_bot_text.strip()[:80]))
    if reference is not None and (reference.user_ids or reference.kind not in {"", "NONE"}):
        add(
            RepairTarget(
                key="t_referent",
                kind="referent",
                summary=f"referent={','.join(str(uid) for uid in reference.user_ids)} {reference.reason}",
                user_id=int(reference.user_ids[0]) if reference.user_ids else 0,
            )
        )
    if ellipsis is not None and ellipsis.kind not in {"", "NONE"} and ellipsis.source_text:
        add(
            RepairTarget(
                key="t_ellipsis",
                kind="item",
                summary=f"{ellipsis.kind}:{ellipsis.source_text[:60]}",
                message_id=ellipsis.source_message_id,
            )
        )
    related_msg = _related_repair_message(
        recent_messages,
        reply=reply,
        reference=reference,
        ellipsis=ellipsis,
        current_text=current_text,
    )
    if related_msg is not None:
        add(related_msg)
    for atom in list(related_memories)[:2]:
        add(
            RepairTarget(
                key=f"mem{atom.id}",
                kind="memory",
                summary=str(atom.content)[:80],
                memory_id=int(atom.id),
            )
        )
    if current_intent.strip() and len(rows) < MAX_REPAIR_TARGETS:
        add(RepairTarget(key="t_intent", kind="intent", summary=current_intent.strip()[:80]))
    if not rows:
        last = _last_non_current_message(recent_messages, current_text=current_text)
        if last is not None:
            add(last)
    return rows[:MAX_REPAIR_TARGETS]


def _message_target(message: object, *, key: str) -> RepairTarget:
    return RepairTarget(
        key=key,
        kind="message",
        summary=str(getattr(message, "text", "") or "")[:80],
        message_id=str(getattr(message, "source_message_id", "") or getattr(message, "id", "") or ""),
        user_id=int(getattr(message, "user_id", 0) or 0),
    )


def _last_non_current_message(recent_messages: Iterable[object], *, current_text: str) -> RepairTarget | None:
    current = str(current_text or "").strip()
    for index, message in enumerate(reversed(tuple(recent_messages))):
        text = str(getattr(message, "text", "") or "").strip()
        if not text or text == current:
            continue
        key = "t_related" if index else "t_previous"
        return _message_target(message, key=key)
    return None


def _related_repair_message(
    recent_messages: Iterable[object],
    *,
    reply: ReplyHint | None,
    reference: ReferenceResolution | None,
    ellipsis: EllipsisResolution | None,
    current_text: str,
) -> RepairTarget | None:
    messages = tuple(recent_messages)
    current = str(current_text or "").strip()
    skip_ids = {str(reply.message_id)} if reply is not None and reply.message_id else set()
    if ellipsis is not None and ellipsis.source_message_id:
        skip_ids.add(str(ellipsis.source_message_id))
    referent_ids = set(reference.user_ids) if reference is not None else set()
    names = {
        int(getattr(msg, "user_id", 0) or 0): str(getattr(msg, "nickname", "") or "").strip()
        for msg in messages
        if int(getattr(msg, "user_id", 0) or 0) in referent_ids
    }
    aliases = {name for name in names.values() if name}
    for message in reversed(messages):
        text = str(getattr(message, "text", "") or "").strip()
        if not text or text == current:
            continue
        mid = str(getattr(message, "source_message_id", "") or getattr(message, "id", "") or "")
        if mid and mid in skip_ids:
            continue
        uid = int(getattr(message, "user_id", 0) or 0)
        related = uid in referent_ids or any(alias and alias in text for alias in aliases)
        if ellipsis is not None and ellipsis.source_text and ellipsis.source_text[:20] in text:
            related = True
        if related:
            return _message_target(message, key="t_related")
    return None


def replacement_named_users(
    text: str,
    resolve_named_users: NamedUserResolver | None,
    *,
    old_user_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    if resolve_named_users is None:
        return ()
    named = _unique_named_users(resolve_named_users, text)
    old = {int(uid) for uid in old_user_ids}
    match = NOT_A_BUT_B_RE.search(re.sub(r"\s+", "", text))
    if match is not None:
        new_ids = _unique_named_users(resolve_named_users, match.group("new"))
        if new_ids:
            return new_ids
    leftover = tuple(uid for uid in named if uid not in old)
    if leftover:
        return leftover
    return named


def resolve_ordinal_item(text: str, recent_messages: Iterable[object]) -> str:
    compact = str(text or "")
    if any(token in compact for token in ("第一", "第二", "第三", "最后")):
        wanted = "第二"
        if "第一" in compact and "第二" not in compact:
            wanted = "第一"
        elif "第三" in compact:
            wanted = "第三"
        elif "最后" in compact:
            wanted = "最后"
        for msg in reversed(tuple(recent_messages)):
            blob = str(getattr(msg, "text", "") or "")
            for item in ITEM_LIST_RE.findall(blob):
                if wanted in item:
                    return item.strip()
            if wanted in blob:
                return blob.strip()[:80]
    collapsed = re.sub(r"\s+", "", compact)
    contrast = NOT_A_BUT_B_RE.search(collapsed)
    if contrast is not None:
        return contrast.group("new").strip()[:80]
    said = re.search(r"我说的是(.+)", compact)
    if said is not None:
        return said.group(1).strip()[:80]
    return ""


def format_repair_jev_state(
    *,
    current_text: str,
    current_label: str,
    targets: Iterable[RepairTarget],
    reply: ReplyHint | None = None,
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
) -> str:
    lines = [
        f"【当前发言人】{current_label}",
        f"【当前消息】{(current_text or '')[:240]}",
    ]
    if reply is not None and reply.exists:
        lines.append(f"【QQ回复】{reply.author_label}:{ (reply.text or '')[:80]}")
    if reference is not None and reference.user_ids:
        lines.append(f"【当前指代】{reference.user_ids} reason={reference.reason}")
    if ellipsis is not None and ellipsis.kind not in {"", "NONE"}:
        lines.append(f"【当前省略】{ellipsis.kind} source={ellipsis.source_text[:60]}")
    lines.append("【可修正目标】")
    for target in list(targets)[:MAX_REPAIR_TARGETS]:
        lines.append(f"- {target.key} | {target.kind} | {target.summary}")
    return "\n".join(lines)


def repair_target_criteria(targets: Iterable[RepairTarget]) -> dict[str, str]:
    criteria = {"NONE": "没有可修正的具体目标，或只是普通聊天"}
    for target in list(targets)[:MAX_REPAIR_TARGETS]:
        criteria[target.key] = f"{target.kind}: {target.summary[:40]}"
    return criteria


def parse_jev_repair_answers(data: dict) -> RepairJudgement:
    answers = _answers(data)
    kind = _choice(answers, "repair_kind", REPAIR_KINDS, "NONE")
    target = str((answers.get("repair_target") or {}).get("choice") or "NONE").strip()
    if kind == "NONE":
        target = "NONE"
    confidence = (
        choice_confidence(answers, "repair_kind")
        if kind == "NONE"
        else joint_choice_confidence(answers, "repair_kind", "repair_target")
    )
    return RepairJudgement(
        kind=kind,
        target_key=target,
        confidence=float(confidence or 0.0),
        reason=f"jev_{kind.lower()}_{target}" if confidence is not None else "error",
    )


def apply_jev_repair_judgement(
    judgement: RepairJudgement | None,
    targets: Iterable[RepairTarget],
    *,
    current_text: str,
    reference: ReferenceResolution | None = None,
    recent_messages: Iterable[object] = (),
    resolve_named_users: NamedUserResolver | None = None,
) -> RepairResolution:
    if judgement is None:
        return RepairResolution(reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    kind = judgement.kind if judgement.kind in REPAIR_KINDS else "NONE"
    if kind == "NONE":
        return RepairResolution(
            kind="NONE",
            reason=judgement.reason or "jev_none",
            status=NOT_APPLICABLE,
            source=SOURCE_JEV,
        )
    if float(judgement.confidence or 0.0) < JEV_REPAIR_CONFIDENCE_MIN:
        return RepairResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_low_confidence",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    target_key = str(judgement.target_key or "NONE")
    known = {item.key for item in targets}
    if target_key not in known and target_key != "NONE":
        return RepairResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_unknown_target",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if target_key == "NONE" and kind in {"REFERENT", "ITEM", "FACT", "RETRACTION"}:
        return RepairResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_no_target",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    replacement_ids: tuple[int, ...] = ()
    replacement_item = ""
    if kind == "REFERENT":
        replacement_ids = replacement_named_users(
            current_text,
            resolve_named_users,
            old_user_ids=(reference.user_ids if reference is not None else ()),
        )
        if not replacement_ids:
            return RepairResolution(
                kind=kind,
                target_key=target_key,
                unresolved=True,
                confidence=float(judgement.confidence or 0.0),
                reason="missing_replacement",
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
    if kind == "ITEM":
        replacement_item = resolve_ordinal_item(current_text, recent_messages)
        if not replacement_item:
            return RepairResolution(
                kind=kind,
                target_key=target_key,
                unresolved=True,
                confidence=float(judgement.confidence or 0.0),
                reason="missing_replacement",
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
    if kind == "INTENT":
        compact = re.sub(r"\s+", "", current_text)
        if "不是这个意思" in compact and not NOT_A_BUT_B_RE.search(compact):
            return RepairResolution(
                kind="INTENT",
                target_key=target_key,
                unresolved=True,
                confidence=float(judgement.confidence or 0.0),
                reason="intent_unspecified",
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
    return RepairResolution(
        kind=kind,
        target_key=target_key,
        replacement_user_ids=replacement_ids,
        replacement_item=replacement_item,
        confidence=float(judgement.confidence or 0.0),
        reason=judgement.reason or "jev_repair",
        status=RESOLVED,
        source=SOURCE_JEV,
        value=target_key,
    )


def apply_repair_to_reference(
    reference: ReferenceResolution,
    repair: RepairResolution,
) -> ReferenceResolution:
    if repair.status != RESOLVED or repair.kind != "REFERENT" or not repair.replacement_user_ids:
        return reference
    return replace(
        reference,
        user_ids=repair.replacement_user_ids,
        reason="repair_referent",
        confidence=max(reference.confidence, repair.confidence),
        kind="PERSON",
        unresolved=False,
        expanded_query=reference.expanded_query,
        status=RESOLVED,
        source=SOURCE_REPAIR,
        value=",".join(str(uid) for uid in repair.replacement_user_ids),
    )


def repair_invalidates(repair: RepairResolution, layer: str) -> bool:
    if repair.status != RESOLVED:
        return False
    return layer in repair.invalidates


def invalidated_layers_for_repair(repair: RepairResolution) -> tuple[str, ...]:
    if repair.status != RESOLVED:
        return ()
    return tuple(repair.invalidates)


def reset_ellipsis_for_repair(ellipsis: EllipsisResolution, repair: RepairResolution) -> EllipsisResolution:
    if not repair_invalidates(repair, "ellipsis"):
        return ellipsis
    return EllipsisResolution(
        reason="invalidated_by_repair",
        status=NOT_APPLICABLE,
        source=SOURCE_REPAIR,
    )


def reset_memory_for_repair(
    memory_effect: MemoryEffectResolution,
    repair: RepairResolution,
) -> MemoryEffectResolution:
    if not repair_invalidates(repair, "memory"):
        return memory_effect
    return MemoryEffectResolution(
        action="IGNORE",
        reason="invalidated_by_repair",
        status=NOT_APPLICABLE,
        source=SOURCE_REPAIR,
    )


def reset_ambiguity_for_repair(
    ambiguity: AmbiguityResolution,
    repair: RepairResolution,
) -> AmbiguityResolution:
    if not repair_invalidates(repair, "ambiguity"):
        return ambiguity
    return AmbiguityResolution(
        reason="invalidated_by_repair",
        status=NOT_APPLICABLE,
        source=SOURCE_REPAIR,
    )


def apply_repair_invalidation(
    *,
    repair: RepairResolution,
    reference: ReferenceResolution,
    ellipsis: EllipsisResolution,
    memory_effect: MemoryEffectResolution | None = None,
    ambiguity: AmbiguityResolution | None = None,
) -> tuple[ReferenceResolution, EllipsisResolution, MemoryEffectResolution | None, AmbiguityResolution | None, list[str]]:
    invalidated: list[str] = []
    if repair.status != RESOLVED:
        return reference, ellipsis, memory_effect, ambiguity, invalidated
    if repair.kind == "REFERENT":
        old_ids = reference.user_ids
        reference = apply_repair_to_reference(reference, repair)
        if reference.user_ids != old_ids or repair_invalidates(repair, "referent"):
            invalidated.append("referent")
    if repair_invalidates(repair, "ellipsis") and (
        ellipsis.kind not in {"", "NONE"} or ellipsis.status == RESOLVED or ellipsis.source_text
    ):
        ellipsis = reset_ellipsis_for_repair(ellipsis, repair)
        invalidated.append("ellipsis")
    if memory_effect is not None and repair_invalidates(repair, "memory") and memory_effect.action not in {"", "IGNORE"}:
        memory_effect = reset_memory_for_repair(memory_effect, repair)
        invalidated.append("memory")
    if ambiguity is not None and repair_invalidates(repair, "ambiguity") and ambiguity.kind not in {"", "NONE"}:
        ambiguity = reset_ambiguity_for_repair(ambiguity, repair)
        invalidated.append("ambiguity")
    return reference, ellipsis, memory_effect, ambiguity, invalidated


def should_ask_jev_memory_effect(
    text: str,
    *,
    repair: RepairResolution | None = None,
    candidate: MemoryCandidate | None = None,
) -> bool:
    if repair is not None and repair.kind in {"FACT", "RETRACTION"} and not repair.unresolved:
        return True
    if candidate is None or not candidate.content.strip():
        return False
    return bool(FACT_CUE_RE.search(str(text or "")))


def related_memories_for_candidate(
    memory: MemoryStore,
    *,
    group_id: int,
    candidate: MemoryCandidate,
    speaker_user_id: int | None = None,
    limit: int = MAX_MEMORY_RELATED,
) -> list[MemoryAtom]:
    subject_ids = [candidate.subject_user_id] if candidate.subject_user_id else []
    return memory.relevant_memory_atoms(
        group_id,
        candidate.content,
        subject_user_ids=subject_ids,
        speaker_user_id=speaker_user_id,
        limit=limit,
    )


def format_memory_effect_state(
    *,
    current_text: str,
    candidate: MemoryCandidate,
    related: Iterable[MemoryAtom],
) -> str:
    lines = [
        f"【当前消息】{(current_text or '')[:240]}",
        "【新记忆候选】",
        f"subject={candidate.subject_user_id or 'NONE'}",
        f"content={candidate.content[:160]}",
        f"source={candidate.source_message_id or ''}",
        "【相关旧记忆】",
    ]
    related_rows = list(related)[:MAX_MEMORY_RELATED]
    if not related_rows:
        lines.append("- 无")
    for atom in related_rows:
        lines.append(f"- mem{atom.id} | {atom.content[:80]} | src={atom.source} | recency={atom.updated_at}")
    return "\n".join(lines)


def memory_target_criteria(related: Iterable[MemoryAtom]) -> dict[str, str]:
    criteria = {"NONE": "没有对应旧记忆，CREATE 时必须选这个"}
    for atom in list(related)[:MAX_MEMORY_RELATED]:
        criteria[f"mem{atom.id}"] = atom.content[:60]
    return criteria


def parse_jev_memory_answers(data: dict) -> MemoryEffectJudgement:
    answers = _answers(data)
    action = _choice(answers, "memory_action", MEMORY_ACTIONS, "IGNORE")
    target = str((answers.get("memory_target") or {}).get("choice") or "NONE").strip()
    if action == "CREATE":
        target = "NONE"
    if action == "IGNORE":
        target = "NONE"
    confidence = (
        choice_confidence(answers, "memory_action")
        if action in {"CREATE", "IGNORE"}
        else joint_choice_confidence(answers, "memory_action", "memory_target")
    )
    return MemoryEffectJudgement(
        action=action,
        target_id=target,
        confidence=float(confidence or 0.0),
        reason=f"jev_{action.lower()}_{target}" if confidence is not None else "error",
    )


def apply_jev_memory_judgement(
    judgement: MemoryEffectJudgement | None,
    related: Iterable[MemoryAtom],
    *,
    candidate: MemoryCandidate | None = None,
) -> MemoryEffectResolution:
    if judgement is None:
        return MemoryEffectResolution(reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    action = judgement.action if judgement.action in MEMORY_ACTIONS else "IGNORE"
    if action == "IGNORE":
        return MemoryEffectResolution(
            action="IGNORE",
            reason=judgement.reason or "ignore",
            status=NOT_APPLICABLE,
            source=SOURCE_JEV,
        )
    if float(judgement.confidence or 0.0) < JEV_MEMORY_CONFIDENCE_MIN:
        return MemoryEffectResolution(
            action=action,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_low_confidence",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    target_atom_id = None
    if judgement.target_id not in {"", "NONE"}:
        raw = str(judgement.target_id).removeprefix("mem")
        try:
            target_atom_id = int(raw)
        except ValueError:
            return MemoryEffectResolution(
                action=action,
                unresolved=True,
                reason="jev_unknown_target",
                confidence=float(judgement.confidence or 0.0),
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
        known = {int(atom.id) for atom in related}
        if target_atom_id not in known:
            return MemoryEffectResolution(
                action=action,
                unresolved=True,
                reason="jev_unknown_target",
                confidence=float(judgement.confidence or 0.0),
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
    if action == "CREATE" and target_atom_id is not None:
        target_atom_id = None
    if action in {"CONFIRM", "REFINE", "REPLACE", "NEGATE", "RETRACT"} and target_atom_id is None:
        return MemoryEffectResolution(
            action=action,
            unresolved=True,
            reason="missing_target",
            confidence=float(judgement.confidence or 0.0),
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if candidate is None or not candidate.content.strip():
        if action != "RETRACT":
            return MemoryEffectResolution(
                action="IGNORE",
                reason="no_candidate",
                status=NOT_APPLICABLE,
                source=SOURCE_JEV,
            )
    return MemoryEffectResolution(
        action=action,
        target_atom_id=target_atom_id,
        confidence=float(judgement.confidence or 0.0),
        reason=judgement.reason or "jev_memory",
        status=RESOLVED,
        source=SOURCE_JEV,
        value=action,
    )


def apply_memory_effect(
    memory: MemoryStore,
    resolution: MemoryEffectResolution,
    *,
    group_id: int,
    candidate: MemoryCandidate,
    actor_user_id: int | None = None,
    source: str = "jev_memory_effect",
) -> MemoryEffectResolution:
    if resolution.status != RESOLVED or resolution.unresolved or resolution.action == "IGNORE":
        return resolution
    action = resolution.action
    target_id = resolution.target_atom_id
    source_message_id = candidate.source_message_id or None
    content = candidate.content.strip()
    applied = False
    if action == "CREATE":
        atom_id = memory.add_memory_atom(
            atom_type="fact",
            group_id=group_id,
            content=content,
            source=source,
            evidence_type="message",
            source_message_id=source_message_id,
            subject_user_id=candidate.subject_user_id,
            confidence=max(candidate.confidence, resolution.confidence),
            actor_user_id=actor_user_id,
            audit_detail="CREATE from current message",
        )
        applied = bool(atom_id)
        return replace(resolution, applied=applied, target_atom_id=atom_id or target_id)
    if target_id is None:
        return replace(resolution, unresolved=True, reason="missing_target", status=AMBIGUOUS)
    if action == "CONFIRM":
        old = memory.memory_atom(target_id)
        if old is None:
            return replace(resolution, unresolved=True, reason="missing_target", status=AMBIGUOUS)
        atom_id = memory.upsert_memory_atom(
            atom_type=old.atom_type,
            group_id=group_id,
            content=old.content,
            source=source,
            subject_user_id=old.subject_user_id,
            object_user_id=old.object_user_id,
            confidence=max(old.confidence, resolution.confidence),
            importance=old.importance,
            evidence_type="message",
            source_message_id=source_message_id,
        )
        applied = bool(atom_id)
    elif action in {"REFINE", "REPLACE"}:
        atom_id = memory.correct_memory_atom(
            target_id,
            content=content,
            source=source,
            source_message_id=source_message_id,
            actor_user_id=actor_user_id,
            reason=action,
            confidence=resolution.confidence,
            subject_user_id=candidate.subject_user_id,
        )
        applied = bool(atom_id)
        return replace(resolution, applied=applied, target_atom_id=atom_id or target_id)
    elif action == "NEGATE":
        expired = memory.expire_memory_atom(
            target_id,
            reason="NEGATE: world state changed",
            source=source,
            actor_user_id=actor_user_id,
        )
        atom_id = 0
        if expired and content:
            atom_id = memory.add_memory_atom(
                atom_type="fact",
                group_id=group_id,
                content=content,
                source=source,
                evidence_type="message",
                source_message_id=source_message_id,
                subject_user_id=candidate.subject_user_id,
                confidence=resolution.confidence,
                actor_user_id=actor_user_id,
                audit_detail=f"NEGATE of {target_id}",
            )
        applied = bool(expired)
        return replace(resolution, applied=applied, target_atom_id=atom_id or target_id)
    elif action == "RETRACT":
        applied = bool(
            memory.expire_memory_atom(
                target_id,
                reason="RETRACT: speaker withdrew assertion",
                source=source,
                actor_user_id=actor_user_id,
            )
        )
    return replace(resolution, applied=applied)



def memory_can_commit(
    resolution: MemoryEffectResolution,
    *,
    candidate: MemoryCandidate | None,
    reference: ReferenceResolution,
    repair: RepairResolution,
    ambiguity: AmbiguityResolution,
    pending_recompute: Iterable[str] = (),
    critic=None,
) -> bool:
    if candidate is None or not str(candidate.content or "").strip():
        return False
    if not has_usable_value(resolution):
        return False
    if resolution.unresolved or resolution.action in {"", "IGNORE"}:
        return False
    if is_unresolved(repair) or is_unresolved(reference):
        return False
    if ambiguity.status == AMBIGUOUS and ambiguity.kind in {"PERSON", "MEMORY_REFERENCE"}:
        return False
    if tuple(pending_recompute):
        return False
    if critic is not None and getattr(critic, "status", "") == RESOLVED:
        from .pre_send_critic import critic_blocks_memory

        if critic_blocks_memory(critic):
            return False
    if reference.kind == "PERSON" and not reference.user_ids and resolution.action != "RETRACT":
        return False
    return True


def should_ask_jev_ambiguity(
    *,
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
    repair: RepairResolution | None = None,
    memory_effect: MemoryEffectResolution | None = None,
    speaking_action: str = "",
    extra_unresolved: bool = False,
) -> bool:
    if extra_unresolved:
        return True
    if (
        reference is not None
        and reference.status == RESOLVED
        and reference.kind == "NON_PERSON"
        and ellipsis is not None
        and str(getattr(ellipsis, "kind", "") or "") in {"", "NONE"}
        and not bool(getattr(ellipsis, "unresolved", False))
    ):
        # "这个" as a thing is not a missing-media / missing-person clarify cue.
        return False
    def _ambiguous(obj) -> bool:
        if obj is None:
            return False
        status = str(getattr(obj, "status", "") or "")
        if status in {UNAVAILABLE, ERROR, NOT_APPLICABLE}:
            return False
        return status == AMBIGUOUS or bool(getattr(obj, "unresolved", False))

    if _ambiguous(reference):
        return True
    if reference is not None and (
        reference.status == RESOLVED
        and reference.kind == "PERSON"
        and not reference.user_ids
    ):
        return True
    if _ambiguous(ellipsis) or _ambiguous(repair):
        return True
    if speaking_action == "clarify":
        return True
    if (
        reference is not None
        and reference.kind == "PERSON"
        and 0 < reference.confidence < 0.55
    ):
        return True
    if ellipsis is not None and 0 < ellipsis.confidence < 0.60 and ellipsis.kind not in {"", "NONE"}:
        return True
    return False


def format_ambiguity_jev_state(
    *,
    current_text: str,
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
    repair: RepairResolution | None = None,
    candidate_labels: Iterable[str] = (),
) -> str:
    lines = [f"【当前消息】{(current_text or '')[:240]}"]
    if reference is not None:
        lines.append(
            f"【指代】kind={reference.kind} ids={reference.user_ids} "
            f"unresolved={reference.unresolved} conf={reference.confidence:.2f}"
        )
    if ellipsis is not None:
        lines.append(
            f"【省略】kind={ellipsis.kind} unresolved={ellipsis.unresolved} "
            f"source={ellipsis.source_text[:40]}"
        )
    if repair is not None:
        lines.append(f"【修正】kind={repair.kind} unresolved={repair.unresolved}")
    labels = [str(item) for item in candidate_labels if str(item).strip()]
    if labels:
        lines.append("【候选】" + "、".join(labels[:6]))
    return "\n".join(lines)


def parse_jev_ambiguity_answers(data: dict) -> AmbiguityJudgement:
    answers = _answers(data)
    kind = _choice(answers, "ambiguity_kind", AMBIGUITY_KINDS, "NONE")
    confidence = choice_confidence(answers, "ambiguity_kind")
    return AmbiguityJudgement(
        kind=kind,
        confidence=float(confidence or 0.0),
        reason=f"jev_{kind.lower()}" if confidence is not None else "error",
    )


def apply_jev_ambiguity_judgement(
    judgement: AmbiguityJudgement | None,
    *,
    candidate_labels: Iterable[str] = (),
) -> AmbiguityResolution:
    if judgement is None:
        return AmbiguityResolution(reason="jev_unavailable", status=UNAVAILABLE, source=SOURCE_JEV)
    kind = judgement.kind if judgement.kind in AMBIGUITY_KINDS else "NONE"
    labels = tuple(str(item) for item in candidate_labels if str(item).strip())[:6]
    if kind == "NONE":
        return AmbiguityResolution(
            kind="NONE",
            confidence=float(judgement.confidence or 0.0),
            candidates=labels,
            status=NOT_APPLICABLE,
            source=SOURCE_JEV,
        )
    if float(judgement.confidence or 0.0) < JEV_AMBIGUITY_CONFIDENCE_MIN:
        return AmbiguityResolution(
            kind=kind,
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            candidates=labels,
            reason="jev_low_confidence",
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    return AmbiguityResolution(
        kind=kind,
        confidence=float(judgement.confidence or 0.0),
        unresolved=kind != "NONE",
        candidates=labels,
        reason=judgement.reason or "jev_ambiguity",
        status=AMBIGUOUS if kind != "NONE" else NOT_APPLICABLE,
        source=SOURCE_JEV,
        value=kind,
    )


def format_repair_prompt_block(repair: RepairResolution) -> str:
    if repair.status == UNAVAILABLE:
        return "[repair]\nstatus=UNAVAILABLE\n不要编造被纠正后的对象或意图。"
    if repair.kind in {"", "NONE"} and not repair.unresolved:
        return ""
    lines = ["[repair]", f"kind={repair.kind}", f"status={repair.status}"]
    if repair.unresolved or repair.status == AMBIGUOUS:
        lines.append("repair_unresolved=true")
        lines.append("不要编造被纠正后的对象或意图。")
        return "\n".join(lines)
    if repair.replacement_user_ids:
        lines.append("replacement_user_ids=" + ",".join(str(uid) for uid in repair.replacement_user_ids))
    if repair.replacement_item:
        lines.append(f'item="{repair.replacement_item}"')
    return "\n".join(lines)


def format_ambiguity_prompt_block(ambiguity: AmbiguityResolution) -> str:
    if ambiguity.status == UNAVAILABLE:
        return "[ambiguity]\nstatus=UNAVAILABLE\n不要把检查失败当成没有歧义，也不要装懂。"
    if ambiguity.status == ERROR:
        return "[ambiguity]\nstatus=ERROR\n不要把程序异常当成已经消歧。"
    if ambiguity.kind in {"", "NONE"}:
        return ""
    lines = ["[ambiguity]", f"kind={ambiguity.kind}", f"status={ambiguity.status}"]
    if ambiguity.candidates:
        quoted = ",".join(f'"{item}"' for item in ambiguity.candidates)
        lines.append(f"candidates=[{quoted}]")
    if ambiguity.unresolved:
        lines.append("unresolved=true")
    return "\n".join(lines)


def _answers(data: dict) -> dict:
    answers = data.get("answers") if isinstance(data, dict) else {}
    return answers if isinstance(answers, dict) else {}


def _choice(answers: dict, key: str, allowed: tuple[str, ...], default: str) -> str:
    raw = answers.get(key) if isinstance(answers.get(key), dict) else {}
    value = str(raw.get("choice") or default).strip().upper()
    return value if value in allowed else default


def _noul(answers: dict, key: str) -> float:
    raw = answers.get(key) if isinstance(answers.get(key), dict) else {}
    try:
        return float(raw.get("noul") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _unique_named_users(resolve_named_users: NamedUserResolver, text: str) -> tuple[int, ...]:
    try:
        return tuple(dict.fromkeys(int(value) for value in resolve_named_users(text) if int(value) > 0))
    except Exception:
        return ()
