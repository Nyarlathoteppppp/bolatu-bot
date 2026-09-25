from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .jev_policy import JEV_PERSON_CONFIDENCE_MIN
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
    joint_choice_confidence,
)


HIGH_CONFIDENCE_SKIP_JEV = 0.85
MAX_REFERENT_CANDIDATES = 10
BOT_SELF_ALIASES = ("张风雪", "风雪")

STRONG_PERSON_RE = re.compile(
    r"(他|她|他们|她们|那个人|这个人|这位群友|那位群友|这人|那人|这鸟|那鸟|你们说的那个)"
)
WEAK_REFERENCE_RE = re.compile(r"(那个|这个|刚才那个|后来呢|然后呢|那后来呢|\b它\b|它)")
ELLIPTICAL_FOLLOWUP_RE = re.compile(
    r"^(?:那|然后|所以|那么)?(?:后来|现在|以前|之后|再后来)?(?:呢|怎么样了?|还.+吗|又.+吗|接着呢)[？?~～。！!]*$"
)

_SOURCE_RANK = {
    "bot": 0,
    "reply": 1,
    "at": 2,
    "reply_mention": 3,
    "mentioned": 4,
    "speaker": 5,
    "relation": 6,
}


@dataclass(frozen=True)
class ReplyHint:
    exists: bool = False
    author_id: int | None = None
    author_label: str = ""
    text: str = ""
    message_id: str = ""


@dataclass(frozen=True)
class ReferentCandidate:
    key: str
    user_id: int
    label: str
    sources: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    is_bot: bool = False
    rule_prior: str = ""
    rule_confidence: float = 0.0


@dataclass(frozen=True)
class ReferentJudgement:
    kind: str = "NONE"
    person_key: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ReferenceResolution:
    user_ids: tuple[int, ...] = ()
    expanded_query: str = ""
    reason: str = "none"
    confidence: float = 0.0
    kind: str = "NONE"
    unresolved: bool = False
    status: str = ""
    source: str = ""
    value: str = ""

    def __post_init__(self) -> None:
        status = str(self.status or "")
        reason = str(self.reason or "")
        if not status:
            if "jev_unavailable" in reason or reason == "error":
                status = UNAVAILABLE if "jev_unavailable" in reason else ERROR
            elif self.unresolved or reason in {
                "jev_ambiguous",
                "jev_low_confidence",
                "jev_unknown_target",
                "ambiguous",
            }:
                status = AMBIGUOUS
            elif self.user_ids or self.kind == "NON_PERSON":
                status = RESOLVED
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        source = str(self.source or "")
        if not source:
            if reason.startswith("jev_"):
                source = SOURCE_JEV
            elif reason.startswith("repair_"):
                source = SOURCE_REPAIR
            else:
                source = SOURCE_RULE
            object.__setattr__(self, "source", source)
        finalize_result(self, has_value=bool(self.user_ids) or self.kind == "NON_PERSON")


NamedUserResolver = Callable[[str], Iterable[int]]


def has_strong_person_reference(text: str) -> bool:
    return bool(STRONG_PERSON_RE.search(str(text) or ""))


def has_weak_reference_cue(text: str) -> bool:
    clean = str(text or "")
    return bool(WEAK_REFERENCE_RE.search(clean) or ELLIPTICAL_FOLLOWUP_RE.match(clean.strip()))


def has_reference_trigger(text: str) -> bool:
    return has_strong_person_reference(text) or has_weak_reference_cue(text)


def should_ask_jev_referent(text: str, resolution: ReferenceResolution) -> bool:
    if resolution.confidence >= HIGH_CONFIDENCE_SKIP_JEV and resolution.user_ids:
        return False
    if not has_reference_trigger(text):
        return False
    return True


def resolve_context_reference(
    text: str,
    recent_messages: Iterable[object],
    *,
    current_user_id: int,
    resolve_named_users: NamedUserResolver | None = None,
) -> ReferenceResolution:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    strong = has_strong_person_reference(clean)
    weak = has_weak_reference_cue(clean)
    if not strong and not weak:
        return ReferenceResolution()

    messages = tuple(recent_messages)
    if resolve_named_users is not None:
        current_named = _unique_named_ids(resolve_named_users, clean)
        if strong and len(current_named) == 1:
            return ReferenceResolution(
                current_named,
                _expanded_query(clean, clean),
                "named_in_current",
                0.92,
                kind="PERSON",
            )
        nearest_text = ""
        for message in reversed(messages):
            if bool(getattr(message, "is_bot", False)):
                continue
            previous_text = str(getattr(message, "text", "") or "").strip()
            if not previous_text or previous_text == clean:
                continue
            nearest_text = previous_text
            break
        if nearest_text:
            resolved = _unique_named_ids(resolve_named_users, nearest_text)
            if len(resolved) == 1:
                return ReferenceResolution(
                    resolved,
                    _expanded_query(clean, nearest_text),
                    "previous_named_member",
                    0.78,
                    kind="PERSON",
                )
        # Named-in-current is the only high-confidence skip. A previous named
        # member is a prior, not a lock: later speakers can change the referent.

    if not strong:
        return ReferenceResolution(reason="none", kind="NONE")

    for message in reversed(messages):
        if bool(getattr(message, "is_bot", False)):
            continue
        user_id = int(getattr(message, "user_id", 0) or 0)
        previous_text = str(getattr(message, "text", "") or "").strip()
        if user_id <= 0 or user_id == current_user_id or not previous_text:
            continue
        return ReferenceResolution(
            (user_id,),
            _expanded_query(clean, previous_text),
            "latest_other_speaker",
            0.72,
            kind="PERSON",
        )
    return ReferenceResolution(reason="ambiguous", kind="PERSON", unresolved=True)


def build_referent_candidates(
    recent_messages: Iterable[object],
    *,
    current_user_id: int,
    current_text: str,
    self_id: int,
    self_label: str = "张风雪",
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
    resolve_named_users: NamedUserResolver | None = None,
    relation_user_ids: Iterable[int] = (),
    rule_guess: ReferenceResolution | None = None,
    max_candidates: int = MAX_REFERENT_CANDIDATES,
) -> list[ReferentCandidate]:
    messages = tuple(recent_messages)
    names = _nickname_map(messages, self_id=self_id, self_label=self_label)
    if reply and reply.author_id:
        names.setdefault(int(reply.author_id), _label_name(reply.author_label, reply.author_id))

    pooled: dict[int, set[str]] = {}
    mention_snips: dict[int, list[str]] = {}

    def add(user_id: int, source: str, snippet: str = "") -> None:
        uid = int(user_id or 0)
        if uid <= 0 or uid == current_user_id:
            return
        pooled.setdefault(uid, set()).add(source)
        clean = re.sub(r"\s+", " ", str(snippet or "")).strip()
        if clean:
            bucket = mention_snips.setdefault(uid, [])
            item = clean[:80]
            if item not in bucket:
                bucket.append(item)

    if self_id > 0:
        pooled.setdefault(int(self_id), set()).add("bot")
        names.setdefault(int(self_id), self_label)

    if reply and reply.exists and reply.author_id:
        add(reply.author_id, "reply")
        if resolve_named_users is not None and reply.text:
            for uid in resolve_named_users(reply.text):
                add(uid, "reply_mention", reply.text)

    for uid in at_user_ids:
        add(uid, "at")

    if resolve_named_users is not None:
        for msg in messages[-16:]:
            blob = str(getattr(msg, "text", "") or "").strip()
            if not blob:
                continue
            speaker = str(getattr(msg, "nickname", "") or getattr(msg, "user_id", ""))
            for uid in resolve_named_users(blob):
                add(uid, "mentioned", f"{speaker}：{blob}")
        if current_text.strip():
            for uid in resolve_named_users(current_text):
                add(uid, "mentioned", current_text)

    for msg in reversed(messages):
        uid = int(getattr(msg, "user_id", 0) or 0)
        if uid <= 0:
            continue
        if bool(getattr(msg, "is_bot", False)) or uid == self_id:
            add(uid, "bot")
            continue
        add(uid, "speaker")

    for uid in relation_user_ids:
        add(uid, "relation")

    ordered_ids = sorted(
        pooled,
        key=lambda uid: (
            min(_SOURCE_RANK.get(src, 9) for src in pooled[uid]),
            0 if uid == self_id else 1,
            -uid,
        ),
    )
    if self_id in pooled:
        ordered_ids = [self_id] + [uid for uid in ordered_ids if uid != self_id]
    ordered_ids = ordered_ids[: max(1, int(max_candidates))]

    guess_ids = set(rule_guess.user_ids) if rule_guess is not None else set()
    rows: list[ReferentCandidate] = []
    for uid in ordered_ids:
        is_bot = uid == self_id
        nick = names.get(uid, str(uid))
        label = _member_label(uid, "张风雪" if is_bot else nick)
        sources = tuple(sorted(pooled[uid], key=lambda src: _SOURCE_RANK.get(src, 9)))
        evidence = _evidence_for_user(
            uid,
            nick,
            messages,
            is_bot=is_bot,
            reply=reply,
            mention_snips=tuple(mention_snips.get(uid, ())),
        )
        prior = ""
        prior_conf = 0.0
        if rule_guess is not None and uid in guess_ids:
            prior = rule_guess.reason
            prior_conf = rule_guess.confidence
        rows.append(
            ReferentCandidate(
                key=f"u{uid}",
                user_id=uid,
                label=label,
                sources=sources,
                evidence=evidence,
                is_bot=is_bot,
                rule_prior=prior,
                rule_confidence=prior_conf,
            )
        )
    return rows


def apply_jev_referent_judgement(
    judgement: ReferentJudgement | None,
    candidates: Iterable[ReferentCandidate],
    *,
    current_text: str,
    fallback: ReferenceResolution,
) -> ReferenceResolution:
    if judgement is None or judgement.reason == "error":
        if fallback.confidence >= HIGH_CONFIDENCE_SKIP_JEV and fallback.user_ids:
            return fallback
        kind = "PERSON" if has_strong_person_reference(current_text) else "NONE"
        return ReferenceResolution(
            reason="jev_unavailable" if judgement is None else "error",
            kind=kind,
            unresolved=True,
            confidence=0.0,
            status=UNAVAILABLE if judgement is None else ERROR,
            source=SOURCE_JEV,
        )

    kind = str(judgement.kind or "NONE").upper()
    if kind not in {"PERSON", "NON_PERSON", "NONE", "OTHER"}:
        return ReferenceResolution(reason="error", status=ERROR, source=SOURCE_JEV)
    if kind == "OTHER" or float(judgement.confidence or 0.0) < JEV_PERSON_CONFIDENCE_MIN:
        return ReferenceResolution(
            kind="PERSON" if has_strong_person_reference(current_text) else "NONE",
            reason="jev_other" if kind == "OTHER" else "jev_low_confidence",
            confidence=float(judgement.confidence or 0.0),
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if kind != "PERSON":
        return ReferenceResolution(
            reason="jev_non_person" if kind == "NON_PERSON" else "jev_none",
            kind=kind,
            confidence=float(judgement.confidence or 0.0),
            status=RESOLVED if kind == "NON_PERSON" else NOT_APPLICABLE,
            source=SOURCE_JEV,
            value=kind,
        )

    key = str(judgement.person_key or "NONE").strip()
    if key.upper() in {"", "NONE", "NONE"}:
        return ReferenceResolution(
            reason="jev_ambiguous",
            kind="PERSON",
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    if float(judgement.confidence or 0.0) < JEV_PERSON_CONFIDENCE_MIN:
        return ReferenceResolution(
            reason="jev_low_confidence",
            kind="PERSON",
            unresolved=True,
            confidence=float(judgement.confidence or 0.0),
            status=AMBIGUOUS,
            source=SOURCE_JEV,
        )
    for candidate in candidates:
        if candidate.key == key:
            evidence = candidate.evidence[-1] if candidate.evidence else candidate.label
            return ReferenceResolution(
                (candidate.user_id,),
                _expanded_query(current_text, evidence),
                judgement.reason or "jev_referent",
                max(float(judgement.confidence), JEV_PERSON_CONFIDENCE_MIN),
                kind="PERSON",
                status=RESOLVED,
                source=SOURCE_JEV,
                value=str(candidate.user_id),
            )
    return ReferenceResolution(
        reason="jev_unknown_target",
        kind="PERSON",
        unresolved=True,
        confidence=float(judgement.confidence or 0.0),
        status=AMBIGUOUS,
        source=SOURCE_JEV,
    )


def format_referent_prompt_block(resolution: ReferenceResolution) -> str:
    if resolution.status == UNAVAILABLE:
        return (
            "[referent]\n"
            "status=UNAVAILABLE\n"
            "不要把指代检查失败当成没有指代，也不要猜人。"
        )
    if resolution.status == ERROR:
        return (
            "[referent]\n"
            "status=ERROR\n"
            "不要把程序异常当成没有指代。"
        )
    if resolution.status == AMBIGUOUS or resolution.unresolved:
        return (
            "[referent]\n"
            f"status={resolution.status}\n"
            "unresolved_reference=true\n"
            "不要点名或套用某人画像。"
        )
    if resolution.status == RESOLVED and resolution.user_ids:
        return (
            "[referent]\n"
            f"status={RESOLVED}\n"
            "user_ids=" + ",".join(str(uid) for uid in resolution.user_ids) + "\n"
            f"reason={resolution.reason}"
        )
    if resolution.kind == "NON_PERSON":
        return f"[referent]\nstatus={resolution.status}\nkind=NON_PERSON"
    return ""


def format_referent_jev_state(
    *,
    current_text: str,
    current_label: str,
    candidates: Iterable[ReferentCandidate],
    reply: ReplyHint | None = None,
    rule_guess: ReferenceResolution | None = None,
) -> str:
    lines = [
        f"【当前发言人】{current_label}",
        f"【当前消息】{(current_text or '')[:400]}",
    ]
    if reply is not None and reply.exists:
        lines.append("【QQ回复】exists=true")
        lines.append(f"reply.author={reply.author_label or reply.author_id}")
        lines.append(f"reply.message_id={reply.message_id or ''}")
        lines.append(f"reply.text={(reply.text or '')[:180]}")
        lines.append("回复作者是强信号，但不能直接当成指代对象。")
    else:
        lines.append("【QQ回复】exists=false")
    if rule_guess is not None and rule_guess.user_ids and rule_guess.confidence < HIGH_CONFIDENCE_SKIP_JEV:
        guess_keys = ",".join(f"u{uid}" for uid in rule_guess.user_ids)
        lines.append(
            f"【规则弱猜测】rule_guess={guess_keys} "
            f"confidence={rule_guess.confidence:.2f} reason={rule_guess.reason}"
        )
        lines.append("这只是 prior，可被证据推翻；不要机械绑定回复对象或最近说话人。")
    lines.append("【候选人】")
    for candidate in list(candidates)[:MAX_REFERENT_CANDIDATES]:
        src = ",".join(candidate.sources) or "unknown"
        lines.append(f"- {candidate.key} | {candidate.label}")
        lines.append(f"  sources: {src}")
        if candidate.rule_prior:
            lines.append(
                f"  rule_prior: {candidate.rule_prior} ({candidate.rule_confidence:.2f})"
            )
        if candidate.evidence:
            lines.append("  evidence:")
            for item in candidate.evidence:
                lines.append(f"  - {item}")
        elif candidate.is_bot:
            lines.append("  evidence:")
            lines.append("  - 风雪自己：风雪/张风雪")
    return "\n".join(lines)


def referent_choice_criteria(candidates: Iterable[ReferentCandidate]) -> dict[str, str]:
    criteria = {
        "NONE": "没有明确的人物对象，或 kind 不是 PERSON 时必须选这个",
    }
    for candidate in list(candidates)[:MAX_REFERENT_CANDIDATES]:
        if candidate.is_bot:
            criteria[candidate.key] = "指的是风雪/张风雪自己（常驻候选）"
        else:
            criteria[candidate.key] = f"指的是 {candidate.label}"
    return criteria


def parse_jev_referent_answers(data: dict) -> ReferentJudgement:
    answers = data.get("answers") if isinstance(data, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    kind_raw = answers.get("reference_kind") if isinstance(answers.get("reference_kind"), dict) else {}
    target_raw = answers.get("person_target") if isinstance(answers.get("person_target"), dict) else {}
    kind_choice = str(kind_raw.get("choice") or "").strip().upper()
    if not kind_choice:
        return ReferentJudgement(kind="NONE", person_key="NONE", confidence=0.0, reason="error")
    if kind_choice not in {"PERSON", "NON_PERSON", "NONE", "OTHER"}:
        return ReferentJudgement(reason="error")
    kind = kind_choice
    person_key = str(target_raw.get("choice") or "NONE").strip() or "NONE"
    if kind != "PERSON":
        person_key = "NONE"
    confidence = (
        joint_choice_confidence(answers, "reference_kind", "person_target")
        if kind == "PERSON"
        else choice_confidence(answers, "reference_kind")
    )
    if confidence is None:
        return ReferentJudgement(kind=kind, person_key=person_key, confidence=0.0, reason="error")
    return ReferentJudgement(
        kind=kind,
        person_key=person_key,
        confidence=confidence,
        reason=f"jev_{kind.lower()}_{person_key}",
    )


def _unique_named_ids(resolve_named_users: NamedUserResolver, text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(value) for value in resolve_named_users(text) if int(value) > 0))
    except Exception:
        return ()
    return values


def _expanded_query(current: str, previous: str) -> str:
    previous = re.sub(r"\s+", " ", str(previous)).strip()
    if len(previous) > 140:
        previous = previous[-140:]
    return f"{current}（承接前文：{previous}）"


def _member_label(user_id: int, nickname: str) -> str:
    clean_name = (nickname or "").strip() or str(user_id)
    return f"{clean_name}[#{str(user_id)[-5:]}]"


def _label_name(label: str, user_id: int | None) -> str:
    text = str(label or "").strip()
    if "[#" in text:
        text = text.split("[#", 1)[0].strip()
    return text or (str(user_id) if user_id else "")


def _nickname_map(messages: tuple[object, ...], *, self_id: int, self_label: str) -> dict[int, str]:
    names: dict[int, str] = {}
    if self_id > 0:
        names[int(self_id)] = self_label
    for msg in messages:
        uid = int(getattr(msg, "user_id", 0) or 0)
        nick = str(getattr(msg, "nickname", "") or "").strip()
        if uid > 0 and nick and uid not in names:
            names[uid] = nick
        if bool(getattr(msg, "is_bot", False)) and uid > 0:
            names[uid] = self_label
    return names


def _evidence_for_user(
    user_id: int,
    nickname: str,
    messages: tuple[object, ...],
    *,
    is_bot: bool,
    reply: ReplyHint | None,
    mention_snips: tuple[str, ...] = (),
    limit: int = 2,
) -> tuple[str, ...]:
    lines: list[str] = []
    nick = (nickname or "").strip()
    aliases = {nick, *BOT_SELF_ALIASES} if is_bot else {nick}
    aliases.discard("")
    for snippet in mention_snips:
        item = snippet if "：" in snippet else f"提及：{snippet}"
        if item not in lines:
            lines.append(item[:90])
    if reply and reply.exists and reply.text:
        blob = reply.text.strip()
        if blob and (reply.author_id == user_id or any(alias and alias in blob for alias in aliases)):
            author = reply.author_label or "被回复消息"
            item = f"{author}：{blob[:80]}"
            if item not in lines:
                lines.append(item)
    for msg in reversed(messages):
        if len(lines) >= limit:
            break
        text = str(getattr(msg, "text", "") or "").strip()
        if not text:
            continue
        uid = int(getattr(msg, "user_id", 0) or 0)
        speaker = str(getattr(msg, "nickname", "") or uid)
        if uid == user_id or (is_bot and bool(getattr(msg, "is_bot", False))):
            who = "风雪" if is_bot or bool(getattr(msg, "is_bot", False)) else speaker
            item = f"{who}：{text[:80]}"
            if item not in lines:
                lines.append(item)
            continue
        if any(alias and alias in text for alias in aliases):
            item = f"{speaker}：{text[:80]}"
            if item not in lines:
                lines.append(item)
    return tuple(reversed(lines[-limit:]))
