from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from .discourse_effects import (
    AmbiguityResolution,
    MemoryEffectResolution,
    RepairResolution,
    apply_jev_ambiguity_judgement,
    apply_jev_repair_judgement,
    apply_repair_invalidation,
    build_repair_targets,
    should_ask_jev_ambiguity,
    should_ask_jev_repair,
)
from .ellipsis_resolver import (
    EllipsisJudgement,
    EllipsisResolution,
    EllipsisSource,
    apply_jev_ellipsis_judgement,
    build_ellipsis_candidates,
    semantic_message_text,
    should_ask_jev_ellipsis,
    source_key_for_message,
)
from .message_segments import is_marketface_segment, message_segments_from_payload, segment_type_and_data
from .reference_resolver import (
    ReferenceResolution,
    ReplyHint,
    apply_jev_referent_judgement,
    build_referent_candidates,
    resolve_context_reference,
    should_ask_jev_referent,
)
from .resolver_result import (
    AMBIGUOUS,
    ERROR,
    NONE,
    NOT_APPLICABLE,
    RESOLVED,
    SOURCE_JEV,
    SOURCE_RULE,
    UNAVAILABLE,
    choice_confidence,
    finalize_result,
)


from .jev_policy import JEV_ADDRESSEE_CONFIDENCE_MIN, JEV_AUDIT_CONFIDENCE_MIN

MAX_DEIXIS_CANDIDATES = 8
MAX_LOGICAL_GAP_SECONDS = 90.0
SAME_SPEAKER_BURST = 3
REAL_MEDIA_TYPES = frozenset({"image", "file", "video"})
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MEDIA_ASK_RE = re.compile(
    r"(把图发一下|发一下截图|发个图|发张图|截图呢|链接呢|图片没看到|没看到图|图片呢|图呢|缺图|把链接|没看到链接)"
)
NAMED_MENTION_RE = re.compile(r"\[@(\d+)\]|\[CQ:at,[^\]]*?qq=(\d+)")

NamedUserResolver = Callable[[str], Iterable[int]]


@dataclass(frozen=True)
class Binding:
    status: str = NOT_APPLICABLE
    source: str = SOURCE_RULE
    confidence: float = 0.0
    target: str = ""
    target_id: int | None = None
    target_text: str = ""
    surface: str = ""
    reason: str = "none"
    value: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if not status:
            if self.target_id is not None or str(self.target_text or "").strip() or str(self.target or "").strip():
                status = RESOLVED
            else:
                status = NOT_APPLICABLE
            object.__setattr__(self, "status", status)
        if not str(self.source or ""):
            object.__setattr__(self, "source", SOURCE_RULE)
        finalize_result(
            self,
            has_value=bool(self.target_id is not None or self.target_text or self.target),
        )


@dataclass(frozen=True)
class AddresseeCandidate:
    key: str
    user_id: int | None
    label: str
    source: str
    evidence: str = ""


@dataclass(frozen=True)
class AddresseeJudgement:
    target_key: str = "NONE"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class DeixisCandidate:
    key: str
    speaker: str
    text: str
    source_reason: str
    message_id: str = ""
    user_id: int = 0
    is_question: bool = False


@dataclass(frozen=True)
class DiscourseAuditJudgement:
    conflict_field: str = "none"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class DiscourseState:
    speaker: str = ""
    speaker_id: int = 0
    current_text: str = ""
    reply_target: Binding = Binding()
    mentions: tuple[Binding, ...] = ()
    addressee: Binding = Binding()
    third_person_referent: Binding = Binding()
    deixis: Binding = Binding()
    conversation_topic: Binding = Binding()
    continuation: Binding = Binding()
    media_present: bool = False
    status: str = NOT_APPLICABLE
    source: str = SOURCE_RULE
    confidence: float = 0.0
    ambiguity: str = "NONE"
    state_audit: str = "PASS"
    reference: ReferenceResolution = ReferenceResolution()
    ellipsis: EllipsisResolution = EllipsisResolution()
    repair: RepairResolution = RepairResolution()
    ambiguity_resolution: AmbiguityResolution = AmbiguityResolution()
    invalidated_layers: tuple[str, ...] = ()
    recomputed_layers: tuple[str, ...] = ()

    @property
    def target_text(self) -> str:
        return self.deixis.target_text


def logical_turns(
    messages: Iterable[object],
    *,
    max_gap_seconds: float = MAX_LOGICAL_GAP_SECONDS,
) -> list[tuple[object, ...]]:
    rows = tuple(messages)
    if not rows:
        return []
    turns: list[list[object]] = []
    current: list[object] = []
    for message in rows:
        if not current:
            current = [message]
            continue
        prev = current[-1]
        same_speaker = int(getattr(message, "user_id", 0) or 0) == int(getattr(prev, "user_id", 0) or 0)
        prev_bot = bool(getattr(prev, "is_bot", False))
        this_bot = bool(getattr(message, "is_bot", False))
        gap = float(getattr(message, "created_at", 0) or 0) - float(getattr(prev, "created_at", 0) or 0)
        interrupted = (not same_speaker) or (this_bot != prev_bot)
        if interrupted or gap > float(max_gap_seconds):
            turns.append(current)
            current = [message]
            continue
        current.append(message)
    if current:
        turns.append(current)
    return [tuple(turn) for turn in turns]


def message_has_real_media(message: object) -> bool:
    segments = _segments_from_message(message)
    if _segments_have_real_media(segments):
        return True
    text = str(getattr(message, "text", "") or "")
    return bool(URL_RE.search(text))


def segments_have_real_media(segments: Iterable[object] | None, *, text: str = "") -> bool:
    if _segments_have_real_media(list(segments or ())):
        return True
    return bool(URL_RE.search(str(text or "")))


def _segments_from_message(message: object) -> list[object]:
    raw = getattr(message, "message_segments_json", "") or ""
    if raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = []
        return message_segments_from_payload(parsed)
    raw_message = getattr(message, "message", None)
    if raw_message is not None:
        return message_segments_from_payload(raw_message)
    return []


def _segments_have_real_media(segments: Iterable[object]) -> bool:
    for segment in segments:
        segment_type, data = segment_type_and_data(segment)
        if segment_type in {"mface", "face"}:
            continue
        if segment_type == "image" and is_marketface_segment(segment_type, data):
            continue
        if segment_type in REAL_MEDIA_TYPES:
            return True
        if segment_type in {"share", "json", "xml"}:
            blob = " ".join(str(data.get(key) or "") for key in ("url", "file", "data", "jumpUrl", "qqdocurl"))
            if URL_RE.search(blob):
                return True
        for key in ("url", "file", "path"):
            if URL_RE.search(str(data.get(key) or "")):
                return True
    return False


def _member_label(user_id: int | None, nickname: str = "") -> str:
    if user_id is None or int(user_id or 0) <= 0:
        return nickname or ""
    clean = (nickname or "").strip() or str(user_id)
    return f"{clean}[#{str(int(user_id))[-5:]}]"


def _nickname_for(user_id: int, messages: Iterable[object], fallback: str = "") -> str:
    for message in messages:
        if int(getattr(message, "user_id", 0) or 0) == int(user_id):
            nick = str(getattr(message, "nickname", "") or "").strip()
            if nick:
                return nick
    return fallback or str(user_id)


def build_addressee_candidates(
    *,
    current_user_id: int,
    current_label: str,
    current_text: str,
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
    mention_labels: Iterable[tuple[str, int]] = (),
    previous_addressee_id: int | None = None,
    previous_addressee_label: str = "",
) -> list[AddresseeCandidate]:
    rows: list[AddresseeCandidate] = []
    seen: set[str] = set()

    def add(candidate: AddresseeCandidate) -> None:
        if candidate.key in seen:
            return
        seen.add(candidate.key)
        rows.append(candidate)

    if reply is not None and reply.exists and reply.author_id:
        add(
            AddresseeCandidate(
                key="c_reply",
                user_id=int(reply.author_id),
                label=reply.author_label or _member_label(reply.author_id),
                source="reply",
                evidence=(reply.text or "")[:80],
            )
        )
    mention_map = {int(uid): label for label, uid in mention_labels if int(uid) > 0}
    at_ids = tuple(dict.fromkeys(int(uid) for uid in at_user_ids if int(uid) > 0 and int(uid) != int(current_user_id)))
    if at_ids:
        uid = at_ids[0]
        add(
            AddresseeCandidate(
                key="c_mention",
                user_id=uid,
                label=mention_map.get(uid, _member_label(uid)),
                source="at",
                evidence=str(current_text or "")[:80],
            )
        )
        for extra in at_ids[1:]:
            add(
                AddresseeCandidate(
                    key=f"c_at_{extra}",
                    user_id=extra,
                    label=mention_map.get(extra, _member_label(extra)),
                    source="at",
                    evidence=str(current_text or "")[:80],
                )
            )
    for label, uid in mention_labels:
        if int(uid) in at_ids or int(uid) == int(current_user_id):
            continue
        add(
            AddresseeCandidate(
                key=f"c_named_{uid}",
                user_id=int(uid),
                label=label or _member_label(uid),
                source="named",
                evidence=str(current_text or "")[:80],
            )
        )
    if previous_addressee_id and int(previous_addressee_id) not in {int(current_user_id), 0}:
        add(
            AddresseeCandidate(
                key="c_prev",
                user_id=int(previous_addressee_id),
                label=previous_addressee_label or _member_label(previous_addressee_id),
                source="previous_addressee",
            )
        )
    add(AddresseeCandidate(key="generic", user_id=None, label="群里泛说/没有特定对象", source="generic"))
    add(AddresseeCandidate(key="other", user_id=None, label="以上均不符合或无法判断", source="other"))
    return rows


def addressee_choice_criteria(candidates: Iterable[AddresseeCandidate]) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for row in candidates:
        if row.key == "generic":
            criteria[row.key] = "这句话是对群里泛说，没有特定听话人"
        elif row.key == "other":
            criteria[row.key] = "以上均不符合或无法判断"
        elif row.source == "reply":
            criteria[row.key] = f"在对 QQ 回复对象 {row.label} 说话"
        elif row.source == "at":
            criteria[row.key] = f"在对艾特/点名的 {row.label} 说话"
        elif row.source == "previous_addressee":
            criteria[row.key] = f"延续上一话轮对象 {row.label}"
        else:
            criteria[row.key] = f"在对 {row.label} 说话"
    if "other" not in criteria:
        criteria["other"] = "以上均不符合或无法判断"
    return criteria


def format_addressee_jev_state(
    *,
    current_text: str,
    current_label: str,
    candidates: Iterable[AddresseeCandidate],
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
) -> str:
    lines = [
        f"speaker.label={current_label}",
        f"current_text={(current_text or '')[:400]}",
        f"reply.exists={str(bool(reply is not None and reply.exists)).lower()}",
    ]
    if reply is not None and reply.exists:
        lines.append(f"reply.author={reply.author_label or reply.author_id}")
        lines.append(f"reply.text={(reply.text or '')[:160]}")
    at_ids = ",".join(str(uid) for uid in at_user_ids)
    lines.append(f"mentions.at_user_ids={at_ids or 'NONE'}")
    lines.append("candidates:")
    for row in candidates:
        lines.append(
            f"- {row.key} | user_id={row.user_id if row.user_id is not None else 'NONE'} "
            f"| source={row.source} | {row.label}"
        )
        if row.evidence:
            lines.append(f"  evidence={row.evidence[:80]}")
    lines.append("reply.author 和 at 同时存在时，不要按规则默认选其中一个。")
    return "\n".join(lines)


def parse_jev_addressee_answers(data: dict) -> AddresseeJudgement:
    answers = data.get("answers") if isinstance(data, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    target_raw = answers.get("addressee_target") if isinstance(answers.get("addressee_target"), dict) else {}
    if not isinstance(target_raw, dict):
        target_raw = {}
    key = str(target_raw.get("choice") or "").strip()
    if not key:
        return AddresseeJudgement(target_key="NONE", confidence=0.0, reason="error")
    confidence = choice_confidence(answers, "addressee_target")
    if confidence is None:
        return AddresseeJudgement(target_key=key, confidence=0.0, reason="error")
    return AddresseeJudgement(target_key=key, confidence=confidence, reason=f"jev_{key}")


def apply_jev_addressee_judgement(
    judgement: AddresseeJudgement | None,
    candidates: Iterable[AddresseeCandidate],
) -> Binding:
    rows = list(candidates)
    if judgement is None:
        return Binding(status=UNAVAILABLE, source=SOURCE_JEV, reason="jev_unavailable")
    if judgement.reason == "error":
        return Binding(status=ERROR, source=SOURCE_JEV, reason="error")
    key = str(judgement.target_key or "").strip()
    if key.upper() in {"", "NONE"}:
        return Binding(
            status=AMBIGUOUS,
            source=SOURCE_JEV,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_ambiguous",
        )
    if float(judgement.confidence or 0.0) < JEV_ADDRESSEE_CONFIDENCE_MIN:
        return Binding(
            status=AMBIGUOUS,
            source=SOURCE_JEV,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_low_confidence",
            target=key,
        )
    if key == "other":
        return Binding(
            status=AMBIGUOUS,
            source=SOURCE_JEV,
            confidence=float(judgement.confidence or 0.0),
            reason="jev_other",
            target="other",
            value="other",
        )
    if key == "generic":
        return Binding(
            status=RESOLVED,
            source=SOURCE_JEV,
            confidence=float(judgement.confidence or 0.0),
            target="generic",
            value="generic",
            reason=judgement.reason or "jev_generic",
            kind="GENERIC",
        )
    for row in rows:
        if row.key != key:
            continue
        return Binding(
            status=RESOLVED,
            source=SOURCE_JEV,
            confidence=float(judgement.confidence or 0.0),
            target=row.label,
            target_id=row.user_id,
            reason=judgement.reason or f"jev+{row.source}",
            value=str(row.user_id) if row.user_id is not None else row.key,
            kind="PERSON",
        )
    return Binding(
        status=AMBIGUOUS,
        source=SOURCE_JEV,
        confidence=float(judgement.confidence or 0.0),
        reason="jev_unknown_target",
        target=key,
    )


def should_ask_jev_addressee(
    text: str,
    *,
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
) -> bool:
    if reply is not None and reply.exists:
        return True
    if any(int(uid) > 0 for uid in at_user_ids):
        return True
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(re.search(r"(你|您|你们)", compact))


def build_deixis_candidates(
    recent_messages: Iterable[object],
    *,
    current_text: str,
    current_user_id: int = 0,
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
    media_present: bool = False,
    current_has_media: bool = False,
    max_candidates: int = MAX_DEIXIS_CANDIDATES,
) -> list[DeixisCandidate]:
    messages = tuple(recent_messages)
    rows: list[DeixisCandidate] = []
    seen: set[str] = set()
    current = str(current_text or "").strip()

    def add(candidate: DeixisCandidate) -> bool:
        if len(rows) >= max_candidates - 1:
            return False
        if candidate.key in seen:
            return False
        text = str(candidate.text or "").strip()
        if not text or text == current:
            return False
        seen.add(candidate.key)
        rows.append(candidate)
        return True

    def from_message(message: object, reason: str) -> DeixisCandidate:
        index = -1
        for idx, item in enumerate(messages):
            if item is message:
                index = idx
                break
        key = source_key_for_message(message, index=index if index >= 0 else len(rows))
        speaker = str(getattr(message, "nickname", "") or getattr(message, "user_id", "") or "")
        if bool(getattr(message, "is_bot", False)):
            speaker = "风雪"
        return DeixisCandidate(
            key=key,
            speaker=speaker,
            text=semantic_message_text(str(getattr(message, "text", "") or "")),
            source_reason=reason,
            message_id=str(getattr(message, "source_message_id", "") or getattr(message, "id", "") or ""),
            user_id=int(getattr(message, "user_id", 0) or 0),
        )

    speaker_id = int(current_user_id or 0)
    if speaker_id > 0:
        turns = logical_turns(messages)
        burst: tuple[object, ...] = ()
        for turn in reversed(turns):
            if turn and int(getattr(turn[0], "user_id", 0) or 0) == speaker_id and not bool(getattr(turn[0], "is_bot", False)):
                burst = turn
                break
        for message in reversed(burst[-SAME_SPEAKER_BURST:]):
            add(from_message(message, "same_speaker"))

    if reply is not None and reply.exists:
        reply_id = str(reply.message_id or "").strip()
        reply_text = str(reply.text or "").strip()
        matched = None
        for message in messages:
            source_id = str(getattr(message, "source_message_id", "") or "").strip()
            if reply_id and source_id == reply_id:
                matched = message
                break
            if reply_text and str(getattr(message, "text", "") or "").strip() == reply_text:
                matched = message
                break
        if matched is not None:
            add(from_message(matched, "reply"))
        elif reply_text:
            add(
                DeixisCandidate(
                    key=f"s{reply_id}" if reply_id else "sreply",
                    speaker=reply.author_label or str(reply.author_id or "被回复消息"),
                    text=reply_text,
                    source_reason="reply",
                    message_id=reply_id,
                    user_id=int(reply.author_id or 0),
                )
            )

    mention_ids = {int(uid) for uid in at_user_ids if int(uid) > 0}
    if mention_ids:
        for message in reversed(messages):
            uid = int(getattr(message, "user_id", 0) or 0)
            if uid in mention_ids and add(from_message(message, "mention_object")):
                break

    for message in reversed(messages):
        text = str(getattr(message, "text", "") or "").strip()
        if text and text != current and add(from_message(message, "topic")):
            break

    if media_present or current_has_media or any(message_has_real_media(msg) for msg in messages[-3:]):
        media_msg = None
        for message in reversed(messages):
            if message_has_real_media(message):
                media_msg = message
                break
        if media_msg is not None:
            media_cand = from_message(media_msg, "media")
            if media_cand.key in seen:
                media_cand = replace(media_cand, key=f"media:{media_cand.key}")
            add(media_cand)
        elif current_has_media or media_present:
            add(
                DeixisCandidate(
                    key="media",
                    speaker="当前消息",
                    text="[真实图片/文件/链接]",
                    source_reason="media",
                )
            )

    extra = build_ellipsis_candidates(
        messages,
        current_text=current_text,
        current_user_id=current_user_id,
        reply=reply,
        max_candidates=max_candidates,
    )
    for source in extra:
        add(
            DeixisCandidate(
                key=source.key,
                speaker=source.speaker,
                text=source.text,
                source_reason=source.source_reason,
                message_id=source.message_id,
                user_id=source.user_id,
                is_question=source.is_question,
            )
        )

    rows.append(
        DeixisCandidate(
            key="other",
            speaker="",
            text="以上均不符合或无法判断",
            source_reason="other",
        )
    )
    return rows[:max_candidates]


def deixis_as_ellipsis_sources(candidates: Iterable[DeixisCandidate]) -> list[EllipsisSource]:
    rows: list[EllipsisSource] = []
    for item in candidates:
        if item.key == "other":
            continue
        rows.append(
            EllipsisSource(
                key=item.key,
                speaker=item.speaker,
                text=item.text,
                source_reason=item.source_reason,
                message_id=item.message_id,
                user_id=item.user_id,
                is_question=item.is_question,
            )
        )
    return rows


def _bind_auto_inherit(
    judgement: EllipsisJudgement | None,
    sources: Iterable[DeixisCandidate | EllipsisSource],
) -> EllipsisJudgement | None:
    if judgement is None:
        return None
    inherit = str(judgement.inherit_from or "").strip()
    if inherit.upper() not in {"AUTO", ""}:
        return judgement
    if judgement.kind in {"", "NONE"}:
        return judgement
    preferred = [row for row in sources if getattr(row, "source_reason", "") == "same_speaker" and row.key != "other"]
    fallback = [row for row in sources if row.key != "other"]
    chosen = (preferred or fallback or [None])[0]
    if chosen is None:
        return judgement
    return EllipsisJudgement(
        kind=judgement.kind,
        inherit_from=chosen.key,
        confidence=judgement.confidence,
        reason=judgement.reason,
    )


def classify_ambiguity_effect(
    *,
    candidates: Iterable[str] = (),
    kind: str = "NONE",
    resolved_deixis_text: str = "",
    jev_effect: str = "",
) -> str:
    labels = tuple(str(item).strip() for item in candidates if str(item).strip())
    kind_u = str(kind or "NONE").upper()
    effect = str(jev_effect or "").upper()
    if kind_u in {"PERSON"} and effect == "BLOCKING":
        return "BLOCKING"
    if kind_u in {"PERSON"} and len({item.casefold() for item in labels}) >= 2:
        return "BLOCKING"
    if _similar_item_texts(labels) or (
        resolved_deixis_text and _similar_item_texts((resolved_deixis_text, *labels))
    ):
        return "NON_BLOCKING"
    if effect in {"BLOCKING", "NON_BLOCKING", "NONE"}:
        return effect
    if kind_u in {"", "NONE"} or not labels:
        return "NONE"
    if kind_u in {"PERSON", "INTENT"}:
        return "BLOCKING"
    return "NON_BLOCKING"


def _similar_item_texts(candidates: Iterable[str]) -> bool:
    texts = [re.sub(r"\s+", "", str(item or "")) for item in candidates if str(item or "").strip()]
    if len(texts) < 2:
        return False
    money = all(re.search(r"(折|块|充|钱|折扣|原价)", item) for item in texts)
    if money:
        return True
    digits = [frozenset(re.findall(r"\d+", item)) for item in texts]
    return bool(digits and digits[0] and all(item & digits[0] for item in digits[1:]))


def targeted_requery_fields(judgement: DiscourseAuditJudgement | None) -> tuple[str, ...]:
    if judgement is None:
        return ()
    field = str(judgement.conflict_field or "").strip().lower()
    if field in {"", "none"}:
        return ()
    if float(judgement.confidence or 0.0) < JEV_AUDIT_CONFIDENCE_MIN:
        return ()
    aliases = {
        "addressee": "addressee",
        "deixis": "deixis",
        "ellipsis": "deixis",
        "referent": "referent",
        "topic": "topic",
        "third_person_referent": "referent",
    }
    mapped = aliases.get(field)
    if mapped:
        return (mapped,)
    if field == "other":
        return ()
    return ()


def apply_jev_discourse_audit(judgement: DiscourseAuditJudgement | None) -> str:
    if judgement is None:
        return UNAVAILABLE
    if judgement.reason == "error":
        return ERROR
    fields = targeted_requery_fields(judgement)
    if fields:
        return "REQUERY"
    return "PASS"


def parse_jev_discourse_audit_answers(data: dict) -> DiscourseAuditJudgement:
    answers = data.get("answers") if isinstance(data, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    field_raw = answers.get("conflict_field") if isinstance(answers.get("conflict_field"), dict) else {}
    if not isinstance(field_raw, dict):
        field_raw = {}
    field = str(field_raw.get("choice") or "none").strip().lower() or "none"
    confidence = choice_confidence(answers, "conflict_field")
    if confidence is None:
        return DiscourseAuditJudgement(conflict_field=field, confidence=0.0, reason="error")
    return DiscourseAuditJudgement(
        conflict_field=field,
        confidence=confidence,
        reason=f"jev_{field}",
    )


def _binding_from_ellipsis(ellipsis: EllipsisResolution | Binding | None) -> Binding:
    if isinstance(ellipsis, Binding):
        return ellipsis
    if ellipsis is None:
        return Binding(status=NOT_APPLICABLE)
    status = ellipsis.status or NOT_APPLICABLE
    if status == NOT_APPLICABLE and ellipsis.kind in {"", "NONE"}:
        status = NOT_APPLICABLE
    return Binding(
        status=status,
        source=ellipsis.source or SOURCE_JEV,
        confidence=float(ellipsis.confidence or 0.0),
        target=ellipsis.source_text,
        target_text=ellipsis.source_text,
        surface="这个",
        reason=ellipsis.reason,
        value=ellipsis.value or ellipsis.source_key,
        kind=ellipsis.kind,
    )


def _binding_from_reference(reference: ReferenceResolution | None) -> Binding:
    if reference is None:
        return Binding(status=NOT_APPLICABLE)
    target_id = reference.user_ids[0] if reference.user_ids else None
    return Binding(
        status=reference.status or NOT_APPLICABLE,
        source=reference.source or SOURCE_RULE,
        confidence=float(reference.confidence or 0.0),
        target=",".join(str(uid) for uid in reference.user_ids),
        target_id=target_id,
        reason=reference.reason,
        value=reference.value,
        kind=reference.kind,
    )


def assemble_discourse_state(
    *,
    speaker_id: int,
    speaker_label: str,
    current_text: str,
    addressee: Binding | None = None,
    deixis: EllipsisResolution | Binding | None = None,
    third_person_referent: Binding | ReferenceResolution | None = None,
    reply_target: Binding | None = None,
    mentions: Iterable[Binding] = (),
    media_present: bool = False,
    ambiguity_effect: str = "NONE",
    state_audit: str = "PASS",
    reference: ReferenceResolution | None = None,
    ellipsis: EllipsisResolution | None = None,
    repair: RepairResolution | None = None,
    ambiguity_resolution: AmbiguityResolution | None = None,
    conversation_topic: Binding | None = None,
    continuation: Binding | None = None,
    invalidated_layers: Iterable[str] = (),
    recomputed_layers: Iterable[str] = (),
) -> DiscourseState:
    ellipsis_res = ellipsis
    if isinstance(deixis, EllipsisResolution):
        ellipsis_res = deixis
    reference_res = reference if isinstance(reference, ReferenceResolution) else ReferenceResolution()
    if isinstance(third_person_referent, ReferenceResolution):
        reference_res = third_person_referent
        third_binding = _binding_from_reference(third_person_referent)
    elif isinstance(third_person_referent, Binding):
        third_binding = third_person_referent
    else:
        third_binding = _binding_from_reference(reference_res)
    deixis_binding = _binding_from_ellipsis(deixis if deixis is not None else ellipsis_res)
    addressee_binding = addressee or Binding(status=NOT_APPLICABLE)
    overall = RESOLVED
    for item in (addressee_binding, deixis_binding, third_binding):
        if item.status == UNAVAILABLE:
            overall = UNAVAILABLE
            break
        if item.status == ERROR:
            overall = ERROR
            break
        if item.status == AMBIGUOUS and overall == RESOLVED:
            overall = AMBIGUOUS
    topic_text = deixis_binding.target_text or ""
    topic = conversation_topic or (
        Binding(status=RESOLVED, target=topic_text, target_text=topic_text, source=SOURCE_JEV)
        if topic_text
        else Binding(status=NOT_APPLICABLE)
    )
    cont = continuation or Binding(
        status=RESOLVED if deixis_binding.kind in {"CONTINUATION", "ITEM_DEIXIS", "SAME_PREDICATE"} else NOT_APPLICABLE,
        target=deixis_binding.kind,
        kind=deixis_binding.kind,
        source=deixis_binding.source,
    )
    return DiscourseState(
        speaker=speaker_label,
        speaker_id=int(speaker_id or 0),
        current_text=current_text,
        reply_target=reply_target or Binding(status=NOT_APPLICABLE),
        mentions=tuple(mentions),
        addressee=addressee_binding,
        third_person_referent=third_binding,
        deixis=deixis_binding,
        conversation_topic=topic,
        continuation=cont,
        media_present=bool(media_present),
        status=overall,
        source=SOURCE_JEV,
        confidence=max(addressee_binding.confidence, deixis_binding.confidence, third_binding.confidence),
        ambiguity=str(ambiguity_effect or "NONE"),
        state_audit=str(state_audit or "PASS"),
        reference=reference_res,
        ellipsis=ellipsis_res or EllipsisResolution(),
        repair=repair or RepairResolution(),
        ambiguity_resolution=ambiguity_resolution or AmbiguityResolution(),
        invalidated_layers=tuple(invalidated_layers),
        recomputed_layers=tuple(recomputed_layers),
    )


def format_discourse_prompt_block(state: DiscourseState) -> str:
    lines = [
        "[discourse]",
        f"speaker={state.speaker}",
        f"addressee={state.addressee.target or state.addressee.value or state.addressee.status}",
        f"addressee.status={state.addressee.status}",
        f"deixis.status={state.deixis.status}",
    ]
    if state.deixis.target_text:
        lines.append(f"deixis={state.deixis.target_text[:80]}")
    lines.append(f"media_present={str(state.media_present).lower()}")
    lines.append(f"ambiguity={state.ambiguity}")
    lines.append(f"state_audit={state.state_audit}")
    if state.addressee.status == RESOLVED and state.addressee.target and state.addressee.target != "generic":
        lines.append(f"- 当前这句话是在对 {state.addressee.target} 说，不是在问有没有图。")
    if state.deixis.status == RESOLVED and state.deixis.target_text:
        lines.append(f"- 当前「这个/那个」已接到：{state.deixis.target_text[:80]}。按这个对象接。")
    if not state.media_present:
        lines.append("- media_present=false：不要问缺图或链接，不要说把图发一下/截图呢/链接呢/图片没看到。")
    if state.addressee.status == UNAVAILABLE or state.deixis.status == UNAVAILABLE:
        lines.append("- 有检查结果不可用，不要把 UNAVAILABLE 当成没有指代。")
    if state.ambiguity == "NON_BLOCKING":
        lines.append("- 候选对象语义一致，不要追问。")
    if state.ambiguity == "BLOCKING":
        lines.append("- 歧义会改变回答对象或事实，优先 clarify。")
    return "\n".join(lines)


def discourse_decision_trace(state: DiscourseState, *, final_action: str = "answer") -> dict[str, object]:
    addressee_target = state.addressee.target or (
        str(state.addressee.target_id) if state.addressee.target_id is not None else ""
    )
    return {
        "speaker": state.speaker,
        "addressee": {
            "target": addressee_target,
            "status": state.addressee.status,
            "source": state.addressee.source if state.addressee.source != SOURCE_RULE else (
                state.addressee.reason or state.addressee.source
            ),
        },
        "deixis": {
            "surface": state.deixis.surface or "这个",
            "target_text": state.deixis.target_text,
            "status": state.deixis.status,
        },
        "media_present": bool(state.media_present),
        "ambiguity": state.ambiguity,
        "state_audit": state.state_audit,
        "final_action": final_action,
    }


def draft_violates_media_gate(draft: str, state: DiscourseState) -> bool:
    if state.media_present:
        return False
    return bool(MEDIA_ASK_RE.search(str(draft or "")))


def should_audit_discourse_state(state: DiscourseState) -> bool:
    """Audit only when there are at least two facts that can contradict."""
    has_competing_addressee_evidence = (
        state.reply_target.status == RESOLVED and bool(state.mentions)
    )
    return has_competing_addressee_evidence or state.repair.status == RESOLVED


def format_discourse_audit_state(state: DiscourseState) -> dict[str, object]:
    def binding_payload(binding: Binding) -> dict[str, object]:
        return {
            "status": binding.status,
            "target": binding.target,
            "target_id": binding.target_id,
            "target_text": binding.target_text[:160],
            "kind": binding.kind,
            "reason": binding.reason,
            "confidence": binding.confidence,
        }

    return {
        "message": {
            "speaker": state.speaker,
            "speaker_id": state.speaker_id,
            "current_text": state.current_text[:240],
            "media_present": state.media_present,
        },
        "evidence": {
            "reply_target": binding_payload(state.reply_target),
            "mentions": [binding_payload(item) for item in state.mentions],
        },
        "bindings": {
            "addressee": binding_payload(state.addressee),
            "deixis": binding_payload(state.deixis),
            "third_person": binding_payload(state.third_person_referent),
            "topic": binding_payload(state.conversation_topic),
        },
        "repair": {
            "status": state.repair.status,
            "kind": state.repair.kind,
            "target_key": state.repair.target_key,
            "replacement_user_ids": list(state.repair.replacement_user_ids),
            "replacement_item": state.repair.replacement_item,
        },
    }


async def _maybe_jev_call(jev, method: str, **kwargs):
    if jev is None:
        return None
    func = getattr(jev, method, None)
    if func is None:
        return None
    return await func(**kwargs)


def _mention_labels_from(
    *,
    at_user_ids: Iterable[int],
    current_text: str,
    recent_messages: Iterable[object],
    named_resolver: NamedUserResolver | None,
) -> list[tuple[str, int]]:
    labels: list[tuple[str, int]] = []
    seen: set[int] = set()
    messages = tuple(recent_messages)
    for uid in at_user_ids:
        uid_i = int(uid)
        if uid_i in seen:
            continue
        seen.add(uid_i)
        labels.append((_member_label(uid_i, _nickname_for(uid_i, messages)), uid_i))
    if named_resolver is not None:
        try:
            named_ids = tuple(int(uid) for uid in named_resolver(current_text) if int(uid) > 0)
        except Exception:
            named_ids = ()
        for uid_i in named_ids:
            if uid_i in seen:
                continue
            seen.add(uid_i)
            labels.append((_member_label(uid_i, _nickname_for(uid_i, messages)), uid_i))
    for match in NAMED_MENTION_RE.finditer(str(current_text or "")):
        raw = match.group(1) or match.group(2)
        try:
            uid_i = int(raw)
        except (TypeError, ValueError):
            continue
        if uid_i in seen:
            continue
        seen.add(uid_i)
        labels.append((_member_label(uid_i, _nickname_for(uid_i, messages)), uid_i))
    return labels


async def resolve_group_discourse(
    *,
    current_text: str,
    current_user_id: int,
    current_nickname: str,
    self_id: int,
    recent_messages: Iterable[object],
    reply: ReplyHint | None = None,
    at_user_ids: Iterable[int] = (),
    named_resolver: NamedUserResolver | None = None,
    jev=None,
    current_has_media: bool = False,
    reply_has_media: bool = False,
    relation_user_ids: Iterable[int] = (),
    previous_addressee_id: int | None = None,
    previous_addressee_label: str = "",
) -> DiscourseState:
    messages = tuple(recent_messages)
    speaker_label = _member_label(current_user_id, current_nickname)
    at_ids = tuple(dict.fromkeys(int(uid) for uid in at_user_ids if int(uid) > 0))
    media_present = bool(current_has_media or reply_has_media)
    mention_labels = _mention_labels_from(
        at_user_ids=at_ids,
        current_text=current_text,
        recent_messages=messages,
        named_resolver=named_resolver,
    )
    reply_binding = Binding(status=NOT_APPLICABLE)
    if reply is not None and reply.exists and reply.author_id:
        reply_binding = Binding(
            status=RESOLVED,
            source=SOURCE_RULE,
            target=reply.author_label or _member_label(reply.author_id),
            target_id=int(reply.author_id),
            target_text=reply.text,
            reason="reply",
            value=str(reply.author_id),
        )
    mention_bindings = tuple(
        Binding(
            status=RESOLVED,
            source=SOURCE_RULE,
            target=label,
            target_id=uid,
            reason="at" if uid in at_ids else "named",
            value=str(uid),
        )
        for label, uid in mention_labels
    )

    addressee_rows = build_addressee_candidates(
        current_user_id=current_user_id,
        current_label=speaker_label,
        current_text=current_text,
        reply=reply,
        at_user_ids=at_ids,
        mention_labels=mention_labels,
        previous_addressee_id=previous_addressee_id,
        previous_addressee_label=previous_addressee_label,
    )
    explicit_addressee_rows = [
        row for row in addressee_rows if row.key not in {"generic", "other"}
    ]
    direct_addressee = None
    direct_reply = next((row for row in addressee_rows if row.source == "reply"), None)
    if (
        direct_reply is not None
        and direct_reply.user_id is not None
        and int(direct_reply.user_id) != int(current_user_id)
        and not at_ids
    ):
        direct_addressee = Binding(
            status=RESOLVED,
            source=SOURCE_RULE,
            confidence=1.0,
            target=direct_reply.label,
            target_id=direct_reply.user_id,
            reason="single_reply",
            value=str(direct_reply.user_id),
            kind="PERSON",
        )
    elif (
        (reply is None or not reply.exists)
        and len(explicit_addressee_rows) == 1
        and explicit_addressee_rows[0].source == "at"
    ):
        row = explicit_addressee_rows[0]
        direct_addressee = Binding(
            status=RESOLVED,
            source=SOURCE_RULE,
            confidence=1.0,
            target=row.label,
            target_id=row.user_id,
            reason="single_at",
            value=str(row.user_id),
            kind="PERSON",
        )
    need_addressee = direct_addressee is None and should_ask_jev_addressee(
        current_text,
        reply=reply,
        at_user_ids=at_ids,
    )
    reference = resolve_context_reference(
        current_text,
        messages,
        current_user_id=current_user_id,
        resolve_named_users=named_resolver,
    )
    rule_reference = reference
    need_referent = should_ask_jev_referent(current_text, rule_reference)
    referent_candidates = []
    if need_referent:
        referent_candidates = build_referent_candidates(
            messages,
            current_user_id=current_user_id,
            current_text=current_text,
            self_id=int(self_id),
            reply=reply,
            at_user_ids=at_ids,
            resolve_named_users=named_resolver,
            relation_user_ids=relation_user_ids,
            rule_guess=rule_reference,
        )

    need_ellipsis = should_ask_jev_ellipsis(current_text)
    deixis_rows: list[DeixisCandidate] = []
    sources: list[EllipsisSource] = []
    if need_ellipsis:
        deixis_rows = build_deixis_candidates(
            messages,
            current_text=current_text,
            current_user_id=current_user_id,
            reply=reply,
            at_user_ids=at_ids,
            media_present=media_present,
            current_has_media=current_has_media,
        )
        sources = deixis_as_ellipsis_sources(deixis_rows)

    judged_addressee = None
    judged_referent = None
    judged_ellipsis = None
    first_pass = getattr(jev, "resolve_discourse_first_pass", None) if jev is not None else None
    # reply + @ is a competing addressee decision. Keep its state compact and
    # run it alongside the referent/ellipsis batch instead of contaminating one
    # shared request. Deterministic single-@ cases were already bound above.
    isolate_addressee = bool(
        need_addressee
        and reply is not None
        and reply.exists
        and at_ids
    )
    pending: list[tuple[str, object]] = []
    if callable(first_pass) and (
        (need_addressee and not isolate_addressee) or need_referent or need_ellipsis
    ):
        pending.append((
            "batch",
            first_pass(
                current_text=current_text,
                current_label=speaker_label,
                addressee_candidates=addressee_rows if need_addressee and not isolate_addressee else None,
                referent_candidates=referent_candidates if need_referent else None,
                ellipsis_sources=sources if need_ellipsis else None,
                reply=reply,
                at_user_ids=at_ids,
                rule_reference=rule_reference,
            ),
        ))
    if need_addressee and (isolate_addressee or not callable(first_pass)):
        pending.append((
            "addressee",
            _maybe_jev_call(
                jev,
                "resolve_addressee",
                current_text=current_text,
                current_label=speaker_label,
                candidates=addressee_rows,
                reply=reply,
                at_user_ids=at_ids,
            ),
        ))
    if not callable(first_pass):
        if need_referent and referent_candidates:
            pending.append((
                "referent",
                _maybe_jev_call(
                    jev,
                    "resolve_referent",
                    current_text=current_text,
                    current_label=speaker_label,
                    candidates=referent_candidates,
                    reply=reply,
                    rule_guess=rule_reference,
                ),
            ))
        if need_ellipsis and sources:
            pending.append((
                "ellipsis",
                _maybe_jev_call(
                    jev,
                    "resolve_ellipsis",
                    current_text=current_text,
                    current_label=speaker_label,
                    sources=sources,
                    reply=reply,
                    reference=rule_reference,
                ),
            ))
    if pending:
        results = await asyncio.gather(*(call for _, call in pending))
        for (kind, _), result in zip(pending, results):
            if kind == "batch" and isinstance(result, dict):
                judged_addressee = result.get("addressee")
                judged_referent = result.get("referent")
                judged_ellipsis = result.get("ellipsis")
            elif kind == "addressee":
                judged_addressee = result
            elif kind == "referent":
                judged_referent = result
            elif kind == "ellipsis":
                judged_ellipsis = result

    addressee = direct_addressee or Binding(status=NOT_APPLICABLE, source=SOURCE_RULE)
    if need_addressee:
        addressee = apply_jev_addressee_judgement(judged_addressee, addressee_rows)
    if need_referent:
        reference = apply_jev_referent_judgement(
            judged_referent,
            referent_candidates,
            current_text=current_text,
            fallback=rule_reference,
        )

    ellipsis = EllipsisResolution()
    if need_ellipsis:
        judged_ellipsis = _bind_auto_inherit(judged_ellipsis, deixis_rows)
        if judged_ellipsis is not None and str(judged_ellipsis.inherit_from or "").lower() == "other":
            ellipsis = EllipsisResolution(
                kind=judged_ellipsis.kind,
                unresolved=True,
                confidence=float(judged_ellipsis.confidence or 0.0),
                reason="jev_other",
                status=AMBIGUOUS,
                source=SOURCE_JEV,
            )
        else:
            ellipsis = apply_jev_ellipsis_judgement(
                judged_ellipsis,
                sources,
                current_text=current_text,
            )

    repair = RepairResolution()
    invalidated: list[str] = []
    recomputed: list[str] = []
    named_in_text = tuple(named_resolver(current_text) or ()) if named_resolver is not None else ()
    if should_ask_jev_repair(current_text, reference=reference, named_in_text=named_in_text):
        recent_bot_text = next(
            (
                str(getattr(msg, "text", "") or "")
                for msg in reversed(messages)
                if bool(getattr(msg, "is_bot", False)) and str(getattr(msg, "text", "") or "").strip()
            ),
            "",
        )
        repair_targets = build_repair_targets(
            recent_messages=messages,
            reply=reply,
            reference=reference,
            ellipsis=ellipsis,
            recent_bot_text=recent_bot_text,
            current_text=current_text,
        )
        judged_repair = None
        if repair_targets:
            judged_repair = await _maybe_jev_call(
                jev,
                "resolve_repair",
                current_text=current_text,
                current_label=speaker_label,
                targets=repair_targets,
                reply=reply,
                reference=reference,
                ellipsis=ellipsis,
            )
        repair = apply_jev_repair_judgement(
            judged_repair,
            repair_targets,
            current_text=current_text,
            reference=reference,
            recent_messages=messages,
            resolve_named_users=named_resolver,
        )
        dummy_memory = MemoryEffectResolution()
        dummy_ambiguity = AmbiguityResolution()
        reference, ellipsis, _, _, invalidated = apply_repair_invalidation(
            repair=repair,
            reference=reference,
            ellipsis=ellipsis,
            memory_effect=dummy_memory,
            ambiguity=dummy_ambiguity,
        )
        if "ellipsis" in invalidated and should_ask_jev_ellipsis(current_text):
            deixis_rows = build_deixis_candidates(
                messages,
                current_text=current_text,
                current_user_id=current_user_id,
                reply=reply,
                at_user_ids=at_ids,
                media_present=media_present,
                current_has_media=current_has_media,
            )
            sources = deixis_as_ellipsis_sources(deixis_rows)
            judged_ellipsis = None
            if sources:
                judged_ellipsis = await _maybe_jev_call(
                    jev,
                    "resolve_ellipsis",
                    current_text=current_text,
                    current_label=speaker_label,
                    sources=sources,
                    reply=reply,
                    reference=reference,
                )
            judged_ellipsis = _bind_auto_inherit(judged_ellipsis, deixis_rows)
            ellipsis = apply_jev_ellipsis_judgement(
                judged_ellipsis,
                sources,
                current_text=current_text,
            )
            recomputed.append("ellipsis")
        if "referent" in invalidated:
            recomputed.append("referent")

    ambiguity = AmbiguityResolution()
    skip_ambiguity = ellipsis.status == RESOLVED and ellipsis.kind in {"ITEM_DEIXIS", "SAME_PREDICATE", "CONTINUATION"}
    if (not skip_ambiguity) and should_ask_jev_ambiguity(
        reference=reference,
        ellipsis=ellipsis,
        repair=repair,
    ):
        candidate_labels = []
        if ellipsis.source_text:
            candidate_labels.append(ellipsis.source_text[:80])
        if repair.replacement_item:
            candidate_labels.append(repair.replacement_item)
        judged_ambiguity = await _maybe_jev_call(
            jev,
            "resolve_ambiguity",
            current_text=current_text,
            reference=reference,
            ellipsis=ellipsis,
            repair=repair,
            candidate_labels=candidate_labels,
        )
        ambiguity = apply_jev_ambiguity_judgement(
            judged_ambiguity,
            candidate_labels=candidate_labels,
        )

    effect = classify_ambiguity_effect(
        candidates=ambiguity.candidates,
        kind=ambiguity.kind,
        resolved_deixis_text=ellipsis.source_text,
        jev_effect="NONE" if ambiguity.kind in {"", "NONE"} else (
            "NON_BLOCKING" if skip_ambiguity else "BLOCKING" if ambiguity.kind == "PERSON" else "NON_BLOCKING"
        ),
    )
    if skip_ambiguity:
        effect = "NON_BLOCKING" if ellipsis.source_text else "NONE"
        if ambiguity.kind not in {"", "NONE"} and ambiguity.status == AMBIGUOUS:
            ambiguity = AmbiguityResolution(
                kind="NONE",
                status=NOT_APPLICABLE,
                source=ambiguity.source,
                reason="non_blocking_resolved_deixis",
            )

    state = assemble_discourse_state(
        speaker_id=current_user_id,
        speaker_label=speaker_label,
        current_text=current_text,
        addressee=addressee,
        deixis=ellipsis,
        third_person_referent=reference,
        reply_target=reply_binding,
        mentions=mention_bindings,
        media_present=media_present,
        ambiguity_effect=effect,
        state_audit="PASS",
        reference=reference,
        ellipsis=ellipsis,
        repair=repair,
        ambiguity_resolution=ambiguity,
        invalidated_layers=invalidated,
        recomputed_layers=recomputed,
    )
    judged_audit = None
    if should_audit_discourse_state(state):
        judged_audit = await _maybe_jev_call(
            jev,
            "audit_discourse_state",
            state=state,
        )
        audit_status = apply_jev_discourse_audit(judged_audit)
        fields = targeted_requery_fields(judged_audit)
    else:
        audit_status = "PASS"
        fields = ()
    if "addressee" in fields and addressee_rows:
        judged_addressee = await _maybe_jev_call(
            jev,
            "resolve_addressee",
            current_text=current_text,
            current_label=speaker_label,
            candidates=addressee_rows,
            reply=reply,
            at_user_ids=at_ids,
        )
        addressee = apply_jev_addressee_judgement(judged_addressee, addressee_rows)
        recomputed = list(recomputed) + ["addressee"]
    if "deixis" in fields and deixis_rows:
        sources = deixis_as_ellipsis_sources(deixis_rows)
        judged_ellipsis = await _maybe_jev_call(
            jev,
            "resolve_ellipsis",
            current_text=current_text,
            current_label=speaker_label,
            sources=sources,
            reply=reply,
            reference=reference,
        )
        judged_ellipsis = _bind_auto_inherit(judged_ellipsis, deixis_rows)
        ellipsis = apply_jev_ellipsis_judgement(judged_ellipsis, sources, current_text=current_text)
        recomputed = list(recomputed) + ["deixis"]
    if "referent" in fields and should_ask_jev_referent(current_text, reference):
        referent_candidates = build_referent_candidates(
            messages,
            current_user_id=current_user_id,
            current_text=current_text,
            self_id=int(self_id),
            reply=reply,
            at_user_ids=at_ids,
            resolve_named_users=named_resolver,
            relation_user_ids=relation_user_ids,
            rule_guess=reference,
        )
        judged_referent = await _maybe_jev_call(
            jev,
            "resolve_referent",
            current_text=current_text,
            current_label=speaker_label,
            candidates=referent_candidates,
            reply=reply,
            rule_guess=reference,
        )
        reference = apply_jev_referent_judgement(
            judged_referent,
            referent_candidates,
            current_text=current_text,
            fallback=reference,
        )
        recomputed = list(recomputed) + ["referent"]
    return assemble_discourse_state(
        speaker_id=current_user_id,
        speaker_label=speaker_label,
        current_text=current_text,
        addressee=addressee,
        deixis=ellipsis,
        third_person_referent=reference,
        reply_target=reply_binding,
        mentions=mention_bindings,
        media_present=media_present,
        ambiguity_effect=effect,
        state_audit=audit_status,
        reference=reference,
        ellipsis=ellipsis,
        repair=repair,
        ambiguity_resolution=ambiguity,
        invalidated_layers=invalidated,
        recomputed_layers=recomputed,
    )
