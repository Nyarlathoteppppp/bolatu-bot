from __future__ import annotations

import re
from dataclasses import dataclass

from .discourse_effects import (
    AmbiguityResolution,
    RepairResolution,
    format_ambiguity_prompt_block,
    format_repair_prompt_block,
)
from .discourse_state import DiscourseState, format_discourse_prompt_block
from .ellipsis_resolver import EllipsisResolution, format_ellipsis_prompt_block
from .member_context import member_label as _member_label
from .memory import ChatMessage
from .reference_resolver import (
    ReferenceResolution,
    format_referent_prompt_block,
    has_strong_person_reference,
)
from .resolver_result import AMBIGUOUS, ERROR, RESOLVED, UNAVAILABLE


def _short_notice_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


BOT_SELF_NAME_ALIASES = ("张风雪", "风雪")


@dataclass(frozen=True)
class MessageRelationFacts:
    current_label: str
    target_scope: str
    target_note: str
    mentioned_bot: bool
    replied_to_bot: bool
    addressed_bot: bool
    followup_addressed: bool
    self_name_mentioned: bool
    reply_speaker_label: str = ""
    reply_target_label: str = ""
    reply_target_is_bot: bool = False
    ambiguous_reference: bool = False
    reference_user_ids: tuple[int, ...] = ()
    reference_reason: str = ""
    reference_confidence: float = 0.0


REPLY_RELATION_RE = re.compile(
    r"(?P<speaker>[^\n【]{1,80}?\[#\d{5}\])回复(?P<target>[^\n【]{1,80}?\[#\d{5}\])消息【"
)


def _format_speaker_reference_context(
    *,
    current_user_id: int,
    current_nickname: str,
    current_text: str,
    recent_messages: list[ChatMessage],
    reference_resolution: ReferenceResolution,
    mentioned: bool,
    replied_to_bot: bool,
    addressed_bot: bool,
    followup_addressed: bool = False,
    followup_soft: bool = False,
    self_id: int,
    relation_facts: MessageRelationFacts | None = None,
    ellipsis_resolution: EllipsisResolution | None = None,
    repair_resolution: RepairResolution | None = None,
    ambiguity_resolution: AmbiguityResolution | None = None,
    discourse_state: DiscourseState | None = None,
) -> str:
    current_label = _member_label(current_user_id, current_nickname)
    facts = relation_facts or _message_relation_facts(
        current_user_id=current_user_id,
        current_nickname=current_nickname,
        current_text=current_text,
        reference_resolution=reference_resolution,
        mentioned=mentioned,
        replied_to_bot=replied_to_bot,
        addressed_bot=addressed_bot,
        followup_addressed=followup_addressed,
        self_id=self_id,
    )
    lines = [
        _bot_self_name_mention_hint(),
        f"- 当前触发人：{current_label}。",
        "- 当前消息优先级最高；最近聊天只用于理解氛围和指代，不代表当前发言人立场。",
        "- 当前要对着说话的人只有上面这个当前触发人；最近真人发言里的其他人是旁人，不要把上下两个人认成同一个，也不要把旁人的话当成当前触发人说的。",
        "- 草稿里的「你」默认对当前触发人；「他/她」不能是当前触发人，也不能自动当成 QQ 回复对象。引语里的人称按原说话人判。",
        _format_message_relation_summary(facts),
    ]
    if discourse_state is not None:
        discourse_block = format_discourse_prompt_block(discourse_state)
        if discourse_block:
            lines.append(discourse_block)
    if addressed_bot:
        if facts.target_scope == "reply_to_other_mentions_bot":
            lines.append(
                "- 当前消息提到了风雪，但回复对象是其他群友；"
                "只有回复正文明确要求风雪接话时，才按在叫你处理。"
            )
        else:
            addressed_parts: list[str] = []
            if mentioned:
                addressed_parts.append("艾特/点名了你")
            if replied_to_bot:
                addressed_parts.append("回复了你之前的话")
            if followup_addressed:
                addressed_parts.append("刚才短时间内艾特/回复过你，当前可能是在延续和你的对话")
            lines.append(f"- 当前是在和风雪互动：{'，'.join(addressed_parts) or '是'}。")
        if followup_addressed:
            lines.append(
                "- 这是短时互动窗口里的后续消息：当前消息通过了后端指向检查，可能是在继续和风雪说话；"
                "如果上下文显示他其实在回复别人或指代不明，不要强行代入自己。"
            )
    else:
        if followup_soft:
            lines.append(
                "- 风雪刚刚和这个人说过话，但已经过了直接接话窗口。"
                "当前不要当成点名，只按普通插话判断：有新角度、笑点、站队或必要时才说，没有新增内容就保持沉默。"
            )
        else:
            lines.append("- 当前不是直接和风雪互动；判断插话时不要把群友互相回复误认为在问你。")
        lines.append(
            "- 当前没有人在和风雪对话：如果决定插话，只能陈述或短评；"
            "禁止反问、追问或用澄清问题把群友对话拉向自己。"
        )

    reply_relation = _extract_reply_relation(current_text)
    if reply_relation is not None:
        speaker_label, target_label = reply_relation
        lines.append(
            f"- 回复关系：{speaker_label} 是当前回复者/当前发言人；"
            f"{target_label} 是被回复对象。不要把 {target_label} 的原话当成 {speaker_label} 说的。"
        )
        if facts.reply_target_is_bot:
            lines.append("- 指代规则：这条是在回复风雪，所以当前回复里的“你”通常指风雪。")
        else:
            lines.append(
                f"- 指代规则：这条首先是在对 {target_label} 说；除非回复正文明确点名风雪，"
                "不要把里面的“你/他/她”自动当成风雪。"
            )

    referent_block = format_referent_prompt_block(reference_resolution)
    if referent_block:
        lines.append(referent_block)
    if reference_resolution.status == RESOLVED and reference_resolution.user_ids:
        labels = _labels_for_user_ids(
            reference_resolution.user_ids,
            recent_messages,
            current_user_id=current_user_id,
            current_nickname=current_nickname,
            self_id=self_id,
        )
        if labels:
            label_text = "、".join(labels)
            lines.append(
                f"- 代词/省略句候选指向：{label_text}；"
                f"来源={reference_resolution.reason}，置信度={reference_resolution.confidence:.2f}。"
            )
    elif reference_resolution.status in {UNAVAILABLE, ERROR}:
        lines.append("- 指代检查不可用，不要把失败当成没有指代，也不要猜人。")
    elif reference_resolution.status == AMBIGUOUS:
        lines.append("- unresolved_reference=true：确实在指人但后端未能唯一解析；优先 clarify，不确定时不要点名或套用某人画像。")
        lines.append("- 当前消息含他/她/这个人/那个人等指代，但后端未能唯一解析；不确定时不要点名或套用某人画像。")
    elif has_strong_person_reference(current_text) and not reference_resolution.user_ids:
        lines.append("- unresolved_reference=true：确实在指人但后端未能唯一解析；优先 clarify，不确定时不要点名或套用某人画像。")
        lines.append("- 当前消息含他/她/这个人/那个人等指代，但后端未能唯一解析；不确定时不要点名或套用某人画像。")
    elif str(getattr(reference_resolution, "kind", "")) == "NON_PERSON":
        if ellipsis_resolution is not None and ellipsis_resolution.status == RESOLVED and ellipsis_resolution.source_text:
            lines.append(
                f"- 当前「这个/那个」已接到前文：{ellipsis_resolution.source_text[:80]}。按这个对象接，不要问缺图或链接。"
            )
        else:
            lines.append("- 当前「那个/这个」等更像在指事/考试/插件/梗，不要绑成某个群友。")

    recent_speakers = _recent_human_speaker_lines(
        recent_messages,
        current_user_id=current_user_id,
        current_nickname=current_nickname,
    )
    if recent_speakers:
        lines.append("- 最近真人发言顺序（旧到新）：")
        lines.extend(recent_speakers)
    ellipsis_block = format_ellipsis_prompt_block(ellipsis_resolution or EllipsisResolution())
    if ellipsis_block:
        lines.append(ellipsis_block)
    repair_block = format_repair_prompt_block(repair_resolution or RepairResolution())
    if repair_block:
        lines.append(repair_block)
    ambiguity_block = format_ambiguity_prompt_block(ambiguity_resolution or AmbiguityResolution())
    if ambiguity_block:
        lines.append(ambiguity_block)
    return "\n".join(lines)


def _extract_reply_relation(text: str) -> tuple[str, str] | None:
    match = REPLY_RELATION_RE.search(text)
    if not match:
        return None
    return match.group("speaker").strip(), match.group("target").strip()


def _label_is_bot_self(label: str, self_id: int) -> bool:
    if not label:
        return False
    if f"#{str(self_id)[-5:]}" in label:
        return True
    return any(alias in label for alias in BOT_SELF_NAME_ALIASES)


def _message_relation_facts(
    *,
    current_user_id: int,
    current_nickname: str,
    current_text: str,
    reference_resolution: ReferenceResolution,
    mentioned: bool,
    replied_to_bot: bool,
    addressed_bot: bool,
    followup_addressed: bool,
    self_id: int,
) -> MessageRelationFacts:
    current_label = _member_label(current_user_id, current_nickname)
    reply_relation = _extract_reply_relation(current_text)
    reply_speaker_label = reply_relation[0] if reply_relation is not None else ""
    reply_target_label = reply_relation[1] if reply_relation is not None else ""
    reply_target_is_bot = replied_to_bot or _label_is_bot_self(reply_target_label, self_id)
    self_name_mentioned = _mentions_bot_self_name(current_text)
    if reference_resolution.status in {UNAVAILABLE, ERROR}:
        ambiguous_reference = False
    else:
        ambiguous_reference = reference_resolution.status == AMBIGUOUS or (
            has_strong_person_reference(current_text)
            and not reference_resolution.user_ids
            and reference_resolution.status != RESOLVED
        )

    if replied_to_bot or reply_target_is_bot:
        target_scope = "reply_to_bot"
        target_note = "当前消息正在回复风雪之前的话"
    elif reply_target_label and mentioned:
        target_scope = "reply_to_other_mentions_bot"
        target_note = f"当前消息在回复 {reply_target_label}，同时提到风雪"
    elif followup_addressed:
        target_scope = "followup_bot"
        target_note = "短时互动窗口判断为可能延续风雪对话"
    elif mentioned:
        target_scope = "mention_bot"
        target_note = "当前消息提到或艾特了风雪"
    elif reply_target_label:
        target_scope = "reply_to_other"
        target_note = f"当前消息首先是在回复 {reply_target_label}"
    elif ambiguous_reference:
        target_scope = "ambiguous"
        target_note = "当前消息有人称/省略指代但后端未能唯一解析"
    else:
        target_scope = "group_chat"
        target_note = "普通群聊消息，未直接指向风雪"

    return MessageRelationFacts(
        current_label=current_label,
        target_scope=target_scope,
        target_note=target_note,
        mentioned_bot=mentioned,
        replied_to_bot=replied_to_bot,
        addressed_bot=addressed_bot,
        followup_addressed=followup_addressed,
        self_name_mentioned=self_name_mentioned,
        reply_speaker_label=reply_speaker_label,
        reply_target_label=reply_target_label,
        reply_target_is_bot=reply_target_is_bot,
        ambiguous_reference=ambiguous_reference,
        reference_user_ids=reference_resolution.user_ids,
        reference_reason=reference_resolution.reason,
        reference_confidence=reference_resolution.confidence,
    )


def _format_message_relation_summary(facts: MessageRelationFacts) -> str:
    pieces = [
        f"target={facts.target_scope}",
        f"说明={facts.target_note}",
        f"mentioned_bot={str(facts.mentioned_bot).lower()}",
        f"replied_to_bot={str(facts.replied_to_bot).lower()}",
        f"followup={str(facts.followup_addressed).lower()}",
    ]
    if facts.reply_target_label:
        pieces.append(f"reply_target={facts.reply_target_label}")
    if facts.reference_user_ids:
        refs = ",".join(str(user_id) for user_id in facts.reference_user_ids[:4])
        pieces.append(f"resolved_refs={refs}")
    return "- 后端关系摘要：" + "；".join(pieces) + "。"


def _labels_for_user_ids(
    user_ids: tuple[int, ...],
    recent_messages: list[ChatMessage],
    *,
    current_user_id: int,
    current_nickname: str,
    self_id: int,
) -> list[str]:
    names: dict[int, str] = {current_user_id: current_nickname}
    for msg in reversed(recent_messages):
        if msg.user_id not in names and msg.nickname:
            names[msg.user_id] = msg.nickname
    labels: list[str] = []
    for user_id in user_ids[:4]:
        if user_id == self_id:
            labels.append(_member_label(user_id, "张风雪"))
        else:
            labels.append(_member_label(user_id, names.get(user_id, str(user_id))))
    return labels


def _recent_human_speaker_lines(
    recent_messages: list[ChatMessage],
    *,
    current_user_id: int,
    current_nickname: str,
) -> list[str]:
    ordered: list[ChatMessage] = [
        msg for msg in recent_messages if not msg.is_bot and msg.text.strip()
    ][-6:]
    if not ordered:
        return [f"  - {_member_label(current_user_id, current_nickname)}：当前消息"]
    lines: list[str] = []
    for msg in ordered:
        prefix = "当前触发人" if msg.user_id == current_user_id else "群友"
        lines.append(f"  - {prefix} {_member_label(msg.user_id, msg.nickname)}：{_short_notice_text(msg.text, 42)}")
    return lines


def _mentions_bot_self_name(text: str) -> bool:
    return any(alias in text for alias in BOT_SELF_NAME_ALIASES)


def _bot_self_name_mention_hint() -> str:
    return (
        "注：风雪和张风雪都是你自己；只有明确提到、艾特或回复风雪/张风雪时才是在说你，"
        "普通“你/她/这个人”不要自动代入。"
    )
