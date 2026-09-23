"""Member identity and memory context formatting for conversation prompts."""

from __future__ import annotations

import re

from .memory import ChatMessage, MemberImpression, MemberProfile, linked_account_note


SELF_MEMORY_QUERY_RE = re.compile(
    r"(?:你)?记录了(?:我|关于我)|你记(?:得|住)我|关于我的哪些|你对我的(?:印象|记忆|记录)|我是谁"
)


def member_label(user_id: int, nickname: str) -> str:
    clean_name = nickname.strip() or str(user_id)
    return f"{clean_name}[#{str(user_id)[-5:]}]"


def trim_inline(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def related_member_user_ids(
    recent_messages: list[ChatMessage], *, current_user_id: int
) -> list[int]:
    user_ids = [current_user_id]
    for msg in reversed(recent_messages):
        if msg.is_bot:
            continue
        user_ids.append(msg.user_id)
        if len(user_ids) >= 12:
            break
    return user_ids


def is_self_memory_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return bool(SELF_MEMORY_QUERY_RE.search(text or "") or SELF_MEMORY_QUERY_RE.search(compact))


def member_memory_user_ids(
    recent_messages: list[ChatMessage], *, current_user_id: int, current_text: str
) -> list[int]:
    if current_user_id and is_self_memory_query(current_text):
        return [int(current_user_id)]
    return related_member_user_ids(recent_messages, current_user_id=current_user_id)


def format_member_context(
    profiles: list[MemberProfile | MemberImpression], *, current_user_id: int = 0
) -> str:
    if not profiles:
        return ""
    lines: list[str] = []
    speaker_lines: list[str] = []
    other_lines: list[str] = []
    for profile in profiles:
        label = member_label(profile.user_id, profile.display_name)
        aliases = [
            alias
            for alias in profile.aliases
            if alias and alias != profile.display_name
        ][:3]
        prefix = f"- {label}"
        if aliases:
            prefix = f"{prefix}，曾用名/历史名：{'、'.join(aliases)}"
        account_note = linked_account_note(profile.user_id)
        if account_note:
            prefix = f"{prefix}；{account_note}"
        details: list[str] = []
        if isinstance(profile, MemberImpression):
            if profile.ai_summary:
                details.append(f"长期印象：{trim_inline(profile.ai_summary, 88)}")
            if profile.ai_interests:
                details.append(f"兴趣/常聊：{'、'.join(profile.ai_interests[:5])}")
            if profile.ai_speaking_style:
                details.append(f"说话方式：{trim_inline(profile.ai_speaking_style, 72)}")
            if profile.top_tags:
                details.append(
                    "后端标签：" + "、".join(f"{tag}x{count}" for tag, count in profile.top_tags[:4])
                )
            if profile.top_keywords:
                details.append("高频词：" + "、".join(term for term, _ in profile.top_keywords[:5]))
            sample_texts = profile.ai_representative_texts or profile.recent_texts
            if sample_texts:
                samples = " / ".join(f"“{trim_inline(text, 34)}”" for text in sample_texts[:2])
                details.append(f"代表性原话：{samples}")
            if profile.message_count:
                details.append(f"已记录发言约 {profile.message_count} 条")
        rendered = f"{prefix}；" + "；".join(details) if details else prefix
        if current_user_id and profile.user_id == current_user_id:
            speaker_lines.append(rendered)
        elif current_user_id:
            other_lines.append(rendered)
        else:
            lines.append(rendered)
    if current_user_id:
        blocks: list[str] = []
        if speaker_lines:
            blocks.append("当前触发人画像（回答「我/你记录了我什么」时只用这段，不要用旁人）：")
            blocks.extend(speaker_lines)
        if other_lines:
            blocks.append("旁人画像（不是当前触发人，禁止说成当前触发人的经历）：")
            blocks.extend(other_lines)
        return "\n".join(blocks)
    return "\n".join(lines)
