from __future__ import annotations

import re
from dataclasses import dataclass


_REPLY_ENVELOPE_MARKER = "消息【"
_CURRENT_REPLY_PREFIX_RE = re.compile(r"(?:^|；)[^；【】\n]{0,180}回复[^：\n]{1,180}：")
_CURRENT_REPLY_ASCII_PREFIX_RE = re.compile(r"(?:^|；)[^；【】\n]{0,180}回复[^:\n]{1,180}:")
_EXPLICIT_RECALL_PATTERNS = (
    re.compile(
        r"(?:之前|以前|上次|刚才|前面)"
        r"(?:我们|你们|咱们|群里|这里|大家)?"
        r"(?:聊过|说过|提过|讨论过|讲过)"
        r"(?P<topic>.+)$"
    ),
    re.compile(
        r"(?:还记得|记得)"
        r"(?:我们|你们|咱们|群里|这里|大家)?"
        r"(?:之前|以前|上次|刚才|前面)?"
        r"(?:聊过|说过|提过|讨论过|讲过)?"
        r"(?P<topic>.+)$"
    ),
)
_RECALL_TOPIC_SUFFIX_RE = re.compile(
    r"(?:你)?(?:忘了|不记得了|还记得吗|记得吗|记不记得|还记得|记得)"
    r"[？?。！!~～]*$"
)
_QUESTION_SUFFIX_RE = re.compile(r"(?:吗|嘛|呢|来着)[？?。！!~～]*$")
_LEADING_FILLER_RE = re.compile(r"^(?:一下|一下子|那个|这个|关于|有关|的事|的话题)+")


@dataclass(frozen=True)
class NormalizedRAGQuery:
    text: str
    current_utterance: str
    focused_topic: str = ""
    reply_envelope_removed: bool = False


def normalize_rag_query(text: str) -> NormalizedRAGQuery:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    current, envelope_removed = _extract_current_utterance(raw)
    topic = _explicit_recall_topic(current)
    query = topic or current or raw
    return NormalizedRAGQuery(
        text=query[:500],
        current_utterance=(current or raw)[:500],
        focused_topic=topic[:120],
        reply_envelope_removed=envelope_removed,
    )


def _extract_current_utterance(text: str) -> tuple[str, bool]:
    if _REPLY_ENVELOPE_MARKER not in text:
        return text, False
    # The normalized reply context ends with “某人回复某人：当前正文】”. Find
    # the last reply-prefix rather than splitting on semicolons because the
    # current user's own text may contain semicolons too.
    for pattern in (_CURRENT_REPLY_PREFIX_RE, _CURRENT_REPLY_ASCII_PREFIX_RE):
        matches = tuple(pattern.finditer(text))
        if not matches:
            continue
        current = re.sub(r"\s+", " ", text[matches[-1].end() :]).strip().rstrip("】").strip()
        if current:
            return current, True
    return text, False


def _explicit_recall_topic(text: str) -> str:
    compact = re.sub(r"\s+", "", text).strip()
    for pattern in _EXPLICIT_RECALL_PATTERNS:
        match = pattern.search(compact)
        if match is None:
            continue
        topic = match.group("topic")
        topic = _RECALL_TOPIC_SUFFIX_RE.sub("", topic)
        topic = _QUESTION_SUFFIX_RE.sub("", topic)
        topic = _LEADING_FILLER_RE.sub("", topic)
        topic = topic.strip("，。！？?！：:；;、~～的了")
        if len(topic) >= 2:
            return topic
    return ""
