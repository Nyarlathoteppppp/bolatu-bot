from __future__ import annotations

import re
from dataclasses import dataclass


MEMORY_SIGNALS = (
    "记得",
    "还记得",
    "之前",
    "以前",
    "上次",
    "刚才",
    "前面",
    "谁说",
    "说过",
    "聊过",
    "提过",
    "历史",
    "原来",
    "当时",
    "哪个群友",
    "什么意思",
    "什么梗",
    "黑话",
    "别名",
    "改名",
)

PERSON_PAST_SIGNALS = (
    "过去",
    "从前",
    "曾经",
    "以前",
    "之前",
    "当年",
    "当初",
    "那时候",
    "那会儿",
    "早年",
    "早些时候",
    "学生时代",
    "大学时",
    "高中时",
    "初中时",
    "原来",
    "后来",
    "最早",
)

PERSON_PRONOUN_RE = re.compile(r"(?:他|她|这个人|那个人|这位群友|那位群友)")

KNOWLEDGE_SIGNALS = (
    "文件",
    "附件",
    "文档",
    "PDF",
    "pdf",
    "网页",
    "链接",
    "文章",
    "资料",
    "页面",
)


@dataclass(frozen=True)
class RAGQueryPlan:
    enabled: bool
    lexical: bool
    semantic: bool
    route: str


def plan_rag_query(
    text: str,
    *,
    addressed: bool,
    related_user_ids: list[int] | None = None,
    has_person_reference: bool = False,
) -> RAGQueryPlan:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if len(clean) < 2:
        return RAGQueryPlan(False, False, False, "too_short")
    memory_signal = any(signal in clean for signal in MEMORY_SIGNALS)
    past_signal = any(signal in clean for signal in PERSON_PAST_SIGNALS)
    contains_identifier = bool(re.search(r"\d{5,12}", clean))
    pronoun_person_reference = bool(related_user_ids and PERSON_PRONOUN_RE.search(clean))
    person_past = past_signal and (has_person_reference or contains_identifier or pronoun_person_reference)
    knowledge_signal = any(signal in clean for signal in KNOWLEDGE_SIGNALS)
    lexical = len(clean) >= 3
    # A remote query embedding must not become part of every addressed reply.
    # Existing structured member/profile selectors cover ordinary person questions;
    # semantic RAG is reserved for explicit historical recall or exact QQ identifiers.
    semantic = person_past or memory_signal or contains_identifier or knowledge_signal
    if person_past:
        route = "person_past"
    elif knowledge_signal:
        route = "knowledge"
    elif memory_signal:
        route = "explicit_memory"
    elif contains_identifier:
        route = "identifier"
    else:
        route = "lexical"
    return RAGQueryPlan(lexical, lexical, semantic, route)
