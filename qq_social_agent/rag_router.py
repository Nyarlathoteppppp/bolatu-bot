from __future__ import annotations

import re
from dataclasses import dataclass


MEMORY_SIGNALS = (
    "记得",
    "还记得",
    "之前",
    "以前",
    "上次",
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

RECENT_CONTEXT_SIGNALS = (
    "刚才",
    "前面",
    "上面",
    "上一条",
    "这句话",
    "这句",
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
    recent_context_signal = any(signal in clean for signal in RECENT_CONTEXT_SIGNALS)
    past_signal = any(signal in clean for signal in PERSON_PAST_SIGNALS)
    contains_identifier = bool(re.search(r"\d{5,12}", clean))
    pronoun_person_reference = bool(related_user_ids and PERSON_PRONOUN_RE.search(clean))
    person_past = past_signal and (has_person_reference or contains_identifier or pronoun_person_reference)
    knowledge_signal = any(signal in clean for signal in KNOWLEDGE_SIGNALS)
    if person_past:
        route = "person_past"
    elif knowledge_signal:
        route = "knowledge"
    elif recent_context_signal:
        # The normal recent-message window is more accurate and cheaper for
        # "刚才/上面" than historical RAG, which intentionally excludes fresh rows.
        return RAGQueryPlan(False, False, False, "recent_context")
    elif memory_signal:
        route = "explicit_memory"
    elif contains_identifier:
        route = "identifier"
    else:
        return RAGQueryPlan(False, False, False, "casual")
    return RAGQueryPlan(True, True, True, route)
