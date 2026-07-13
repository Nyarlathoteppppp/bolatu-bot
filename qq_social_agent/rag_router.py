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


@dataclass(frozen=True)
class RAGQueryPlan:
    enabled: bool
    lexical: bool
    semantic: bool
    route: str


def plan_rag_query(text: str, *, addressed: bool, related_user_ids: list[int] | None = None) -> RAGQueryPlan:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if len(clean) < 2:
        return RAGQueryPlan(False, False, False, "too_short")
    memory_signal = any(signal in clean for signal in MEMORY_SIGNALS)
    contains_identifier = bool(re.search(r"\d{5,12}", clean))
    lexical = len(clean) >= 3
    # A remote query embedding must not become part of every addressed reply.
    # Existing structured member/profile selectors cover ordinary person questions;
    # semantic RAG is reserved for explicit historical recall or exact QQ identifiers.
    semantic = memory_signal or contains_identifier
    if memory_signal:
        route = "explicit_memory"
    elif contains_identifier:
        route = "identifier"
    else:
        route = "lexical"
    return RAGQueryPlan(lexical, lexical, semantic, route)
