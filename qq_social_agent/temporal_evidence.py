from __future__ import annotations

import math
import re
import time
from enum import Enum


class TemporalIntent(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    NEUTRAL = "neutral"


CURRENT_SIGNALS = (
    "现在",
    "目前",
    "如今",
    "最近",
    "今年",
    "最新",
    "已经",
    "还在",
    "到底",
)

HISTORICAL_SIGNALS = (
    "以前",
    "之前",
    "过去",
    "当时",
    "那时候",
    "原来",
    "曾经",
    "当年",
    "最早",
)

NEGATIVE_MARKERS = (
    "不再",
    "没有",
    "还没",
    "并非",
    "不是",
    "不会",
    "不要",
    "不想",
    "不读",
    "不考",
    "放弃",
    "取消",
    "停止",
    "离开",
    "没",
    "不",
)

POSITIVE_MARKERS = (
    "已经",
    "仍然",
    "继续",
    "开始",
    "准备",
    "打算",
    "决定",
    "会",
    "要",
    "想",
    "喜欢",
    "在",
    "有",
    "是",
)


def detect_temporal_intent(query: str) -> TemporalIntent:
    clean = re.sub(r"\s+", "", str(query))
    current = any(signal in clean for signal in CURRENT_SIGNALS)
    historical = any(signal in clean for signal in HISTORICAL_SIGNALS)
    # A question such as “之前说过考研，现在呢” asks for the current state;
    # the historical phrase supplies comparison context and must not neutralize it.
    if current:
        return TemporalIntent.CURRENT
    if historical:
        return TemporalIntent.HISTORICAL
    return TemporalIntent.NEUTRAL


def recency_adjustment(
    created_at: float,
    intent: TemporalIntent,
    *,
    now: float | None = None,
) -> float:
    if intent is TemporalIntent.HISTORICAL:
        return 0.0
    age_days = max(0.0, ((now if now is not None else time.time()) - created_at) / 86400.0)
    if intent is TemporalIntent.CURRENT:
        if age_days <= 7:
            return 0.14
        if age_days <= 30:
            return 0.10
        if age_days <= 180:
            return 0.04
        if age_days > 365:
            return -0.08
        return 0.0
    # Neutral queries get only a small freshness tie-breaker. This is not enough
    # to hide older evidence when the user is asking about history.
    return 0.05 * math.exp(-age_days / 45.0)


def statements_conflict(newer: str, older: str, query_terms: list[str]) -> bool:
    """Detect only explicit polarity changes around a shared query term.

    This intentionally has high precision and low recall. Ambiguous differences
    remain visible as a timeline instead of being silently treated as a conflict.
    """

    for term in query_terms:
        if len(term) < 2 or term not in newer or term not in older:
            continue
        newer_polarity = _local_polarity(newer, term)
        older_polarity = _local_polarity(older, term)
        if newer_polarity and older_polarity and newer_polarity != older_polarity:
            return True
    return False


def evidence_kind_label(kind: str) -> str:
    return {
        "reported_claim": "说话者当时陈述",
        "summary": "历史摘要，需回看原话",
        "structured_fact": "结构化事实",
        "directory_fact": "群资料事实",
        "profile_summary": "阶段画像推断",
        "curated_definition": "人工维护定义",
        "approval_feedback": "审批反馈",
        "reference": "文件或网页资料",
        "unknown": "未分类证据",
    }.get(str(kind), "未分类证据")


def default_evidence_kind(doc_type: str) -> str:
    return {
        "conversation": "reported_claim",
        "summary": "summary",
        "memory_atom": "structured_fact",
        "member": "profile_summary",
        "jargon": "curated_definition",
        "feedback": "approval_feedback",
        "file_knowledge": "reference",
        "web_knowledge": "reference",
    }.get(str(doc_type), "unknown")


def _local_polarity(text: str, term: str) -> int:
    positions = [match.start() for match in re.finditer(re.escape(term), text)]
    for position in positions:
        window = text[max(0, position - 12) : position + len(term) + 8]
        if any(marker in window for marker in NEGATIVE_MARKERS):
            return -1
        if any(marker in window for marker in POSITIVE_MARKERS):
            return 1
    return 0
