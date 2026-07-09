from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .memory import ChatMessage
from .persona import Persona


QUESTION_MARKERS = ("?", "？", "吗", "么", "咋", "怎么", "如何", "为什么", "哪", "谁", "啥")
LOW_VALUE_TEXT = {"草", "6", "？", "?", "啊", "哦", "嗯", "哈哈", "hhh", "笑死"}


@dataclass(frozen=True)
class ScoreResult:
    score: int
    should_reply: bool
    reasons: tuple[str, ...]


def score_message(
    *,
    text: str,
    recent_messages: list[ChatMessage],
    persona: Persona,
    mentioned: bool,
    replied_to_bot: bool,
    passive_threshold: int,
    passive_probability: float,
    rng: random.Random | None = None,
) -> ScoreResult:
    rng = rng or random.Random()
    stripped = text.strip()
    score = 0
    reasons: list[str] = []

    if mentioned:
        score += 100
        reasons.append("mentioned")
    if replied_to_bot:
        score += 80
        reasons.append("reply_to_bot")
    if any(marker in stripped for marker in QUESTION_MARKERS):
        score += 28
        reasons.append("question")

    matched_keywords = [kw for kw in persona.keywords if kw and kw.lower() in stripped.lower()]
    if matched_keywords:
        score += min(42, 14 + len(matched_keywords) * 9)
        reasons.append("persona_keyword:" + ",".join(matched_keywords[:3]))

    if len(stripped) >= 2 and stripped not in LOW_VALUE_TEXT:
        score += 20
        reasons.append("casual_chat")

    if len(stripped) >= 18:
        score += 10
        reasons.append("substantial_text")

    if _topic_heat(recent_messages, stripped):
        score += 16
        reasons.append("topic_heat")

    if _is_group_spam(recent_messages):
        score -= 15
        reasons.append("group_spam")

    if stripped in LOW_VALUE_TEXT or len(stripped) <= 1:
        score -= 10
        reasons.append("low_value_text")

    if _recent_bot_spoke(recent_messages, 25):
        score -= 10
        reasons.append("recent_bot_reply")

    if mentioned or replied_to_bot:
        return ScoreResult(score=score, should_reply=score > 0, reasons=tuple(reasons))

    if score < passive_threshold:
        return ScoreResult(score=score, should_reply=False, reasons=tuple(reasons))

    probability = min(passive_probability, persona.passive_reply_probability)
    should_reply = rng.random() < probability
    if should_reply:
        reasons.append("passive_probability_hit")
    else:
        reasons.append("passive_probability_miss")
    return ScoreResult(score=score, should_reply=should_reply, reasons=tuple(reasons))


def _topic_heat(recent_messages: list[ChatMessage], text: str) -> bool:
    if not recent_messages or len(text) < 4:
        return False
    current_terms = {token for token in _simple_terms(text) if len(token) >= 2}
    if not current_terms:
        return False
    hits = 0
    for msg in recent_messages[-8:]:
        if current_terms.intersection(_simple_terms(msg.text)):
            hits += 1
    return hits >= 2


def _simple_terms(text: str) -> set[str]:
    chunks = text.replace("，", " ").replace("。", " ").replace(",", " ").replace(".", " ")
    return {chunk.strip().lower() for chunk in chunks.split() if chunk.strip()}


def _is_group_spam(recent_messages: list[ChatMessage]) -> bool:
    now = time.time()
    return sum(1 for msg in recent_messages if now - msg.created_at <= 15) >= 8


def _recent_bot_spoke(recent_messages: list[ChatMessage], seconds: int) -> bool:
    now = time.time()
    return any(msg.is_bot and now - msg.created_at <= seconds for msg in recent_messages)
