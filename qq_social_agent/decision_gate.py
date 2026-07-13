from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .cue_patterns import CueRepeatState
from .deepseek_client import ReplyDecision, ToolSymbol
from .memory import ChatMessage
from .persona import Persona
from .tools.fresh_context import FreshIntent
from .tools.market_intent import MarketIntent


@dataclass(frozen=True)
class PreDecisionGateResult:
    decision: ReplyDecision | None = None
    skip_reason: str = ""


def pre_decision_gate(
    *,
    text: str,
    recent_messages: list[ChatMessage],
    persona: Persona,
    addressed_bot: bool,
    mentioned: bool,
    replied_to_bot: bool,
    cue_repeat_state: CueRepeatState | None,
    market_intents: list[MarketIntent],
    fresh_intent: FreshIntent | None,
) -> PreDecisionGateResult:
    if addressed_bot:
        if market_intents and is_explicit_market_lookup(text):
            return PreDecisionGateResult(
                decision=_market_decision(
                    text=text,
                    market_intents=market_intents,
                    reason="local_addressed_market_lookup",
                    confidence=1.0,
                    mode="addressed",
                )
            )
        if fresh_intent is not None and (fresh_intent.explicit or fresh_intent.required):
            return PreDecisionGateResult(
                decision=ReplyDecision(
                    should_reply=True,
                    confidence=1.0,
                    reason="local_addressed_fresh_lookup",
                    mode="addressed",
                    action="answer",
                    need_fresh_context=True,
                    fresh_query=fresh_intent.query,
                    fresh_kind=fresh_intent.kind,
                )
            )
        return PreDecisionGateResult(
            decision=ReplyDecision(
                should_reply=True,
                confidence=1.0,
                reason="local_addressed_reply",
                mode="addressed",
                action="answer" if _looks_like_question_or_request(text) else "reply",
            )
        )

    if is_low_value_group_text(text):
        return PreDecisionGateResult(skip_reason="low_value_local")

    if market_intents and is_explicit_market_lookup(text):
        return PreDecisionGateResult(
            decision=_market_decision(
                text=text,
                market_intents=market_intents,
                reason="local_explicit_market_lookup",
            )
        )

    return PreDecisionGateResult()


def apply_backend_tool_decision(
    decision: ReplyDecision,
    *,
    text: str,
    market_intents: list[MarketIntent],
    fresh_intent: FreshIntent | None,
) -> ReplyDecision:
    if not decision.should_reply:
        return decision
    result = decision
    if market_intents and is_explicit_market_lookup(text):
        result = _market_decision(
            text=text,
            market_intents=market_intents,
            reason=decision.reason or "backend_market_lookup",
            confidence=decision.confidence,
            mode=decision.mode,
        )
    if fresh_intent is not None and (fresh_intent.explicit or fresh_intent.required):
        action = "answer" if result.action == "fresh_context" else result.action
        result = replace(
            result,
            action=action,
            need_fresh_context=True,
            fresh_query=fresh_intent.query,
            fresh_kind=fresh_intent.kind,
        )
    return result


def is_low_value_group_text(text: str) -> bool:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    if not compact:
        return True
    if compact in LOW_VALUE_GROUP_TEXTS:
        return True
    if re.fullmatch(r"[6哈啊呃嗯哦噢喔草艹wW]+", compact) and len(compact) <= 8:
        return True
    return False


def is_explicit_market_lookup(text: str) -> bool:
    lowered = text.casefold()
    lookup_terms = (
        "股价",
        "价格",
        "多少钱",
        "多少",
        "行情",
        "盘前",
        "盘后",
        "财报",
        "涨",
        "跌",
        "走势",
        "能买吗",
        "还能买",
        "能冲",
        "做多",
        "做空",
        "爆仓",
        "怎么了",
        "咋了",
        "咋样",
        "如何",
        "怎么看",
        "为什么",
        "今天",
        "现在",
    )
    return any(term in lowered for term in lookup_terms)


def _looks_like_question_or_request(text: str) -> bool:
    markers = (
        "?",
        "？",
        "吗",
        "么",
        "怎么",
        "咋",
        "为什么",
        "谁",
        "哪",
        "多少",
        "能不能",
        "可以吗",
        "帮我",
        "说说",
        "解释",
        "看看",
    )
    return any(marker in text for marker in markers)


def market_comment_after_tool(text: str) -> bool:
    lowered = text.casefold()
    comment_terms = (
        "怎么看",
        "怎么了",
        "咋了",
        "咋样",
        "如何",
        "为什么",
        "原因",
        "能买吗",
        "还能买",
        "能冲",
        "走势",
        "做多",
        "做空",
        "该不该",
        "要不要",
        "今天",
        "现在",
    )
    return any(term in lowered for term in comment_terms)


def context_query_text(
    current_text: str,
    current_nickname: str,
    recent_messages: list[ChatMessage],
) -> str:
    parts = [current_nickname, current_text]
    for msg in recent_messages[-5:]:
        if msg.is_bot:
            continue
        parts.append(msg.nickname)
        parts.append(msg.text)
    return "\n".join(parts)


def _market_decision(
    *,
    text: str,
    market_intents: list[MarketIntent],
    reason: str,
    confidence: float = 0.78,
    mode: str = "tool",
) -> ReplyDecision:
    return ReplyDecision(
        should_reply=True,
        confidence=confidence,
        reason=reason,
        mode=mode or "tool",
        action="market_check",
        need_tool=True,
        tool="market",
        symbols=_tool_symbols_from_market_intents(market_intents),
        comment_after_tool=market_comment_after_tool(text),
    )


def _tool_symbols_from_market_intents(intents: list[MarketIntent]) -> tuple[ToolSymbol, ...]:
    return tuple(
        ToolSymbol(kind=intent.kind, symbol=intent.symbol, display=intent.display_name)
        for intent in intents[:2]
    )


def _has_interesting_local_signal(text: str, persona: Persona) -> bool:
    lowered = text.casefold()
    if any(str(keyword).casefold() in lowered for keyword in persona.keywords if keyword):
        return True
    if any(
        marker in text
        for marker in (
            "?",
            "？",
            "吗",
            "么",
            "怎么",
            "咋",
            "为什么",
            "谁",
            "哪个",
            "哪边",
            "要不要",
            "该不该",
            "能不能",
            "怎么办",
            "咋办",
            "还是",
        )
    ):
        return True
    if any(
        term in lowered
        for term in (
            "亏",
            "亏了",
            "没人理",
            "坏没坏",
            "完蛋",
            "寄",
            "破防",
            "倒霉",
            "绷",
            "离谱",
            "抽象",
            "傻逼",
            "弱智",
            "闹麻",
            "赢麻",
            "乐死",
            "笑死",
            "成本",
            "风险",
            "工资",
            "就业",
            "学校",
            "专业",
            "钱",
            "贵",
            "便宜",
            "胖",
            "饿",
            "累",
            "劳累",
            "烦",
            "吐了",
        )
    ):
        return True
    return len(text.strip()) >= 18


LOW_VALUE_GROUP_TEXTS = {
    "6",
    "66",
    "666",
    "嗯",
    "嗯嗯",
    "哦",
    "噢",
    "喔",
    "啊",
    "呃",
    "绷",
    "蹦",
    "没绷住",
    "没绷住了",
    "绷不住",
    "绷不住了",
    "哈哈",
    "哈哈哈",
    "哈哈哈哈",
    "好",
    "好好",
    "好好好",
    "好的",
    "一般",
    "可以",
    "草",
    "艹",
    "nb",
    "ok",
    "OK",
}
