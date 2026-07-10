from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from nonebot import logger
from openai import AsyncOpenAI

from .config import DeepSeekConfig, LLMModelRoute, LLMProviderConfig, parse_llm_model_route
from .memory import ChatMessage
from .persona import Persona
from .prompts import PromptRegistry


@dataclass(frozen=True)
class ToolSymbol:
    kind: str
    symbol: str
    display: str


@dataclass(frozen=True)
class ReplyDecision:
    should_reply: bool
    confidence: float
    reason: str
    mode: str = "silent"
    action: str = "ignore"
    need_tool: bool = False
    tool: str = ""
    symbols: tuple[ToolSymbol, ...] = ()
    comment_after_tool: bool = False
    need_fresh_context: bool = False
    fresh_query: str = ""
    fresh_kind: str = "news"


@dataclass(frozen=True)
class MidMemoryDraft:
    summary: str
    recall_cues: tuple[str, ...]


@dataclass(frozen=True)
class StyleRuleDraft:
    situation: str
    style: str
    source_text: str = ""


@dataclass(frozen=True)
class MemberProfileDraft:
    summary: str
    interests: tuple[str, ...]
    speaking_style: str
    representative_texts: tuple[str, ...]


@dataclass(frozen=True)
class ReplyCandidateDraft:
    text: str
    action: str
    style: str


SOCIAL_ACTIONS = {
    "ignore",
    "reply",
    "answer",
    "agree",
    "care",
    "tease",
    "ask_back",
    "mock_repeated_question",
    "at_someone",
    "market_check",
    "fresh_context",
}

LLMUsageRecorder = Callable[[str, str, Optional[int], Optional[int], Optional[int]], None]
_usage_recorder: LLMUsageRecorder | None = None


def set_usage_recorder(recorder: LLMUsageRecorder | None) -> None:
    global _usage_recorder
    _usage_recorder = recorder


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.clients: dict[str, AsyncOpenAI] = {}
        for name, provider in config.providers.items():
            api_key = os.getenv(provider.api_key_env)
            if not api_key:
                logger.warning(
                    "qq_social_agent llm provider key missing: "
                    f"provider={name} env={provider.api_key_env}"
                )
                continue
            self.clients[name] = AsyncOpenAI(
                api_key=api_key,
                base_url=provider.base_url,
                timeout=config.timeout_seconds,
            )
        if not self.clients:
            raise RuntimeError("No LLM API key is configured. Put provider keys in .env.")
        self.route_overrides: dict[str, LLMModelRoute] = {}
        self.prompts = PromptRegistry()

    async def _chat_completion(
        self,
        *,
        task: str,
        route_name: str,
        request: dict[str, object],
    ) -> object:
        routes = self._candidate_routes(route_name)
        last_error: Exception | None = None
        for route in routes:
            client = self.clients.get(route.provider)
            if client is None:
                logger.warning(
                    "qq_social_agent llm provider unavailable, trying fallback: "
                    f"task={task} provider={route.provider} model={route.model}"
                )
                continue
            provider = self.config.providers[route.provider]
            provider_request = dict(request)
            provider_request["model"] = route.model
            extra_body = _extra_body_for_route(provider, route)
            if extra_body:
                provider_request["extra_body"] = extra_body
            try:
                response = await client.chat.completions.create(**provider_request)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "qq_social_agent llm provider failed, trying fallback: "
                    f"task={task} provider={route.provider} model={route.model} error={exc}"
                )
                continue
            if self.config.usage_tracking_enabled:
                _log_llm_usage(task, response, model=route.label)
            return response
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No available LLM provider for route={route_name}")

    def _candidate_routes(self, route_name: str) -> tuple[LLMModelRoute, ...]:
        primary = self.route_overrides.get(route_name, self.config.routes[route_name])
        fallback = self.config.fallback_routes.get(route_name)
        if fallback is None or fallback == primary:
            return (primary,)
        return (primary, fallback)

    def parse_model_route(self, value: str, *, default_provider: str = "siliconflow") -> LLMModelRoute:
        if default_provider not in self.config.providers:
            default_provider = "deepseek"
        return parse_llm_model_route(value, self.config.providers, default_provider=default_provider)

    def set_route_override(self, route_name: str, route: LLMModelRoute | None) -> None:
        if route_name not in self.config.routes:
            raise ValueError(f"unknown route: {route_name}")
        if route is None:
            self.route_overrides.pop(route_name, None)
            return
        self.route_overrides[route_name] = route

    def current_route(self, route_name: str) -> LLMModelRoute:
        return self.route_overrides.get(route_name, self.config.routes[route_name])

    async def should_reply(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool = False,
        replied_to_bot: bool = False,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        market_topic: bool = False,
        chat_label: str = "QQ 群聊",
        memory_context: str = "",
        style_context: str = "",
        raw_corpus_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
        fresh_context_hint: str = "",
    ) -> ReplyDecision:
        context = "\n".join(_format_decision_message(msg) for msg in recent_messages[-30:])
        if not context:
            context = "（暂无更多上下文）"
        addressed = mentioned or replied_to_bot
        interaction_state = "有人艾特或回复了你，这是强信号，但不是必须回复。"
        if not addressed:
            interaction_state = "当前没有艾特你，也不是回复你，你是在判断要不要自然插话。"
        elif addressed_repeat_count >= 3:
            interaction_state = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次艾特或回复你；"
                "这是重复 cue，不要机械回答他问什么。"
            )
        if cue_repeat_context:
            interaction_state = f"{interaction_state}\n反复题型状态：{cue_repeat_context}"
        system = self.prompts.render(
            "decision",
            "system",
            persona_name=persona.name,
            persona_decision_prompt=persona.decision_prompt,
        )
        user = self.prompts.render(
            "decision",
            "user",
            chat_label=chat_label,
            interaction_state=interaction_state,
            fresh_context_hint_section=_optional_section("后端最新背景候选", fresh_context_hint),
            context=context,
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            jargon_context_section=_optional_section("群内黑话词典", jargon_context),
            current_nickname=current_nickname,
            current_text=current_text,
        )
        response = await self._chat_completion(
            task="decision",
            route_name="decision",
            request={
                "temperature": 0.2,
                "max_tokens": 180,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        content = response.choices[0].message.content or ""
        return _parse_reply_decision(content)

    async def select_jargon_terms(
        self,
        *,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        jargon_catalog: str,
        heuristic_terms: tuple[str, ...] = (),
        chat_label: str = "QQ 群聊",
    ) -> tuple[str, ...]:
        context_messages = recent_messages[-18:]
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        heuristic_text = "、".join(heuristic_terms) if heuristic_terms else "无"
        system = self.prompts.render("jargon_select", "system")
        user = self.prompts.render(
            "jargon_select",
            "user",
            chat_label=chat_label,
            jargon_catalog=jargon_catalog,
            heuristic_text=heuristic_text,
            context=context,
            current_nickname=current_nickname,
            current_text=current_text,
        )
        response = await self._chat_completion(
            task="jargon",
            route_name="jargon",
            request={
                "temperature": 0.1,
                "max_tokens": 160,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return _parse_jargon_terms(response.choices[0].message.content or "")

    async def reply(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        action: str = "reply",
        chat_label: str = "QQ 群聊",
        market_context: str = "",
        fresh_context: str = "",
        memory_context: str = "",
        style_context: str = "",
        raw_corpus_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
        recall_feedback_context: str = "",
        mention_targets: str = "",
        priority_context: str = "",
        include_bot_history: bool = True,
    ) -> str:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        mode = "你被直接点名或回复，需要回应。" if mentioned else "你是自然插话，只能在合适时短句接话。"
        if mentioned and addressed_repeat_count >= 3:
            mode = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次点名或回复你。"
                "你可以像真人一样先吐槽他反复 cue 你，而不是直接回答问题。"
            )
        if mentioned and cue_repeat_context:
            mode = f"{mode}\n反复题型状态：{cue_repeat_context}"
        normalized_action = _normalize_action(action, should_reply=True)
        action_guide = self.prompts.action_guide(
            normalized_action,
            self.prompts.action_guide("reply", "行动：普通接话。结合群友聊天内容接一句话。"),
        )
        silence_rule = (
            "- 当前是直接对话，必须给出自然回复，绝对不要输出“空字符串”或类似占位文本。"
            if mentioned
            else "- 不合适回复时输出真正的空内容，绝对不要写出“空字符串”四个字。"
        )
        system = self.prompts.render(
            "reply",
            "system",
            persona_prompt=persona.prompt,
            chat_label=chat_label,
            mode=mode,
            action_guide=action_guide,
            max_reply_chars=persona.max_reply_chars,
            silence_rule=silence_rule,
        )
        market_section = f"\n\n{market_context}" if market_context else ""
        fresh_section = f"\n\n{fresh_context}" if fresh_context else ""
        user = self.prompts.render(
            "reply",
            "user",
            context=context,
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            recall_feedback_context_section=_optional_section("主人撤回反馈", recall_feedback_context),
            style_context_section=_optional_section("群聊表达风格参考", style_context),
            raw_corpus_context_section=_optional_section("群友原文语料参考", raw_corpus_context),
            jargon_context_section=_optional_section("群内黑话词典", jargon_context),
            mention_targets_section=_optional_section("可艾特目标", mention_targets),
            priority_context_section=_optional_section("私聊优先级", priority_context),
            market_section=market_section,
            fresh_section=fresh_section,
            current_nickname=current_nickname,
            current_text=current_text,
        )
        request = {
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.thinking == "enabled":
            request["reasoning_effort"] = self.config.reasoning_effort
        else:
            request["temperature"] = self.config.temperature

        response = await self._chat_completion(task="reply", route_name="reply", request=request)
        content = response.choices[0].message.content or ""
        return _sanitize_reply(content, persona.max_reply_chars)

    async def summarize_member_profile(
        self,
        *,
        messages: list[ChatMessage],
        member_label: str,
        chat_label: str = "QQ 群聊",
    ) -> MemberProfileDraft:
        context = "\n".join(_format_message(msg) for msg in messages)
        if not context:
            return MemberProfileDraft("", (), "", ())
        system = (
            "你只做群友画像摘要。只根据给出的这个人的原始发言判断，"
            "不要编造身份、现实信息或关系；不要给政治立场、意识形态、阵营归属下定性标签，"
            "只可客观写成常聊话题和表达习惯。输出严格 JSON。"
        )
        user = (
            f"聊天场景：{chat_label}\n"
            f"画像对象：{member_label}\n"
            "任务：总结这个群友的发言印象、兴趣话题、说话方式，并挑选少量代表性原话。\n"
            "要求：短、具体、可用于以后回复这个人；不要用政治立场标签概括这个人；"
            "代表性原话必须来自原文，不要改写。\n"
            "JSON 格式："
            "{\"summary\":\"...\",\"interests\":[\"...\"],"
            "\"speaking_style\":\"...\",\"representative_texts\":[\"...\"]}\n\n"
            f"该群友最近发言：\n{context}"
        )
        response = await self._chat_completion(
            task="member_profile",
            route_name="member_profile",
            request={
                "temperature": 0.2,
                "max_tokens": 420,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return _parse_member_profile_draft(response.choices[0].message.content or "")

    async def reply_candidates(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        action: str = "reply",
        chat_label: str = "QQ 群聊",
        market_context: str = "",
        fresh_context: str = "",
        memory_context: str = "",
        style_context: str = "",
        raw_corpus_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
        recall_feedback_context: str = "",
        positive_feedback_context: str = "",
        mention_targets: str = "",
        include_bot_history: bool = True,
        candidate_count: int = 3,
    ) -> tuple[ReplyCandidateDraft, ...]:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        mode = "你被直接点名或回复，需要回应。" if mentioned else "你是自然插话，只能在合适时短句接话。"
        if mentioned and addressed_repeat_count >= 3:
            mode = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次点名或回复你。"
                "你可以像真人一样先吐槽他反复 cue 你，而不是直接回答问题。"
            )
        if mentioned and cue_repeat_context:
            mode = f"{mode}\n反复题型状态：{cue_repeat_context}"
        normalized_action = _normalize_action(action, should_reply=True)
        action_guide = self.prompts.action_guide(
            normalized_action,
            self.prompts.action_guide("reply", "行动：普通接话。结合群友聊天内容接一句话。"),
        )
        system = self.prompts.render(
            "reply_candidates",
            "system",
            persona_prompt=persona.prompt,
            chat_label=chat_label,
            mode=mode,
            action_guide=action_guide,
            candidate_count=candidate_count,
            max_reply_chars=persona.max_reply_chars,
            normalized_action=normalized_action,
        )
        market_section = f"\n\n{market_context}" if market_context else ""
        fresh_section = f"\n\n{fresh_context}" if fresh_context else ""
        user = self.prompts.render(
            "reply_candidates",
            "user",
            context=context,
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            recall_feedback_context_section=_optional_section("主人撤回/不准奏反馈", recall_feedback_context),
            positive_feedback_context_section=_optional_section("审批人标记过的优质发言方向", positive_feedback_context),
            style_context_section=_optional_section("群聊表达风格参考", style_context),
            raw_corpus_context_section=_optional_section("群友原文语料参考", raw_corpus_context),
            jargon_context_section=_optional_section("群内黑话词典", jargon_context),
            mention_targets_section=_optional_section("可艾特目标", mention_targets),
            market_section=market_section,
            fresh_section=fresh_section,
            current_nickname=current_nickname,
            current_text=current_text,
            candidate_count=candidate_count,
        )
        request = {
            "max_tokens": max(self.config.max_tokens, 620),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.thinking == "enabled":
            request["reasoning_effort"] = self.config.reasoning_effort
        else:
            request["temperature"] = self.config.temperature

        response = await self._chat_completion(task="reply_candidates", route_name="reply", request=request)
        content = response.choices[0].message.content or ""
        return _parse_reply_candidates(
            content,
            max_chars=persona.max_reply_chars,
            fallback_action=normalized_action,
            limit=candidate_count,
        )

    async def daily_review(
        self,
        *,
        persona: Persona,
        messages: list[ChatMessage],
        chat_label: str,
        today_label: str,
        max_chars: int = 520,
    ) -> str:
        context_messages = messages[-140:]
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（今天还没有可复盘的聊天记录）"
        system = self.prompts.render(
            "daily_review",
            "system",
            persona_prompt=persona.prompt,
            max_reply_chars=max_chars,
        )
        user = self.prompts.render(
            "daily_review",
            "user",
            chat_label=chat_label,
            today_label=today_label,
            context=context,
        )
        request = {
            "max_tokens": max(self.config.max_tokens, 700),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.thinking == "enabled":
            request["reasoning_effort"] = self.config.reasoning_effort
        else:
            request["temperature"] = min(0.85, max(0.5, self.config.temperature))

        response = await self._chat_completion(task="daily_review", route_name="reply", request=request)
        content = response.choices[0].message.content or ""
        return _sanitize_reply(content, max_chars)

    async def summarize_mid_memory(
        self,
        *,
        messages: list[ChatMessage],
        chat_label: str = "QQ 群聊",
    ) -> MidMemoryDraft:
        context = "\n".join(_format_message(msg) for msg in messages)
        system = self.prompts.render("mid_memory", "system")
        user = self.prompts.render(
            "mid_memory",
            "user",
            chat_label=chat_label,
            context=context,
        )
        response = await self._chat_completion(
            task="mid_memory",
            route_name="memory",
            request={
                "temperature": 0.2,
                "max_tokens": 360,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return _parse_mid_memory(response.choices[0].message.content or "")

    async def learn_style_rules(
        self,
        *,
        messages: list[ChatMessage],
        chat_label: str = "QQ 群聊",
    ) -> tuple[StyleRuleDraft, ...]:
        context = "\n".join(
            f"[source_id:{index}] {_format_message(msg)}"
            for index, msg in enumerate(messages, start=1)
        )
        system = self.prompts.render("style_learning", "system")
        user = self.prompts.render(
            "style_learning",
            "user",
            chat_label=chat_label,
            context=context,
        )
        response = await self._chat_completion(
            task="style_learning",
            route_name="style",
            request={
                "temperature": 0.2,
                "max_tokens": 420,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return _parse_style_rules(response.choices[0].message.content or "", messages)

def _format_message(msg: ChatMessage) -> str:
    speaker = "机器人" if msg.is_bot else _speaker_label(msg.user_id, msg.nickname)
    return f"{speaker}: {msg.text}"


def _format_decision_message(msg: ChatMessage) -> str:
    if msg.is_bot:
        return f"机器人之前发言（只判断互动状态，禁止复用措辞）: {msg.text}"
    return _format_message(msg)


def _reply_context_messages(
    messages: list[ChatMessage],
    *,
    include_bot_history: bool,
    limit: int = 30,
) -> list[ChatMessage]:
    if include_bot_history:
        return messages[-limit:]

    human_messages = [msg for msg in messages if not msg.is_bot]
    if human_messages:
        return human_messages[-limit:]
    return messages[-min(limit, len(messages)):]


def _optional_section(title: str, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    return f"\n\n{title}：\n{content}"


def _speaker_label(user_id: int, nickname: str) -> str:
    name = nickname.strip() or str(user_id)
    return f"{name}[#{str(user_id)[-5:]}]"


def _parse_reply_decision(content: str) -> ReplyDecision:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ReplyDecision(False, 0.0, "invalid_json")
    should_reply = bool(raw.get("should_reply", False))
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(raw.get("reason", "")).strip()
    mode = str(raw.get("mode", "silent")).strip() or "silent"
    action = _normalize_action(str(raw.get("action", "") or mode), should_reply=should_reply)
    need_tool = bool(raw.get("need_tool", False))
    tool = str(raw.get("tool", "") or "").strip().lower()
    comment_after_tool = bool(raw.get("comment_after_tool", False))
    symbols = _parse_tool_symbols(raw.get("symbols", []))
    need_fresh_context = bool(raw.get("need_fresh_context", False))
    fresh_query = str(raw.get("fresh_query", "") or "").strip()
    fresh_kind = str(raw.get("fresh_kind", "news") or "news").strip().lower()
    if fresh_kind not in {"news", "sports", "web"}:
        fresh_kind = "news"
    if action == "market_check":
        need_tool = True
        tool = "market"
    if action == "fresh_context":
        need_fresh_context = True
    if action == "ignore":
        should_reply = False
    if need_tool and tool == "market":
        action = "market_check"
    if need_fresh_context:
        action = "fresh_context"
    if not should_reply:
        action = "ignore"
    return ReplyDecision(
        should_reply,
        confidence,
        reason,
        mode,
        action,
        need_tool,
        tool,
        symbols,
        comment_after_tool,
        need_fresh_context,
        fresh_query[:120],
        fresh_kind,
    )


def _parse_jargon_terms(content: str) -> tuple[str, ...]:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ()
    raw_terms = raw.get("terms", [])
    if not isinstance(raw_terms, list):
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw_terms:
        term = str(item).strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term[:32])
        if len(terms) >= 8:
            break
    return tuple(terms)


def _loads_json_object(content: str) -> dict[str, object]:
    text = content.strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise json.JSONDecodeError("json root is not object", text, 0)
    return raw


def _normalize_action(value: str, *, should_reply: bool) -> str:
    if not should_reply:
        return "ignore"
    action = value.strip().lower()
    aliases = {
        "": "reply",
        "silent": "ignore",
        "none": "ignore",
        "chat": "reply",
        "natural": "reply",
        "normal": "reply",
        "reply": "reply",
        "answer": "answer",
        "normal_answer": "answer",
        "回答": "answer",
        "正常回答": "answer",
        "agree": "agree",
        "support": "agree",
        "approve": "agree",
        "认可": "agree",
        "同意": "agree",
        "care": "care",
        "comfort": "care",
        "empathy": "care",
        "关心": "care",
        "安慰": "care",
        "承接": "care",
        "market": "market_check",
        "tool": "market_check",
        "search": "fresh_context",
        "fresh": "fresh_context",
        "news": "fresh_context",
        "tease": "tease",
        "mock": "tease",
        "roast": "tease",
        "ask": "ask_back",
        "ask_back": "ask_back",
        "question": "ask_back",
        "mock_repeated_question": "mock_repeated_question",
        "repeat_mock": "mock_repeated_question",
        "at": "at_someone",
        "mention": "at_someone",
        "at_someone": "at_someone",
        "market_check": "market_check",
        "fresh_context": "fresh_context",
        "ignore": "ignore",
    }
    normalized = aliases.get(action, action)
    if normalized not in SOCIAL_ACTIONS:
        return "reply"
    return normalized


def _parse_mid_memory(content: str) -> MidMemoryDraft:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return MidMemoryDraft("", ())
    summary = str(raw.get("summary", "")).strip()
    raw_cues = raw.get("recall_cues", [])
    if not isinstance(raw_cues, list):
        raw_cues = []
    cues = tuple(str(cue).strip() for cue in raw_cues if str(cue).strip())[:5]
    return MidMemoryDraft(summary, cues)


def _parse_member_profile_draft(content: str) -> MemberProfileDraft:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return MemberProfileDraft("", (), "", ())
    summary = str(raw.get("summary", "")).strip()
    speaking_style = str(raw.get("speaking_style", "")).strip()
    interests = _parse_string_list(raw.get("interests", []), limit=8, item_limit=32)
    representative_texts = _parse_string_list(raw.get("representative_texts", []), limit=5, item_limit=140)
    return MemberProfileDraft(
        summary[:420],
        tuple(interests),
        speaking_style[:260],
        tuple(representative_texts),
    )


def _parse_string_list(value: object, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text[:item_limit])
        if len(result) >= limit:
            break
    return result


def _parse_style_rules(
    content: str,
    source_messages: list[ChatMessage],
) -> tuple[StyleRuleDraft, ...]:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return ()
    raw_rules = raw.get("rules", raw if isinstance(raw, list) else [])
    if not isinstance(raw_rules, list):
        return ()

    parsed: list[StyleRuleDraft] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        if not situation or not style:
            continue
        key = (situation, style)
        if key in seen:
            continue
        seen.add(key)
        source_text = _source_text_for_style_rule(item, source_messages)
        parsed.append(
            StyleRuleDraft(
                situation=situation[:60],
                style=style[:80],
                source_text=source_text,
            )
        )
        if len(parsed) >= 8:
            break
    return tuple(parsed)


def _source_text_for_style_rule(
    raw_rule: dict[str, object],
    source_messages: list[ChatMessage],
) -> str:
    raw_source_id = str(raw_rule.get("source_id", "")).strip()
    try:
        source_index = int(raw_source_id) - 1
    except ValueError:
        return ""
    if 0 <= source_index < len(source_messages):
        return source_messages[source_index].text
    return ""


def _log_llm_usage(task: str, response: object, *, model: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return
    logger.info(
        "qq_social_agent llm usage: "
        f"task={task} model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
    )
    if _usage_recorder is not None:
        try:
            _usage_recorder(task, model, prompt_tokens, completion_tokens, total_tokens)
        except Exception as exc:
            logger.warning(f"qq_social_agent failed recording llm usage: task={task} error={exc}")


def _extra_body_for_route(provider: LLMProviderConfig, route: LLMModelRoute) -> dict[str, object]:
    if provider.thinking not in {"enabled", "disabled"}:
        return {}
    model = route.model.casefold()
    if provider.name == "siliconflow" and provider.thinking == "disabled" and model.startswith("qwen/"):
        return {"enable_thinking": False}
    if provider.name != "deepseek":
        return {}
    return {"thinking": {"type": provider.thinking}}


def _usage_value(usage: object, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tool_symbols(raw_symbols: object) -> tuple[ToolSymbol, ...]:
    if not isinstance(raw_symbols, list):
        return ()

    parsed: list[ToolSymbol] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_symbols:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"stock", "crypto"}:
            continue
        symbol = str(item.get("symbol", "")).strip()
        display = str(item.get("display", "") or symbol).strip()
        if not symbol:
            continue
        if kind == "stock":
            symbol = symbol.upper()
        else:
            symbol = symbol.lower()
        key = (kind, symbol)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(ToolSymbol(kind=kind, symbol=symbol, display=display or symbol))
        if len(parsed) >= 2:
            break
    return tuple(parsed)


def _sanitize_reply(content: str, max_chars: int) -> str:
    text = content.strip().strip("\"'")
    marker = re.sub(r"[\s\"'`“”‘’()（）\[\]【】{}<>《》。.!！?？:：;；,，、-]+", "", text)
    if marker in {"", "空字符串", "无", "不回复", "空", "null", "None"}:
        return ""
    if len(text) > max_chars:
        text = _trim_to_sentence(text, max_chars)
    return text


def _parse_reply_candidates(
    content: str,
    *,
    max_chars: int,
    fallback_action: str,
    limit: int,
) -> tuple[ReplyCandidateDraft, ...]:
    dropped_reasons: list[str] = []
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        text = _sanitize_reply(content, max_chars)
        if not text:
            _log_reply_candidate_parse_diagnostic(
                raw_count=0,
                parsed_count=0,
                limit=limit,
                dropped_reasons=("invalid_json_empty",),
            )
            return ()
        _log_reply_candidate_parse_diagnostic(
            raw_count=0,
            parsed_count=1,
            limit=limit,
            dropped_reasons=("non_json_fallback",),
        )
        return (ReplyCandidateDraft(text=text, action=fallback_action, style="模型返回非 JSON，按原回复处理"),)

    raw_candidates = raw.get("candidates", [])
    if not isinstance(raw_candidates, list):
        dropped_reasons.append("candidates_not_list")
        raw_candidates = []
    raw_count = len(raw_candidates)
    parsed: list[ReplyCandidateDraft] = []
    seen_texts: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            dropped_reasons.append("item_not_object")
            continue
        text = _sanitize_reply(str(item.get("text", "") or ""), max_chars)
        if not text:
            dropped_reasons.append("empty_text")
            continue
        compact_text = re.sub(r"\s+", "", text)
        if compact_text in seen_texts:
            dropped_reasons.append("duplicate_text")
            continue
        seen_texts.add(compact_text)
        action = _normalize_action(str(item.get("action", fallback_action) or fallback_action), should_reply=True)
        style = str(item.get("style", "") or "").strip()
        if not style:
            style = "当前语境下的自然接话策略"
        parsed.append(
            ReplyCandidateDraft(
                text=text,
                action=action,
                style=style[:80],
            )
        )
        if len(parsed) >= limit:
            break
    if len(parsed) < limit:
        _log_reply_candidate_parse_diagnostic(
            raw_count=raw_count,
            parsed_count=len(parsed),
            limit=limit,
            dropped_reasons=tuple(dropped_reasons),
        )
    return tuple(parsed)


def _log_reply_candidate_parse_diagnostic(
    *,
    raw_count: int,
    parsed_count: int,
    limit: int,
    dropped_reasons: tuple[str, ...],
) -> None:
    if parsed_count >= limit:
        return
    logger.info(
        "qq_social_agent reply candidates parse diagnostic: "
        f"raw_count={raw_count} parsed_count={parsed_count} "
        f"limit={limit} dropped_reason={_format_drop_reasons(dropped_reasons)}"
    )


def _format_drop_reasons(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "none"
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return ",".join(f"{reason}={count}" for reason, count in sorted(counts.items()))


def _trim_to_sentence(text: str, max_chars: int) -> str:
    clipped = text[:max_chars].rstrip()
    sentence_min = max(6, int(max_chars * 0.35))
    clause_min = max(8, int(max_chars * 0.5))
    last_stop = max(clipped.rfind(mark) for mark in "。！？!?")
    if last_stop >= sentence_min:
        return clipped[: last_stop + 1]
    last_comma = max(clipped.rfind(mark) for mark in "，,；;")
    if last_comma >= clause_min:
        return clipped[:last_comma].rstrip() + "。"
    return clipped.rstrip("，,；;：:、 ") + "。"
