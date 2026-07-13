from __future__ import annotations

import asyncio
import json
import os
import re
import time
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
    reaction: str = ""


@dataclass(frozen=True)
class MemoryFactDraft:
    kind: str
    content: str
    subject_user_id: int | None = None
    object_user_id: int | None = None
    evidence_message_ids: tuple[int, ...] = ()
    confidence: float = 0.7
    importance: float = 0.5
    valid_for_days: int | None = None


@dataclass(frozen=True)
class MidMemoryDraft:
    summary: str
    recall_cues: tuple[str, ...]
    facts: tuple[MemoryFactDraft, ...] = ()
    member_deltas: tuple[MemoryFactDraft, ...] = ()
    jargon_candidates: tuple[MemoryFactDraft, ...] = ()
    open_threads: tuple[MemoryFactDraft, ...] = ()


@dataclass(frozen=True)
class DailyReviewDraft:
    public_reply: str
    events: tuple[MemoryFactDraft, ...] = ()
    member_changes: tuple[MemoryFactDraft, ...] = ()
    jargon_candidates: tuple[MemoryFactDraft, ...] = ()
    feedback_lessons: tuple[MemoryFactDraft, ...] = ()
    style_observations: tuple[MemoryFactDraft, ...] = ()


@dataclass(frozen=True)
class StyleRuleDraft:
    situation: str
    style: str
    source_text: str = ""
    source_user_ids: tuple[int, ...] = ()
    source_message_ids: tuple[int, ...] = ()


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
    "at_someone",
    "observe",
    "echo_mood",
    "shift_topic",
    "self_comment",
    "relationship_reply",
    "market_check",
    "fresh_context",
    "react",
    "poke",
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
                max_retries=config.max_retries,
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
        attempt_timeout, total_timeout = self._task_timeouts(task=task, route_name=route_name)
        deadline = time.monotonic() + total_timeout
        for route in routes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = asyncio.TimeoutError(
                    f"LLM total timeout after {total_timeout:g}s for task={task}"
                )
                break
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
                current_timeout = max(0.25, min(attempt_timeout, remaining))
                operation = client.with_options(
                    timeout=current_timeout,
                    max_retries=0,
                ).chat.completions.create(**provider_request)
                response = await asyncio.wait_for(operation, timeout=current_timeout + 0.25)
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

    def _task_timeouts(self, *, task: str, route_name: str) -> tuple[float, float]:
        if route_name == "decision" or task == "decision":
            attempt = self.config.decision_timeout_seconds
            total = self.config.decision_total_timeout_seconds
        elif task == "daily_review":
            attempt = getattr(self.config, "daily_review_timeout_seconds", 35.0)
            total = getattr(self.config, "daily_review_total_timeout_seconds", 75.0)
        elif route_name == "reply" or task in {"reply", "reply_direct", "reply_candidates"}:
            attempt = self.config.reply_timeout_seconds
            total = self.config.reply_total_timeout_seconds
        elif route_name in {"utility", "jargon", "memory", "style", "member_profile"}:
            attempt = self.config.utility_timeout_seconds
            total = self.config.utility_total_timeout_seconds
        else:
            attempt = float(self.config.timeout_seconds)
            total = float(self.config.timeout_seconds) * max(1, len(self._candidate_routes(route_name)))
        attempt = max(1.0, float(attempt))
        total = max(attempt, float(total))
        return attempt, total

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
        memory_atoms_context: str = "",
        fresh_context_hint: str = "",
    ) -> ReplyDecision:
        context = _format_context_with_local_focus(
            recent_messages[-30:],
            formatter=_format_decision_message,
        )
        if not context:
            context = "（暂无更多上下文）"
        addressed = mentioned or replied_to_bot
        interaction_state = "有人艾特或回复了你：必须回应当前实际问题，不得因为对方重复询问而拒答或只反问。"
        if not addressed:
            interaction_state = "当前没有艾特你，也不是回复你，你是在判断要不要自然插话。"
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
            memory_atoms_context_section=_optional_section("长期记忆单元", memory_atoms_context),
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
        context = _format_context_with_local_focus(context_messages, formatter=_format_message)
        recent_bot_replies = _recent_bot_reply_texts(recent_messages)
        context = _append_recent_bot_duplicate_guard(context, recent_bot_replies)
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
        memory_atoms_context: str = "",
        recall_feedback_context: str = "",
        mention_targets: str = "",
        priority_context: str = "",
        include_bot_history: bool = True,
    ) -> str:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = _format_context_with_local_focus(context_messages, formatter=_format_message)
        if not context:
            context = "（暂无更多上下文）"
        mode = (
            "你被直接点名或回复，必须先回应当前实际问题；即使对方重复问，也不能只吐槽、拒答或反问。"
            if mentioned
            else "你是自然插话，只能在合适时短句接话。"
        )
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
            memory_atoms_context_section=_optional_section("长期记忆单元", memory_atoms_context),
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

    async def summarize_long_message(
        self,
        *,
        text: str,
        speaker_label: str,
        chat_label: str = "QQ 群聊",
        original_chars: int | None = None,
    ) -> str:
        source_chars = original_chars if original_chars is not None else len(text)
        system = self.prompts.render("long_message_summary", "system")
        user = self.prompts.render(
            "long_message_summary",
            "user",
            chat_label=chat_label,
            speaker_label=speaker_label,
            source_chars=source_chars,
            source_text=text,
        )
        response = await self._chat_completion(
            task="long_message_summary",
            route_name="memory",
            request={
                "temperature": 0.1,
                "max_tokens": 180,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return _parse_long_message_summary(response.choices[0].message.content or "")

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
        system = self.prompts.render("member_profile", "system")
        user = self.prompts.render(
            "member_profile",
            "user",
            chat_label=chat_label,
            member_label=member_label,
            context=context,
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
        memory_atoms_context: str = "",
        recall_feedback_context: str = "",
        positive_feedback_context: str = "",
        mention_targets: str = "",
        priority_context: str = "",
        include_bot_history: bool = True,
        candidate_count: int = 3,
        prompt_flow: str = "reply_candidates",
        task_name: str = "reply_candidates",
    ) -> tuple[ReplyCandidateDraft, ...]:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = _format_context_with_local_focus(context_messages, formatter=_format_message)
        recent_bot_replies = _recent_bot_reply_texts(recent_messages)
        context = _append_recent_bot_duplicate_guard(context, recent_bot_replies)
        if not context:
            context = "（暂无更多上下文）"
        mode = (
            "你被直接点名或回复，必须先回应当前实际问题；即使对方重复问，也不能只吐槽、拒答或反问。"
            if mentioned
            else "你是自然插话，只能在合适时短句接话。"
        )
        normalized_action = _normalize_action(action, should_reply=True)
        action_guide = self.prompts.action_guide(
            normalized_action,
            self.prompts.action_guide("reply", "行动：普通接话。结合群友聊天内容接一句话。"),
        )
        system = self.prompts.render(
            prompt_flow,
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
            prompt_flow,
            "user",
            context=context,
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            memory_atoms_context_section=_optional_section("长期记忆单元", memory_atoms_context),
            recall_feedback_context_section=_optional_section("主人撤回/不准奏反馈", recall_feedback_context),
            positive_feedback_context_section=_optional_section("审批人标记过的优质发言方向", positive_feedback_context),
            style_context_section=_optional_section("群聊表达风格参考", style_context),
            raw_corpus_context_section=_optional_section("群友原文语料参考", raw_corpus_context),
            jargon_context_section=_optional_section("群内黑话词典", jargon_context),
            mention_targets_section=_optional_section("可艾特目标", mention_targets),
            priority_context_section=_optional_section("最高优先级语气要求", priority_context),
            market_section=market_section,
            fresh_section=fresh_section,
            current_nickname=current_nickname,
            current_text=current_text,
            candidate_count=candidate_count,
        )
        direct_reply = prompt_flow == "reply_direct" and candidate_count == 1
        request = {
            "max_tokens": max(self.config.max_tokens, 320) if direct_reply else max(self.config.max_tokens, 900),
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

        response = await self._chat_completion(task=task_name, route_name="reply", request=request)
        content = response.choices[0].message.content or ""
        candidates = _parse_reply_candidates(
            content,
            max_chars=persona.max_reply_chars,
            fallback_action=normalized_action,
            limit=candidate_count,
        )
        candidates = _filter_recent_bot_duplicate_candidates(candidates, recent_bot_replies)
        if len(candidates) >= candidate_count:
            return candidates

        retry_request = _reply_candidates_retry_request(
            request,
            previous_content=content,
            parsed_count=len(candidates),
            candidate_count=candidate_count,
            avoid_texts=recent_bot_replies,
        )
        try:
            retry_response = await self._chat_completion(
                task=task_name,
                route_name="reply",
                request=retry_request,
            )
            retry_content = retry_response.choices[0].message.content or ""
            retry_candidates = _parse_reply_candidates(
                retry_content,
                max_chars=persona.max_reply_chars,
                fallback_action=normalized_action,
                limit=candidate_count,
            )
            retry_candidates = _filter_recent_bot_duplicate_candidates(
                retry_candidates,
                recent_bot_replies,
            )
        except Exception as exc:
            logger.warning(f"qq_social_agent reply candidates retry failed: error={exc}")
            retry_candidates = ()
        merged = _merge_reply_candidates(candidates, retry_candidates, limit=candidate_count)
        if len(merged) >= candidate_count:
            return merged
        return _pad_reply_candidates(
            merged,
            fallback_action=normalized_action,
            max_chars=persona.max_reply_chars,
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
        feedback_context: str = "",
    ) -> str:
        draft = await self.daily_review_draft(
            persona=persona,
            messages=messages,
            chat_label=chat_label,
            today_label=today_label,
            max_chars=max_chars,
            feedback_context=feedback_context,
        )
        return draft.public_reply

    async def daily_review_draft(
        self,
        *,
        persona: Persona,
        messages: list[ChatMessage],
        chat_label: str,
        today_label: str,
        max_chars: int = 520,
        feedback_context: str = "",
    ) -> DailyReviewDraft:
        context_messages = messages[-140:]
        context = "\n".join(
            f"[message_id:{msg.id}] {_format_message(msg)}"
            for msg in context_messages
        )
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
            feedback_context=feedback_context.strip() or "（无审批反馈）",
        )
        request = {
            "max_tokens": max(self.config.max_tokens, 1200),
            "response_format": {"type": "json_object"},
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
        return _parse_daily_review(content, messages=context_messages, max_chars=max_chars)

    async def summarize_mid_memory(
        self,
        *,
        messages: list[ChatMessage],
        chat_label: str = "QQ 群聊",
    ) -> MidMemoryDraft:
        context = "\n".join(
            f"[message_id:{msg.id}] {_format_message(msg)}"
            for msg in messages
        )
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
        return _parse_mid_memory(response.choices[0].message.content or "", messages=messages)

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


def _recent_bot_reply_texts(messages: list[ChatMessage], *, limit: int = 4) -> tuple[str, ...]:
    return tuple(msg.text for msg in messages[-16:] if msg.is_bot and msg.text.strip())[-limit:]


def _append_recent_bot_duplicate_guard(context: str, recent_bot_replies: tuple[str, ...]) -> str:
    if not recent_bot_replies:
        return context
    lines = "\n".join(f"- {text}" for text in recent_bot_replies)
    guard = (
        "【机器人刚刚发过的话（只用于查重，禁止复用措辞或核心答案）】\n"
        f"{lines}\n"
        "如果不同群友连续问同一种模板问题，必须按当前这个人分别回答；"
        "不要把刚给别人的人名、结论或包袱机械再给一次。"
    )
    return f"{context}\n\n{guard}" if context else guard


def _format_context_with_local_focus(
    messages: list[ChatMessage],
    *,
    formatter: Callable[[ChatMessage], str],
    local_limit: int = 6,
    topic_gap_seconds: float = 180.0,
) -> str:
    if not messages:
        return ""
    local_start = max(0, len(messages) - max(1, local_limit))
    for index in range(len(messages) - 1, local_start, -1):
        gap = float(messages[index].created_at) - float(messages[index - 1].created_at)
        if gap > topic_gap_seconds:
            local_start = index
            break
    older = messages[:local_start]
    local = messages[local_start:]
    sections: list[str] = []
    if older:
        sections.append("\n".join(formatter(msg) for msg in older))
    sections.append(
        "【紧邻当前消息的连续话题（最高优先级）：解释‘这/那/太可怕了/是吧’等省略表达时，"
        "必须优先承接下面这些消息，禁止跨越话题断点拼接旧词】\n"
        + "\n".join(formatter(msg) for msg in local)
    )
    return "\n\n".join(sections)


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
    reaction = _normalize_reaction_name(str(raw.get("reaction", "") or ""))
    if fresh_kind not in {"news", "sports", "web"}:
        fresh_kind = "news"
    if action == "market_check":
        need_tool = True
        tool = "market"
    if action == "fresh_context":
        need_fresh_context = True
        action = "answer"
    if action == "ignore":
        should_reply = False
    if need_tool and tool == "market":
        action = "market_check"
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
        reaction,
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
        "mock_repeated_question": "reply",
        "repeat_mock": "reply",
        "poke": "poke",
        "戳一戳": "poke",
        "observe": "observe",
        "旁观": "observe",
        "冒泡": "observe",
        "echo_mood": "echo_mood",
        "mood": "echo_mood",
        "情绪承接": "echo_mood",
        "接情绪": "echo_mood",
        "shift_topic": "shift_topic",
        "change_topic": "shift_topic",
        "转话题": "shift_topic",
        "self_comment": "self_comment",
        "自评": "self_comment",
        "自嘲": "self_comment",
        "relationship_reply": "relationship_reply",
        "relation": "relationship_reply",
        "关系回应": "relationship_reply",
        "react": "react",
        "reaction": "react",
        "emoji": "react",
        "emoji_like": "react",
        "表情回应": "react",
        "点表情": "react",
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


def _normalize_reaction_name(value: str) -> str:
    key = value.strip().lower()
    aliases = {
        "": "",
        "thumb": "agree",
        "thumbsup": "agree",
        "thumbs_up": "agree",
        "like": "agree",
        "赞": "agree",
        "hug": "care",
        "抱抱": "care",
        "comfort": "care",
        "哈哈": "laugh",
        "笑": "laugh",
        "laughing": "laugh",
        "bad_laugh": "tease",
        "坏笑": "tease",
        "surprised": "surprise",
        "问号": "question",
        "clap": "applause",
        "鼓掌": "applause",
        "heart": "heart",
        "爱心": "heart",
    }
    normalized = aliases.get(key, key)
    return normalized if normalized in {
        "agree",
        "care",
        "laugh",
        "tease",
        "surprise",
        "question",
        "applause",
        "heart",
    } else ""


def _parse_mid_memory(
    content: str,
    *,
    messages: list[ChatMessage] | None = None,
) -> MidMemoryDraft:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return MidMemoryDraft("", ())
    summary = str(raw.get("summary", "")).strip()
    raw_cues = raw.get("recall_cues", [])
    if not isinstance(raw_cues, list):
        raw_cues = []
    cues = tuple(str(cue).strip()[:100] for cue in raw_cues if str(cue).strip())[:5]
    return MidMemoryDraft(
        summary[:1200],
        cues,
        _parse_memory_fact_list(raw.get("facts"), messages=messages, default_kind="fact", limit=12),
        _parse_memory_fact_list(
            raw.get("member_deltas"), messages=messages, default_kind="member_delta", limit=10
        ),
        _parse_memory_fact_list(
            raw.get("jargon_candidates"), messages=messages, default_kind="jargon_candidate", limit=6
        ),
        _parse_memory_fact_list(
            raw.get("open_threads"), messages=messages, default_kind="open_thread", limit=6
        ),
    )


def _parse_daily_review(
    content: str,
    *,
    messages: list[ChatMessage] | None = None,
    max_chars: int = 520,
) -> DailyReviewDraft:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        recovered = _recover_json_string_field(content, "public_reply")
        if recovered:
            return DailyReviewDraft(_sanitize_reply(recovered, max_chars))
        if content.lstrip().startswith("{"):
            return DailyReviewDraft("")
        return DailyReviewDraft(_sanitize_reply(content, max_chars))
    public_reply = _sanitize_reply(str(raw.get("public_reply", "")), max_chars)
    if not public_reply:
        public_reply = _sanitize_reply(str(raw.get("reply", "")), max_chars)
    return DailyReviewDraft(
        public_reply=public_reply,
        events=_parse_memory_fact_list(raw.get("events"), messages=messages, default_kind="event", limit=12),
        member_changes=_parse_memory_fact_list(
            raw.get("member_changes"), messages=messages, default_kind="member_delta", limit=10
        ),
        jargon_candidates=_parse_memory_fact_list(
            raw.get("jargon_candidates"), messages=messages, default_kind="jargon_candidate", limit=6
        ),
        feedback_lessons=_parse_memory_fact_list(
            raw.get("feedback_lessons"), messages=messages, default_kind="feedback_lesson", limit=8
        ),
        style_observations=_parse_memory_fact_list(
            raw.get("style_observations"), messages=messages, default_kind="style_observation", limit=8
        ),
    )


def _recover_json_string_field(content: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', content)
    if match is None:
        return ""
    escaped = False
    raw_value: list[str] = []
    for char in content[match.end() :]:
        if escaped:
            raw_value.extend(("\\", char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            break
        raw_value.append(char)
    encoded = "".join(raw_value)
    try:
        return str(json.loads(f'"{encoded}"')).strip()
    except json.JSONDecodeError:
        return encoded.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").strip()


def _parse_memory_fact_list(
    value: object,
    *,
    messages: list[ChatMessage] | None,
    default_kind: str,
    limit: int,
) -> tuple[MemoryFactDraft, ...]:
    if not isinstance(value, list):
        return ()
    by_message_id = {message.id: message for message in (messages or []) if message.id > 0}
    valid_ids = set(by_message_id)
    facts: list[MemoryFactDraft] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()[:320]
        if not content:
            continue
        kind = re.sub(r"[^a-z0-9_-]+", "_", str(item.get("kind", default_kind)).strip().lower())[:32]
        kind = kind or default_kind
        evidence_ids = _parse_positive_ints(item.get("source_message_ids"), limit=8)
        if valid_ids:
            evidence_ids = tuple(message_id for message_id in evidence_ids if message_id in valid_ids)
        subject_message_id = _optional_positive_int(item.get("subject_message_id"))
        object_message_id = _optional_positive_int(item.get("object_message_id"))
        subject_user_id = by_message_id.get(subject_message_id).user_id if subject_message_id in by_message_id else None
        object_user_id = by_message_id.get(object_message_id).user_id if object_message_id in by_message_id else None
        if subject_message_id in valid_ids and subject_message_id not in evidence_ids:
            evidence_ids = (subject_message_id, *evidence_ids)[:8]
        confidence = _bounded_float(item.get("confidence"), default=0.7)
        importance = _bounded_float(item.get("importance"), default=0.5)
        valid_for_days = _optional_positive_int(item.get("valid_for_days"))
        if valid_for_days is not None:
            valid_for_days = min(valid_for_days, 3650)
        key = (kind, content.casefold(), subject_user_id, object_user_id)
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            MemoryFactDraft(
                kind=kind,
                content=content,
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                evidence_message_ids=evidence_ids,
                confidence=confidence,
                importance=importance,
                valid_for_days=valid_for_days,
            )
        )
        if len(facts) >= limit:
            break
    return tuple(facts)


def _parse_positive_ints(value: object, *, limit: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    result: list[int] = []
    for raw in value:
        parsed = _optional_positive_int(raw)
        if parsed is None or parsed in result:
            continue
        result.append(parsed)
        if len(result) >= limit:
            break
    return tuple(result)


def _optional_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _bounded_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0.0, min(1.0, parsed))


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


def _parse_long_message_summary(content: str) -> str:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ""
    summary = re.sub(r"\s+", " ", str(raw.get("summary", ""))).strip()
    return summary[:180]


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
        evidence_messages = _source_messages_for_style_rule(item, source_messages)
        source_text = evidence_messages[0].text if evidence_messages else ""
        parsed.append(
            StyleRuleDraft(
                situation=situation[:60],
                style=style[:80],
                source_text=source_text,
                source_user_ids=tuple(dict.fromkeys(msg.user_id for msg in evidence_messages)),
                source_message_ids=tuple(dict.fromkeys(msg.id for msg in evidence_messages if msg.id)),
            )
        )
        if len(parsed) >= 8:
            break
    return tuple(parsed)


def _source_messages_for_style_rule(
    raw_rule: dict[str, object],
    source_messages: list[ChatMessage],
) -> list[ChatMessage]:
    raw_ids = raw_rule.get("support_source_ids", raw_rule.get("source_ids", []))
    if not raw_ids:
        raw_ids = [raw_rule.get("source_id", "")]
    if not isinstance(raw_ids, list):
        raw_ids = [raw_rule.get("source_id", "")]
    result: list[ChatMessage] = []
    for raw_source_id in raw_ids[:8]:
        try:
            source_index = int(str(raw_source_id).strip()) - 1
        except ValueError:
            continue
        if 0 <= source_index < len(source_messages):
            result.append(source_messages[source_index])
    return result


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


def _reply_candidates_retry_request(
    request: dict[str, object],
    *,
    previous_content: str,
    parsed_count: int,
    candidate_count: int,
    avoid_texts: tuple[str, ...] = (),
) -> dict[str, object]:
    messages = list(request.get("messages", []))
    avoid_instruction = ""
    if avoid_texts:
        avoid_instruction = (
            "另外，机器人刚才已经对别人说过以下内容，本轮禁止复用其措辞或核心答案："
            + "；".join(avoid_texts)
            + "。"
        )
    messages.extend(
        [
            {"role": "assistant", "content": previous_content},
            {
                "role": "user",
                "content": (
                    f"上一轮只成功解析出 {parsed_count} 条候选，但必须给满 {candidate_count} 条。"
                    "请重新输出一个完整 JSON 对象，格式严格为 "
                    '{"candidates":[{"text":"...","style":"...","action":"reply"}]}。'
                    f"candidates 必须正好 {candidate_count} 条，text 不能空，三条不能重复，"
                    f"不要输出 JSON 以外的任何文字。{avoid_instruction}"
                ),
            },
        ]
    )
    retry_request = dict(request)
    retry_request["messages"] = messages
    retry_request["max_tokens"] = max(int(request.get("max_tokens", 0) or 0), 900)
    retry_request["response_format"] = {"type": "json_object"}
    return retry_request


def _filter_recent_bot_duplicate_candidates(
    candidates: tuple[ReplyCandidateDraft, ...],
    recent_bot_replies: tuple[str, ...],
) -> tuple[ReplyCandidateDraft, ...]:
    if not recent_bot_replies:
        return candidates
    return tuple(
        candidate
        for candidate in candidates
        if not any(_substantially_repeats(candidate.text, previous) for previous in recent_bot_replies)
    )


def _substantially_repeats(current: str, previous: str, *, min_common: int = 8) -> bool:
    clean = lambda value: re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())
    left = clean(current)
    right = clean(previous)
    if not left or not right:
        return False
    if left in right or right in left:
        return min(len(left), len(right)) >= min_common
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    for size in range(len(shorter), min_common - 1, -1):
        if any(shorter[start : start + size] in longer for start in range(len(shorter) - size + 1)):
            return True
    return False


def _merge_reply_candidates(
    first: tuple[ReplyCandidateDraft, ...],
    second: tuple[ReplyCandidateDraft, ...],
    *,
    limit: int,
) -> tuple[ReplyCandidateDraft, ...]:
    merged: list[ReplyCandidateDraft] = []
    seen: set[str] = set()
    for candidate in (*first, *second):
        compact_text = re.sub(r"\s+", "", candidate.text)
        if not compact_text or compact_text in seen:
            continue
        seen.add(compact_text)
        merged.append(candidate)
        if len(merged) >= limit:
            break
    return tuple(merged)


def _pad_reply_candidates(
    candidates: tuple[ReplyCandidateDraft, ...],
    *,
    fallback_action: str,
    max_chars: int,
    limit: int,
) -> tuple[ReplyCandidateDraft, ...]:
    padded = list(candidates)
    seen = {re.sub(r"\s+", "", candidate.text) for candidate in padded}
    fallback_texts = _fallback_candidate_texts(fallback_action)
    for text in fallback_texts:
        clean_text = _sanitize_reply(text, max_chars)
        compact_text = re.sub(r"\s+", "", clean_text)
        if not clean_text or compact_text in seen:
            continue
        seen.add(compact_text)
        padded.append(
            ReplyCandidateDraft(
                text=clean_text,
                action=fallback_action,
                style="后端补齐：模型候选不足时的保守备选",
            )
        )
        if len(padded) >= limit:
            break
    return tuple(padded[:limit])


def _fallback_candidate_texts(action: str) -> tuple[str, ...]:
    if action == "care":
        return (
            "风雪觉得这事先别急，慢慢捋清楚比较好。",
            "先缓一下，别把自己逼太紧。",
            "这个先别硬扛，能少受点罪就少受点。",
        )
    if action == "agree":
        return (
            "风雪觉得这个说法有点道理。",
            "这句方向没跑偏，至少抓到重点了。",
            "这个判断还行，不算乱说。",
        )
    if action == "answer":
        return (
            "风雪觉得先按这个方向看，别把关键点漏了。",
            "简单说，这事要看成本和后果。",
            "先别绕，核心就是值不值。",
        )
    if action == "tease":
        return (
            "风雪觉得这事有点抽象。",
            "这也太会给自己加戏了。",
            "先别急着上强度，路都快走歪了。",
        )
    if action == "ask_back":
        return (
            "风雪有点好奇，你这是认真问还是在钓我？",
            "那你自己先说，你到底想听哪种答案？",
            "你这句重点是问结果，还是问态度？",
        )
    return (
        "风雪觉得这句可以先轻轻放着。",
        "那先看他后面怎么说。",
        "这句接一下可以，但别聊太满。",
    )


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
