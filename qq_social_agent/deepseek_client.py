from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from nonebot import logger

from .config import LLMConfig
from .jev_client import JevClient
from .llm_gateway import LLMGateway, _log_llm_usage, _usage_value, set_usage_recorder
from .memory import ChatMessage
from .persona import Persona
from .pipeline_types import ContextPacket
from .prompts import PromptRegistry
from .timing_gate import TimingDecision, parse_timing_decision

if TYPE_CHECKING:
    from .discourse_state import DiscourseState


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
    side_reaction: str = ""


@dataclass(frozen=True)
class FreshSearchDecision:
    need_search: bool
    query: str = ""
    kind: str = "web"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ToolRoutingDecision:
    tool: str = "none"
    query: str = ""
    queries: tuple[str, ...] = ()
    kind: str = "web"
    symbols: tuple[ToolSymbol, ...] = ()
    confidence: float = 0.0
    reason: str = ""
    comment_after_tool: bool = True


@dataclass(frozen=True)
class MemeSelectionDecision:
    send: bool = False
    meme_id: int | None = None
    reason: str = ""


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


SPEAKING_ACTIONS = {
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
    "clarify",
    "warm_tease",
    "deflate",
    "take_side",
    "share_self",
    "comfort_joke",
    "mirror_style",
    "amp_bit",
    "deadpan_echo",
    "commit_bit",
    "hyperbole",
    "wrong_register",
    "protect",
}

SOCIAL_ACTIONS = SPEAKING_ACTIONS | {
    "ignore",
    "market_check",
    "fresh_context",
    "react",
    "poke",
}

ADDRESSED_QUESTION_ACTIONS = {"answer", "ask_back", "clarify", "care"}


class LLMTaskClient(LLMGateway):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.prompts = PromptRegistry()
        self.jev_client = JevClient()

    async def aclose(self) -> None:
        await asyncio.gather(super().aclose(), self.jev_client.aclose(), return_exceptions=True)

    def _jev_timeout(self, seconds: float = 2.5) -> float:
        return max(1.0, min(4.0, float(seconds)))

    async def _try_jev(self, factory, *, what: str, timeout: float | None = 2.5):
        jev = getattr(self, "jev_client", None)
        if jev is None or not getattr(jev, "available", False):
            return None
        try:
            # Draft review owns its deadline so completed branch results survive.
            if timeout is None:
                return await factory()
            return await asyncio.wait_for(factory(), timeout=self._jev_timeout(timeout))
        except Exception as exc:
            recorder = getattr(jev, "record_telemetry", None)
            if callable(recorder):
                recorder({
                    "provider": getattr(jev, "provider", "unknown"),
                    "model": getattr(jev, "model", ""),
                    "status": "fallback",
                    "operation": what,
                    "error_type": type(exc).__name__,
                })
            error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            logger.warning(f"qq_social_agent jev {what} failed, falling back: error={error}")
            return None



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
        speaker_context: str = "",
    ) -> ReplyDecision:
        context = _format_context_with_local_focus(
            recent_messages[-30:],
            formatter=_format_decision_message,
        )
        if not context:
            context = "（暂无更多上下文）"
        # Jev is a decisions API, not a chat-completions model. timing_gate
        # still uses the configured decision_model; only should_reply hops here.
        jev_decision = await self._try_jev(
            lambda: self.jev_client.should_reply(
                persona=persona,
                recent_messages=recent_messages,
                current_text=current_text,
                current_nickname=current_nickname,
                mentioned=mentioned,
                replied_to_bot=replied_to_bot,
            ),
            what="decision",
        )
        if jev_decision is not None:
            return jev_decision

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
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
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

    async def route_fresh_search(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        addressed: bool,
        decision_action: str,
        decision_reason: str,
        backend_hint: str = "",
        chat_label: str = "QQ 群聊",
        speaker_context: str = "",
    ) -> FreshSearchDecision:
        context = _format_context_with_local_focus(
            recent_messages[-10:],
            formatter=_format_decision_message,
            local_limit=5,
        ) or "（暂无更多上下文）"
        system = self.prompts.render(
            "fresh_router",
            "system",
            persona_name=persona.name,
            persona_decision_prompt=persona.decision_prompt,
        )
        user = self.prompts.render(
            "fresh_router",
            "user",
            chat_label=chat_label,
            addressed="true" if addressed else "false",
            decision_action=decision_action,
            decision_reason=decision_reason,
            backend_hint=backend_hint or "无",
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
            context=context,
            current_nickname=current_nickname,
            current_text=current_text,
        )
        response = await self._chat_completion(
            task="fresh_router",
            route_name="search",
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
        return _parse_fresh_search_decision(response.choices[0].message.content or "")

    async def route_tool_use(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        addressed: bool,
        decision_action: str,
        decision_reason: str,
        backend_hint: str = "",
        chat_label: str = "QQ 群聊",
        speaker_context: str = "",
    ) -> ToolRoutingDecision:
        context = _format_context_with_local_focus(
            recent_messages[-10:],
            formatter=_format_decision_message,
            local_limit=5,
        ) or "（暂无更多上下文）"
        system = self.prompts.render(
            "tool_router",
            "system",
            persona_name=persona.name,
            persona_decision_prompt=persona.decision_prompt,
        )
        jev_routed = await self._try_jev(
            lambda: self.jev_client.route_tool(
                persona=persona,
                recent_messages=recent_messages,
                current_text=current_text,
                current_nickname=current_nickname,
                addressed=addressed,
                speaker_context=speaker_context,
            ),
            what="tool_router",
        )
        if jev_routed is not None and jev_routed.tool in {"none", "probability"}:
            # ProbabilityTool already refines its own event; do not refine twice.
            return jev_routed
        if jev_routed is not None and jev_routed.tool == "deep_url":
            urls = re.findall(r"https?://[^\s<>\"']+", current_text)
            if len(urls) == 1:
                return replace(jev_routed, query=urls[0])
        from .tool_observation import tool_routing_questions
        system += "\n实际工具能力与边界：" + json.dumps(
            tool_routing_questions()["tool_choice"]["criteria"], ensure_ascii=False,
        )
        if jev_routed is not None:
            system += (
                f"\n本轮工具已确定为 {jev_routed.tool}，只补全该工具的 query/kind/symbols 参数，不切换工具。"
                "结合当前句和已解析上下文补全对象，不把‘那个/刚才那个’原样当搜索词。"
                "若上下文无法确定对象，返回 tool=none，不猜查询主题。"
            )
        user = self.prompts.render(
            "tool_router",
            "user",
            chat_label=chat_label,
            addressed="true" if addressed else "false",
            decision_action=decision_action,
            decision_reason=decision_reason,
            backend_hint=backend_hint or "无",
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
            context=context,
            current_nickname=current_nickname,
            current_text=current_text,
        )
        response = await self._chat_completion(
            task="tool_router",
            route_name="search",
            request={
                "temperature": 0.1,
                "max_tokens": 360,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        result = _parse_tool_routing_decision(response.choices[0].message.content or "")
        if jev_routed is not None and result.tool not in {"none", jev_routed.tool}:
            return ToolRoutingDecision(tool="none", reason="tool_parameters_mismatch")
        return result

    async def timing_gate(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        chat_label: str = "QQ 群聊",
        speaker_context: str = "",
        discourse_state: DiscourseState | None = None,
        followup_addressed: bool = False,
    ) -> TimingDecision:
        """Decide only whether/how to surface; tools and memory route elsewhere."""

        context = _format_context_with_local_focus(
            recent_messages[-14:],
            formatter=_format_decision_message,
        ) or "（暂无更多上下文）"
        system = self.prompts.render(
            "timing_gate",
            "system",
            persona_name=persona.name,
            persona_decision_prompt=persona.decision_prompt,
        )
        user = self.prompts.render(
            "timing_gate",
            "user",
            chat_label=chat_label,
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
            context=context,
            current_nickname=current_nickname,
            current_text=current_text,
        )
        jev_timing = await self._try_jev(
            lambda: self.jev_client.timing_gate(
                persona=persona,
                recent_messages=recent_messages,
                current_text=current_text,
                current_nickname=current_nickname,
                speaker_context=speaker_context,
                chat_label=chat_label,
                discourse_state=discourse_state,
                followup_addressed=followup_addressed,
            ),
            what="timing_gate",
        )
        if jev_timing is not None:
            return jev_timing
        response = await self._chat_completion(
            task="decision",
            route_name="decision",
            request={
                "temperature": 0.15,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        content = response.choices[0].message.content or ""
        return parse_timing_decision(_loads_json_object(content))

    async def audit_proactive_reply(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        candidate: str,
        chat_label: str = "QQ 私聊",
        addressed: bool = False,
        current_text: str = "",
    ) -> tuple[bool, str]:
        """Block only confident Jev self-repetition; unavailable audits pass."""

        jev_audit = await self._try_jev(
            lambda: self.jev_client.audit_proactive_reply(
                persona=persona,
                recent_messages=recent_messages,
                candidate=candidate,
                chat_label=chat_label,
                addressed=addressed,
                current_text=current_text,
            ),
            what="proactive_audit",
        )
        if jev_audit is not None:
            return jev_audit
        return True, "jev_unavailable_audit_pass"

    async def should_continue_private_chat(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
    ) -> tuple[bool, str]:
        jev_result = await self._try_jev(
            lambda: self.jev_client.should_continue_private_chat(
                persona=persona,
                recent_messages=recent_messages,
            ),
            what="private_continue",
        )
        if jev_result is not None:
            return jev_result
        return True, "jev_unavailable_continue"

    async def resolve_referent(
        self,
        *,
        current_text: str,
        current_label: str,
        candidates: list | None = None,
        reply=None,
        rule_guess=None,
        state: str | None = None,
        criteria: dict | None = None,
    ):
        return await self._try_jev(
            lambda: self.jev_client.resolve_referent(
                current_text=current_text,
                current_label=current_label,
                candidates=candidates,
                reply=reply,
                rule_guess=rule_guess,
                state=state,
                criteria=criteria,
            ),
            what="referent",
        )

    async def resolve_ellipsis(
        self,
        *,
        current_text: str,
        current_label: str,
        sources: list | None = None,
        reply=None,
        reference=None,
        state: str | None = None,
        criteria: dict | None = None,
    ):
        return await self._try_jev(
            lambda: self.jev_client.resolve_ellipsis(
                current_text=current_text,
                current_label=current_label,
                sources=sources,
                reply=reply,
                reference=reference,
                state=state,
                criteria=criteria,
            ),
            what="ellipsis",
        )

    async def resolve_addressee(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.resolve_addressee(**kwargs),
            what="addressee",
        )

    async def resolve_discourse_first_pass(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.resolve_discourse_first_pass(**kwargs),
            what="discourse_first_pass",
            timeout=4.0,
        )

    async def audit_discourse_state(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.audit_discourse_state(**kwargs),
            what="discourse_audit",
        )

    async def resolve_repair(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.resolve_repair(**kwargs),
            what="repair",
        )

    async def resolve_memory_effect(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.resolve_memory_effect(**kwargs),
            what="memory_effect",
        )

    async def resolve_ambiguity(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.resolve_ambiguity(**kwargs),
            what="ambiguity",
        )

    async def choose_proactive_topic(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.choose_proactive_topic(**kwargs),
            what="proactive_topic",
        )

    async def political_mask_keys(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.political_mask_keys(**kwargs),
            what="political_mask",
        )

    async def review_draft(self, **kwargs):
        result = await self._try_jev(
            lambda: self.jev_client.review_draft(**kwargs),
            what="draft_review",
            timeout=None,
        )
        return result if result is not None else (None, None)

    async def check_pronoun(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.check_pronoun(**kwargs),
            what="pronoun",
            timeout=4.0,
        )

    async def critique_draft(self, **kwargs):
        return await self._try_jev(
            lambda: self.jev_client.critique_draft(**kwargs),
            what="pre_send_critic",
        )

    async def should_ask_back(
        self,
        *,
        current_text: str,
        action: str,
        addressed: bool,
    ) -> bool | None:
        return await self._try_jev(
            lambda: self.jev_client.should_ask_back(
                current_text=current_text,
                action=action,
                addressed=addressed,
            ),
            what="ask_back",
        )

    async def should_read_media(
        self,
        *,
        kind: str,
        caption: str,
        addressed: bool,
        item_count: int,
    ) -> bool | None:
        return await self._try_jev(
            lambda: self.jev_client.should_read_media(
                kind=kind,
                caption=caption,
                addressed=addressed,
                item_count=item_count,
            ),
            what="media_gate",
            timeout=2.0,
        )

    async def select_speaking_action(
        self,
        *,
        current_text: str,
        current_label: str,
        addressed: bool,
        baseline_action: str,
        speaker_context: str = "",
        recent_messages: list | None = None,
    ) -> tuple[str, str] | None:
        return await self._try_jev(
            lambda: self.jev_client.select_speaking_action(
                current_text=current_text,
                current_label=current_label,
                addressed=addressed,
                baseline_action=baseline_action,
                speaker_context=speaker_context,
                recent_messages=recent_messages,
            ),
            what="speaking_action",
        )

    async def _select_relevant_generation_context(
        self,
        messages: list[ChatMessage],
        *,
        current_text: str,
        current_nickname: str,
        keep_recent: int = 6,
        target_total: int = 12,
    ) -> list[ChatMessage]:
        if len(messages) <= target_total:
            return list(messages)
        keep_recent = 6
        older = list(messages[:-keep_recent])
        scores = await self._try_jev(
            lambda: self.jev_client.rank_context_messages(
                current_nickname=current_nickname,
                current_text=current_text,
                candidates=older,
            ),
            what="context_rank",
            timeout=4.0,
        )
        return select_relevant_context_messages(
            messages,
            scores if isinstance(scores, list) else None,
            keep_recent=keep_recent,
            target_total=target_total,
        )

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
        jev_terms = await self._try_jev(
            lambda: self.jev_client.select_jargon_terms(
                current_text=current_text,
                heuristic_terms=heuristic_terms,
                jargon_catalog=jargon_catalog,
            ),
            what="jargon_select",
        )
        if jev_terms is not None:
            return tuple(jev_terms)
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
        speaker_context: str = "",
    ) -> str:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context_messages = await self._select_relevant_generation_context(
            context_messages,
            current_text=current_text,
            current_nickname=current_nickname,
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
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            memory_atoms_context_section=_optional_section("长期记忆单元", memory_atoms_context),
            recall_feedback_context_section=_optional_section("主人撤回反馈", recall_feedback_context),
            social_action_context_section="",
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

    async def select_private_meme(
        self,
        *,
        current_text: str,
        reply_text: str,
        candidates: str,
    ) -> MemeSelectionDecision:
        system = self.prompts.render("private_meme_selector", "system")
        user = self.prompts.render(
            "private_meme_selector",
            "user",
            current_text=current_text,
            reply_text=reply_text,
            candidates=candidates,
        )
        jev_choice = await self._try_jev(
            lambda: self.jev_client.select_meme(
                current_text=current_text,
                reply_text=reply_text,
                candidates=candidates,
            ),
            what="meme_selector",
        )
        if jev_choice is not None:
            return jev_choice
        response = await self._chat_completion(
            task="private_meme_selector",
            route_name="utility",
            request={
                "temperature": 0.15,
                "max_tokens": 120,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        raw = _loads_json_object(response.choices[0].message.content or "")
        try:
            meme_id = int(raw.get("meme_id")) if raw.get("meme_id") is not None else None
        except (TypeError, ValueError):
            meme_id = None
        return MemeSelectionDecision(
            send=bool(raw.get("send", False)) and meme_id is not None,
            meme_id=meme_id,
            reason=str(raw.get("reason", "") or "").strip()[:40],
        )

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
        previous_summary: str = "",
    ) -> MemberProfileDraft:
        context = "\n".join(_format_learning_source_message(msg) for msg in messages)
        if not context:
            return MemberProfileDraft("", (), "", ())
        previous = previous_summary.strip()
        if previous:
            context = f"已有画像：{previous}\n\n新增发言：\n{context}"
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
                "max_tokens": 220,
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
        social_action_context: str = "",
        mention_targets: str = "",
        priority_context: str = "",
        include_bot_history: bool = True,
        context_message_limit: int | None = None,
        candidate_count: int = 3,
        prompt_flow: str = "reply_candidates",
        task_name: str = "reply_candidates",
        context_packet: ContextPacket | None = None,
        speaker_context: str = "",
    ) -> tuple[ReplyCandidateDraft, ...]:
        if context_packet is not None:
            memory_context = context_packet.get("memory")
            style_context = context_packet.get("style")
            raw_corpus_context = context_packet.get("raw_corpus")
            jargon_context = context_packet.get("jargon")
            member_context = context_packet.get("member")
            memory_atoms_context = context_packet.get("memory_atoms")
            recall_feedback_context = context_packet.get("recall_feedback")
            positive_feedback_context = context_packet.get("positive_feedback")
            social_action_context = context_packet.get("social_actions")
        selected_recent = (
            recent_messages[-max(1, context_message_limit) :]
            if context_message_limit is not None
            else recent_messages
        )
        context_messages = _reply_context_messages(
            selected_recent,
            include_bot_history=include_bot_history,
        )
        if context_message_limit is None or int(context_message_limit) >= 20:
            context_messages = await self._select_relevant_generation_context(
                context_messages,
                current_text=current_text,
                current_nickname=current_nickname,
            )
        context = _format_context_with_local_focus(context_messages, formatter=_format_message)
        search_reply = prompt_flow == "search_answer" and candidate_count == 1
        direct_reply = prompt_flow == "reply_direct" and candidate_count == 1
        single_reply = candidate_count == 1
        recent_bot_replies = () if search_reply else _recent_bot_reply_texts(recent_messages)
        if not search_reply and not single_reply:
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
            speaker_context_section=_optional_section("本轮说话关系", speaker_context),
            memory_context_section=_optional_section("中期聊天回想", memory_context),
            member_context_section=_optional_section("当前相关群友", member_context),
            memory_atoms_context_section=_optional_section("长期记忆单元", memory_atoms_context),
            recall_feedback_context_section=_optional_section("主人撤回/不准奏反馈", recall_feedback_context),
            positive_feedback_context_section=_optional_section("审批人标记过的优质发言方向", positive_feedback_context),
            social_action_context_section=_optional_section("最近表情动作", social_action_context),
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
        request = {
            "max_tokens": (
                180
                if search_reply
                else max(self.config.max_tokens, 320)
                if direct_reply
                else max(self.config.max_tokens, 900)
            ),
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

        generation_route = "search" if search_reply else "reply"
        response = await self._chat_completion(
            task=task_name,
            route_name=generation_route,
            request=request,
        )
        content = response.choices[0].message.content or ""
        parsed_candidates = _parse_reply_candidates(
            content,
            max_chars=persona.max_reply_chars,
            fallback_action=normalized_action,
            limit=candidate_count,
        )
        candidates = _filter_recent_bot_duplicate_candidates(parsed_candidates, recent_bot_replies)
        if single_reply and not candidates and parsed_candidates:
            return parsed_candidates
        if len(candidates) >= candidate_count:
            return candidates

        retry_request = _reply_candidates_retry_request(
            request,
            previous_content=content,
            parsed_count=len(candidates),
            candidate_count=candidate_count,
            avoid_texts=() if single_reply else recent_bot_replies,
        )
        try:
            retry_response = await self._chat_completion(
                task=task_name,
                route_name=generation_route,
                request=retry_request,
            )
            retry_content = retry_response.choices[0].message.content or ""
            retry_parsed = _parse_reply_candidates(
                retry_content,
                max_chars=persona.max_reply_chars,
                fallback_action=normalized_action,
                limit=candidate_count,
            )
            retry_candidates = _filter_recent_bot_duplicate_candidates(
                retry_parsed,
                () if single_reply else recent_bot_replies,
            )
            if single_reply and not retry_candidates and retry_parsed:
                retry_candidates = retry_parsed
        except Exception as exc:
            logger.warning(f"qq_social_agent reply candidates retry failed: error={exc}")
            retry_candidates = ()
        merged = _merge_reply_candidates(candidates, retry_candidates, limit=candidate_count)
        if single_reply and not merged and parsed_candidates:
            return parsed_candidates
        return merged

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
            _format_learning_source_message(msg)
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
            _format_learning_source_message(msg)
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
                "max_tokens": 900,
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
            _format_style_source_message(index, msg)
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
    speaker = "风雪" if msg.is_bot else _speaker_label(msg.user_id, msg.nickname)
    return f"{speaker}: {msg.text}"


def _format_learning_source_message(msg: ChatMessage) -> str:
    role = "self" if msg.is_bot else "member"
    message_id = msg.id if msg.id > 0 else "unknown"
    speaker = "风雪" if msg.is_bot else _speaker_label(msg.user_id, msg.nickname)
    return f"[message_id:{message_id}][role:{role}] {speaker}: {msg.text}"


def _format_style_source_message(index: int, msg: ChatMessage) -> str:
    role = "self" if msg.is_bot else "member"
    message_id = msg.id if msg.id > 0 else "unknown"
    speaker = "风雪" if msg.is_bot else _speaker_label(msg.user_id, msg.nickname)
    return f"[source_id:{index}][message_id:{message_id}][role:{role}] {speaker}: {msg.text}"


def _format_decision_message(msg: ChatMessage) -> str:
    if msg.is_bot:
        return f"风雪之前发言（只判断互动状态，禁止复用措辞）: {msg.text}"
    return _format_message(msg)


def select_relevant_context_messages(
    messages: list[ChatMessage],
    scores_for_older: list[float] | None,
    *,
    keep_recent: int = 6,
    target_total: int = 12,
) -> list[ChatMessage]:
    """Pin the newest 6 lines; only older lines may be dropped or ranked."""
    if not messages:
        return []
    keep_recent = 6 if len(messages) >= 6 else len(messages)
    target_total = max(keep_recent, int(target_total))
    if len(messages) <= target_total:
        return list(messages)
    recent = list(messages[-keep_recent:])
    older = list(messages[:-keep_recent])
    slots = max(0, target_total - len(recent))
    if not older or slots <= 0:
        return recent
    if scores_for_older is None or len(scores_for_older) != len(older):
        return list(messages[-target_total:])
    ranked = sorted(
        range(len(older)),
        key=lambda index: (-float(scores_for_older[index]), -index),
    )
    chosen = {index for index in ranked[:slots] if float(scores_for_older[index]) >= 0.28}
    if len(chosen) < min(2, slots):
        return list(messages[-target_total:])
    selected = [msg for index, msg in enumerate(older) if index in chosen] + recent
    return selected


def _reply_context_messages(
    messages: list[ChatMessage],
    *,
    include_bot_history: bool,
    limit: int = 40,
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
        "【风雪刚刚发过的话（只用于查重，禁止复用措辞或核心答案）】\n"
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
        older_block = "\n".join(formatter(msg) for msg in older)
        sections.append(f"<older_messages>\n{older_block}\n</older_messages>")
    local_block = (
        "【紧邻当前消息的连续话题（最高优先级）：解释‘这/那/太可怕了/是吧’等省略表达时，"
        "必须优先承接下面这些消息，禁止跨越话题断点拼接旧词】\n"
        + "\n".join(formatter(msg) for msg in local)
    )
    sections.append(f"<current_topic>\n{local_block}\n</current_topic>")
    return "\n\n".join(sections)


_CONTEXT_SECTION_TAGS = {
    "本轮说话关系": "speaker_relation",
    "中期聊天回想": "mid_memory",
    "当前相关群友": "member_profiles",
    "长期记忆单元": "memory_atoms",
    "主人撤回反馈": "owner_recall_feedback",
    "主人撤回/不准奏反馈": "owner_recall_feedback",
    "审批人标记过的优质发言方向": "approved_style",
    "最近表情动作": "recent_reactions",
    "群聊表达风格参考": "style_examples",
    "群友原文语料参考": "raw_corpus",
    "群内黑话词典": "group_jargon",
    "可艾特目标": "mention_targets",
    "最高优先级语气要求": "priority_tone",
    "私聊优先级": "private_priority",
    "后端最新背景候选": "fresh_hint",
}


def _optional_section(title: str, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    tag = _CONTEXT_SECTION_TAGS.get(title, "context_section")
    return f'\n\n<{tag} title="{title}">\n{title}：\n{content}\n</{tag}>'


def _speaker_label(user_id: int, nickname: str) -> str:
    name = nickname.strip() or str(user_id)
    return f"{name}[#{str(user_id)[-5:]}]"


def _parse_fresh_search_decision(content: str) -> FreshSearchDecision:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return FreshSearchDecision(False, reason="invalid_json")
    need_search = bool(raw.get("need_search", raw.get("need_fresh_context", False)))
    query = re.sub(r"\s+", " ", str(raw.get("query", raw.get("fresh_query", "")) or "")).strip()
    kind = str(raw.get("kind", raw.get("fresh_kind", "web")) or "web").strip().lower()
    if kind not in {"news", "sports", "web"}:
        kind = "web"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(raw.get("reason", "") or "").strip()
    if not query:
        need_search = False
    return FreshSearchDecision(
        need_search=need_search,
        query=query[:120],
        kind=kind,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason[:60],
    )



def _parse_research_queries(raw: object, *, primary: str) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(raw, str) and raw.strip():
        values.append(raw)
    elif isinstance(raw, (list, tuple)):
        values.extend(str(item or "") for item in raw)
    if primary:
        values.insert(0, primary)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        query = re.sub(r"\s+", " ", str(item or "")).strip()[:160]
        key = re.sub(r"\s+", "", query.casefold())
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) >= 4:
            break
    return tuple(cleaned)


def _parse_tool_routing_decision(content: str) -> ToolRoutingDecision:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ToolRoutingDecision(reason="invalid_json")
    tool = str(raw.get("tool", raw.get("tool_name", "")) or "").strip().lower()
    if bool(raw.get("need_search", raw.get("need_fresh_context", False))):
        tool = "fresh_search"
    if bool(raw.get("need_tool", False)) and not tool:
        tool = str(raw.get("tool_name", raw.get("name", "")) or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "ignore": "none",
        "no_tool": "none",
        "fresh_search": "fresh_search",
        "search": "fresh_search",
        "web_search": "fresh_search",
        "fresh": "fresh_search",
        "fresh_context": "fresh_search",
        "news": "fresh_search",
        "sports": "fresh_search",
        "market": "market",
        "market_check": "market",
        "quote": "market",
        "stock": "market",
        "crypto": "market",
        "url": "deep_url",
        "webpage": "deep_url",
        "deep_url": "deep_url",
        "probability": "probability",
        "prob": "probability",
        "jev": "probability",
        "jev_probability": "probability",
    }
    tool = aliases.get(tool, "none")
    query = re.sub(
        r"\s+",
        " ",
        str(raw.get("query", raw.get("fresh_query", raw.get("url", ""))) or ""),
    ).strip()
    kind = str(raw.get("kind", raw.get("fresh_kind", "web")) or "web").strip().lower()
    if kind not in {"news", "sports", "web"}:
        kind = "web"
    symbols = _parse_tool_symbols(raw.get("symbols", []))
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(raw.get("reason", "") or "").strip()
    comment_after_tool = bool(raw.get("comment_after_tool", True))
    if tool == "fresh_search" and not query:
        tool = "none"
        reason = reason or "empty_query"
    if tool == "deep_url" and not re.search(r"https?://", query, re.IGNORECASE):
        tool = "none"
        reason = reason or "missing_url"
    if tool == "market" and not symbols:
        tool = "none"
        reason = reason or "missing_symbols"
    if tool == "none":
        query = ""
        symbols = ()
    queries = _parse_research_queries(raw.get("queries"), primary=query)
    if tool != "fresh_search":
        queries = ()
    return ToolRoutingDecision(
        tool=tool,
        query=query[:160],
        queries=queries,
        kind=kind,
        symbols=symbols,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason[:80],
        comment_after_tool=comment_after_tool,
    )


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
    side_reaction = _normalize_reaction_name(
        str(
            raw.get("side_reaction", "")
            or raw.get("sideReaction", "")
            or raw.get("emoji_reaction", "")
            or raw.get("emojiReaction", "")
            or ""
        )
    )
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
    if action == "react":
        if not reaction and side_reaction:
            reaction = side_reaction
        side_reaction = ""
    elif reaction and not side_reaction:
        side_reaction = reaction
        reaction = ""
    if need_tool and tool == "market":
        action = "market_check"
    if not should_reply:
        action = "ignore"
        reaction = ""
        side_reaction = ""
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
        side_reaction,
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
        "clarify": "clarify",
        "clarification": "clarify",
        "对齐": "clarify",
        "warm_tease": "warm_tease",
        "亲昵吐槽": "warm_tease",
        "deflate": "deflate",
        "扎破": "deflate",
        "take_side": "take_side",
        "站边": "take_side",
        "share_self": "share_self",
        "分享自己": "share_self",
        "comfort_joke": "comfort_joke",
        "玩笑安慰": "comfort_joke",
        "mirror_style": "mirror_style",
        "学语气": "mirror_style",
        "amp_bit": "amp_bit",
        "加码": "amp_bit",
        "deadpan_echo": "deadpan_echo",
        "冷接": "deadpan_echo",
        "commit_bit": "commit_bit",
        "入戏": "commit_bit",
        "hyperbole": "hyperbole",
        "夸张": "hyperbole",
        "wrong_register": "wrong_register",
        "错位正经": "wrong_register",
        "protect": "protect",
        "挡一句": "protect",
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
        # Some providers truncate the optional structured arrays after already
        # completing the summary. Preserve that useful prefix so one malformed
        # batch cannot block the summary cursor forever.
        recovered_summary = _recover_json_string_field(content, "summary").strip()
        return MidMemoryDraft(recovered_summary[:1200], ())
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
        if _style_rule_leaks_specific_identity(situation, style):
            continue
        key = (situation, style)
        if key in seen:
            continue
        seen.add(key)
        evidence_messages = [
            message for message in _source_messages_for_style_rule(item, source_messages)
            if not message.is_bot
        ]
        if _style_rule_copies_evidence(situation, style, evidence_messages):
            continue
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


def _style_rule_leaks_specific_identity(situation: str, style: str) -> bool:
    text = f"{situation} {style}"
    if "[#" in text or "QQ" in text.upper() or "source_id" in text:
        return True
    return bool(re.search(r"\d{5,}", text))


def _style_rule_copies_evidence(
    situation: str,
    style: str,
    evidence_messages: list[ChatMessage],
) -> bool:
    style_compact = re.sub(r"\s+", "", style)
    situation_compact = re.sub(r"\s+", "", situation)
    if not style_compact:
        return True
    for message in evidence_messages:
        source_compact = re.sub(r"\s+", "", message.text)
        if not source_compact:
            continue
        if style_compact == source_compact:
            return True
        if len(style_compact) >= 6 and style_compact in source_compact:
            return True
        if len(situation_compact) >= 10 and situation_compact in source_compact:
            return True
        if _has_common_substring(style_compact, source_compact, min_len=10):
            return True
    return False


def _has_common_substring(a: str, b: str, *, min_len: int) -> bool:
    if len(a) < min_len or len(b) < min_len:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    max_size = min(len(shorter), 24)
    for size in range(max_size, min_len - 1, -1):
        for start in range(0, len(shorter) - size + 1):
            if shorter[start : start + size] in longer:
                return True
    return False


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


_REPLY_JSON_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)response_format\s*[:=]\s*\{?\s*[\"']?type[\"']?\s*[:=]\s*[\"']?json_object[\"']?\s*\}?"),
    re.compile(r"(?i)please\s+respond\s+in\s+json[^，。！？!?\n]*"),
    re.compile(r"(?i)valid\s+json[^，。！？!?\n]*"),
    re.compile(r"请严格按照\s*JSON(?:格式)?输出?"),
    re.compile(r"必须输出合法\s*JSON"),
    re.compile(r"不要代码块"),
    re.compile(r"不要输出\s*JSON\s*以外的任何文字"),
    re.compile(r"格式\s*[:：]\s*\{?"),
    re.compile(r"(?i)\bcandidates\b"),
)


def _strip_reply_json_artifacts(text: str) -> str:
    at_placeholders: list[str] = []

    def protect_at(match: re.Match[str]) -> str:
        at_placeholders.append(match.group(0))
        return f"__QQ_AT_PLACEHOLDER_{len(at_placeholders) - 1}__"

    cleaned = re.sub(r"\[\[at:\d+\]\]", protect_at, text)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.IGNORECASE)
    for pattern in _REPLY_JSON_ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.translate(str.maketrans({"{": "", "}": "", "[": "", "]": ""}))
    for idx, placeholder in enumerate(at_placeholders):
        cleaned = cleaned.replace(f"__QQ_AT_PLACEHOLDER_{idx}__", placeholder)
    cleaned = re.sub(r"[\"'“”‘’]+\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([，。！？!?；;：:])", r"\1", cleaned)
    cleaned = re.sub(r"[，,；;：:]+\s*$", "", cleaned)
    return cleaned.strip()


def _strip_internal_source_markers(text: str) -> str:
    cleaned = re.sub(r"\[S\d+\]", "", text)
    cleaned = re.sub(r"(?<![A-Za-z0-9])S\d+(?:S\d+)+", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *([，。！？,.!?])", r"\1", cleaned)
    return cleaned.strip()


def _sanitize_reply(content: str, max_chars: int) -> str:
    text = content.strip().strip("\"'")
    text = _strip_reply_json_artifacts(text).strip("\"'")
    marker = re.sub(r"[\s\"'`“”‘’()（）\[\]【】{}<>《》。.!！?？:：;；,，、-]+", "", text)
    if marker in {"", "空字符串", "无", "不回复", "空", "null", "None"}:
        return ""
    if len(text) > max_chars:
        text = _trim_to_sentence(text, max_chars)
        text = _strip_reply_json_artifacts(text).strip("\"'")
    return _strip_internal_source_markers(text)


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
        _log_reply_candidate_parse_diagnostic(
            raw_count=0,
            parsed_count=0,
            limit=limit,
            dropped_reasons=("invalid_json",),
        )
        return ()

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
            "另外，风雪刚才已经对别人说过以下内容，本轮禁止复用其措辞或核心答案："
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
                    f"candidates 必须正好 {candidate_count} 条，text 不能空，各条不能重复，"
                    f"不要输出 JSON 以外的任何文字。{avoid_instruction}"
                ),
            },
        ]
    )
    retry_request = dict(request)
    retry_request["messages"] = messages
    retry_floor = 320 if candidate_count == 1 else 900
    retry_request["max_tokens"] = max(int(request.get("max_tokens", 0) or 0), retry_floor)
    retry_request["response_format"] = {"type": "json_object"}
    return retry_request


def _filter_recent_bot_duplicate_candidates(
    candidates: tuple[ReplyCandidateDraft, ...],
    recent_bot_replies: tuple[str, ...],
) -> tuple[ReplyCandidateDraft, ...]:
    # Lexical overlap is not a send gate. Pre-send Jev decides whether a draft
    # is actually repeating 风雪. Keep the helper so generation still has a hook.
    return candidates


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


# Keep the public import used by existing task modules and tests.
DeepSeekClient = LLMTaskClient
