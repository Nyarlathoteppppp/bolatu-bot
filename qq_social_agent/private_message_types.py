from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nonebot.adapters.onebot.v11 import Bot, PrivateMessageEvent

from .deepseek_client import ReplyDecision
from .memory import ChatMessage
from .persona import Persona
from .rag_retriever import RAGRetrievalResult
from .tool_router import ToolRoutePlan


@dataclass(frozen=True)
class BufferedPrivateMessage:
    bot: Bot
    event: PrivateMessageEvent
    text: str
    user_id: int
    nickname: str
    created_at: float
    source_message_id: str = ""
    correlation_id: str = ""


@dataclass(frozen=True)
class PrivateTurn:
    """Prepared private turn; later stages do not infer text from the event."""

    bot: Bot
    user_id: int
    chat_id: int
    self_id: int
    nickname: str
    source_message_id: str
    correlation_id: str
    text: str
    forced_once_context: str
    received_message_count: int


@dataclass(frozen=True)
class PrivateToolStage:
    turn: PrivateTurn
    persona: Persona
    context_recent: tuple[ChatMessage, ...]
    context_query: str
    decision: ReplyDecision
    tool_plan: ToolRoutePlan
    speaker_context: str
    private_state_context: str
    market_context: str
    fresh_context: str
    rag_task: asyncio.Task[RAGRetrievalResult]


@dataclass(frozen=True)
class PrivateGenerationContext:
    persona: Persona
    recent_messages: tuple[ChatMessage, ...]
    current_text: str
    current_nickname: str
    decision: ReplyDecision
    market_context: str
    fresh_context: str
    memory_context: str
    member_context: str
    memory_atoms_context: str
    style_context: str
    raw_corpus_context: str
    jargon_context: str
    recall_feedback_context: str
    speaker_context: str
    priority_context: str
