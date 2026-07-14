from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class OutputChannel(str, Enum):
    SILENT = "silent"
    TEXT = "text"
    REACT = "react"
    POKE = "poke"


class PipelineMode(str, Enum):
    CHAT = "chat"
    SEARCH = "search"
    MARKET = "market"
    DEEP_URL = "deep_url"


class SocialIntent(str, Enum):
    ANSWER = "answer"
    CARE = "care"
    PLAY = "play"
    AGREE = "agree"
    CHAT = "chat"


class ToolKind(str, Enum):
    MARKET = "market"
    FRESH_SEARCH = "fresh_search"
    DEEP_URL = "deep_url"
    MEMORY = "memory"


@dataclass(frozen=True)
class ToolRequest:
    kind: ToolKind
    query: str = ""
    reason: str = ""
    confidence: float = 1.0
    required: bool = False
    arguments: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    kind: ToolKind
    status: str
    context: str = ""
    evidence: str = ""
    elapsed_ms: int = 0
    error: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class ContextSection:
    name: str
    content: str
    source: str = ""
    priority: int = 50


@dataclass(frozen=True)
class ContextPacket:
    mode: PipelineMode = PipelineMode.CHAT
    sections: tuple[ContextSection, ...] = ()
    rag_document_ids: tuple[int, ...] = ()
    rag_document_types: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()

    def get(self, name: str, default: str = "") -> str:
        for section in self.sections:
            if section.name == name:
                return section.content
        return default

    def as_dict(self) -> dict[str, str]:
        return {section.name: section.content for section in self.sections}


@dataclass
class PipelineState:
    correlation_id: str
    group_id: int
    user_id: int
    nickname: str
    text: str
    addressed: bool
    mode: PipelineMode = PipelineMode.CHAT
    trigger_sequence: int = 0
    output_channel: OutputChannel = OutputChannel.SILENT
    social_intent: SocialIntent = SocialIntent.CHAT
    decision_action: str = "ignore"
    decision_reason: str = ""
    decision_confidence: float = 0.0
    tool_requests: tuple[ToolRequest, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    context: ContextPacket = field(default_factory=ContextPacket)

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results = (*self.tool_results, result)

    def tool_result(self, kind: ToolKind) -> ToolResult | None:
        return next((item for item in self.tool_results if item.kind == kind), None)
