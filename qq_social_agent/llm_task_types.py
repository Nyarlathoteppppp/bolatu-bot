from __future__ import annotations

from dataclasses import dataclass


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
    reply_angle: str = ""



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

