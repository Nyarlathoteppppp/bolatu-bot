from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    group_id: int
    user_id: int
    nickname: str
    text: str
    is_bot: bool
    created_at: float
    id: int = 0
    source_message_id: str = ""
    session_id: str = ""
    message_segments_json: str = ""
    raw_message_json: str = ""
    sender_json: str = ""



@dataclass(frozen=True)
class RawCorpusExample:
    message: ChatMessage
    before: tuple[ChatMessage, ...]
    after: tuple[ChatMessage, ...]
    tags: tuple[str, ...]
    score: float



@dataclass(frozen=True)
class MemorySummary:
    group_id: int
    summary: str
    recall_cues: tuple[str, ...]
    start_at: float
    end_at: float
    created_at: float
    id: int = 0
    status: str = "active"
    locked: bool = False
    updated_at: float = 0.0



@dataclass(frozen=True)
class StyleRule:
    group_id: int
    situation: str
    style: str
    source_text: str
    created_at: float
    scope: str = "group"
    source_user_ids: tuple[int, ...] = ()
    source_message_ids: tuple[int, ...] = ()
    support_user_count: int = 1
    evidence_count: int = 1
    confidence: float = 0.6
    status: str = "active"
    valid_to: float | None = None
    rule_fingerprint: str = ""
    last_seen_at: float = 0.0
    merged_count: int = 0



@dataclass(frozen=True)
class MemberProfile:
    group_id: int
    user_id: int
    display_name: str
    aliases: tuple[str, ...]
    last_seen_at: float



@dataclass(frozen=True)
class PrivateConversationState:
    """Small, explicit state for one direct conversation.

    Facts stay in ``memory_atoms`` because they need evidence and an audit trail.
    This record only holds the mutable conversation state that makes a follow-up
    feel coherent: the relationship note, the current subject, and open threads.
    """

    chat_id: int
    user_id: int
    display_name: str
    relationship_note: str
    interaction_tone: str
    current_topic: str
    open_threads: tuple[str, ...]
    frozen_fields: tuple[str, ...]
    updated_at: float



@dataclass(frozen=True)
class GroupInfo:
    group_id: int
    group_name: str
    member_count: int
    max_member_count: int
    last_synced_at: float



@dataclass(frozen=True)
class GroupMember:
    group_id: int
    user_id: int
    nickname: str
    card: str
    role: str
    title: str
    joined_at: float
    last_sent_at: float
    last_synced_at: float
    active: bool



@dataclass(frozen=True)
class MemberImpression:
    group_id: int
    user_id: int
    display_name: str
    aliases: tuple[str, ...]
    message_count: int
    top_tags: tuple[tuple[str, int], ...]
    top_keywords: tuple[tuple[str, int], ...]
    recent_texts: tuple[str, ...]
    ai_summary: str
    ai_interests: tuple[str, ...]
    ai_speaking_style: str
    ai_representative_texts: tuple[str, ...]
    ai_summary_at: float
    last_seen_at: float
    updated_at: float



@dataclass(frozen=True)
class MemberProfileSummary:
    group_id: int
    user_id: int
    profile_summary: str
    interests: tuple[str, ...]
    speaking_style: str
    representative_texts: tuple[str, ...]
    start_at: float
    end_at: float
    message_count: int
    created_at: float



@dataclass(frozen=True)
class BotSentMessage:
    group_id: int
    message_id: int
    bot_reply: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    created_at: float



@dataclass(frozen=True)
class RecalledReplyFeedback:
    group_id: int
    message_id: int
    bot_reply: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    owner_reason: str
    scene_summary: str
    bad_reply_problem: str
    avoid_rule: str
    better_direction: str
    tags: tuple[str, ...]
    operator_id: int
    reason_user_id: int
    recalled_at: float
    reason_at: float



@dataclass(frozen=True)
class ApprovedReplyFeedback:
    group_id: int
    candidate_text: str
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    action: str
    style: str
    tags: tuple[str, ...]
    operator_id: int
    created_at: float



@dataclass(frozen=True)
class CustomJargonEntry:
    group_id: int
    term: str
    explanation: str
    created_by: int
    created_at: float



@dataclass(frozen=True)
class LLMUsageSummary:
    task: str
    model: str
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_at: float
    last_at: float



@dataclass(frozen=True)
class LLMUsageEvent:
    task: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: float



@dataclass(frozen=True)
class BotMetricEvent:
    event_type: str
    group_id: int | None
    user_id: int | None
    stage: str
    action: str
    metadata: dict[str, object]
    created_at: float



@dataclass(frozen=True)
class BotMetricSummary:
    event_type: str
    stage: str
    action: str
    count: int



@dataclass(frozen=True)
class MemeAsset:
    id: int
    sha256: str
    source_message_id: str
    file_path: str
    mime_type: str
    byte_size: int
    description: str
    tags: tuple[str, ...]
    enabled: bool
    created_at: float
    last_used_at: float | None
    use_count: int



@dataclass(frozen=True)
class MemoryAtom:
    id: int
    atom_type: str
    group_id: int
    subject_user_id: int | None
    object_user_id: int | None
    content: str
    source: str
    confidence: float
    importance: float
    expires_at: float | None
    created_at: float
    updated_at: float
    evidence_type: str = "manual"
    source_message_id: str | None = None
    observed_at: float = 0.0
    valid_from: float | None = None
    valid_to: float | None = None
    status: str = "active"
    supersedes_id: int | None = None

    @property
    def evidence_source(self) -> str:
        return self.evidence_type



@dataclass(frozen=True)
class MemoryAtomAuditEvent:
    id: int
    atom_id: int
    action: str
    evidence_type: str
    source: str
    source_message_id: str | None
    actor_user_id: int | None
    detail: str
    observed_at: float
    created_at: float
    metadata: dict[str, object]

