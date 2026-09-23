from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .pipeline_types import PipelineState


@dataclass(frozen=True)
class PendingApprovalCandidate:
    index: int
    text: str
    action: str
    style: str


@dataclass
class DeliveryProgress:
    parts: tuple[str, ...]
    sent_message_ids: list[int | None] = field(default_factory=list)
    uncertain_index: int | None = None
    completed: bool = False


@dataclass(frozen=True)
class PendingGroupApproval:
    approval_id: str
    group_id: int
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    persona_name: str
    self_id: int
    candidates: tuple[PendingApprovalCandidate, ...]
    mention_targets: dict[int, str]
    created_at: float
    correlation_id: str = ""
    tool_evidence: str = ""
    trigger_sequence: int = 0
    pipeline_state: PipelineState | None = None
    source_message_id: str = ""
    delivery_progress: dict[str, DeliveryProgress] = field(default_factory=dict, compare=False, repr=False)
    delivery_lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)
