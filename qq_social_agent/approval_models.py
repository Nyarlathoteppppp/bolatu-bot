from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingApprovalCandidate:
    index: int
    text: str
    action: str
    style: str


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
