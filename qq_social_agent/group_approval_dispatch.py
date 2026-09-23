from __future__ import annotations

import time
from typing import Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot

from .approval_models import PendingApprovalCandidate, PendingGroupApproval
from .pipeline_stages import apply_candidates, mark_approval_pending
from .pipeline_types import PipelineState


async def queue_group_reply_approval(
    bot: Bot,
    *,
    group_id: int,
    user_id: int,
    nickname: str,
    text: str,
    persona_name: str,
    self_id: int,
    candidates: tuple[PendingApprovalCandidate, ...],
    mention_targets: dict[int, str],
    trigger_sequence: int,
    pipeline_state: PipelineState,
    source_message_id: str,
    correlation_id: str,
    new_approval_id: Callable[[int], str],
    request_approval: Callable[[Bot, PendingGroupApproval], Awaitable[None]],
    tool_evidence: str = "",
    apply_candidates_first: bool = False,
) -> None:
    approval_id = new_approval_id(group_id)
    if apply_candidates_first:
        apply_candidates(pipeline_state, candidates)
    mark_approval_pending(pipeline_state, approval_id)
    await request_approval(
        bot,
        PendingGroupApproval(
            approval_id=approval_id,
            group_id=group_id,
            trigger_user_id=user_id,
            trigger_nickname=nickname,
            trigger_text=text,
            persona_name=persona_name,
            self_id=self_id,
            candidates=candidates,
            mention_targets=mention_targets,
            created_at=time.time(),
            correlation_id=correlation_id,
            tool_evidence=tool_evidence,
            trigger_sequence=trigger_sequence,
            pipeline_state=pipeline_state,
            source_message_id=source_message_id,
        ),
    )
