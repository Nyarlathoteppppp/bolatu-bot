from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .approval_models import PendingApprovalCandidate, PendingGroupApproval


@dataclass(frozen=True)
class ApprovalControl:
    kind: str
    candidate_index: int = 1
    high_quality: bool = False
    rejection_requested: bool = False
    rejection_reason: str = ""
    raw_text: str = ""


@dataclass(frozen=True)
class ApprovalStateServices:
    approval_user_ids: Callable[[], tuple[int, ...]]
    send_private_text: Callable[[Bot, int, str], Awaitable[None]]
    send_private_message: Callable[..., Awaitable[object]]
    save_rejection_feedback: Callable[..., None]
    record_metric_event: Callable[..., None]
    short_notice_text: Callable[[str, int], str]
    send_approved_group_reply: Callable[..., Awaitable[None]]
    cooldown_seconds: float
    logger: Any


class ApprovalStateService:
    """Own pending approval records, arbitration, stale-choice cooldowns, and selection."""

    def __init__(self) -> None:
        self.pending: dict[int, PendingGroupApproval] = {}
        self.choice_cooldowns: dict[int, float] = {}
        self.processing_lock = asyncio.Lock()

    def latest(self) -> PendingGroupApproval | None:
        if not self.pending:
            return None
        return max(self.pending.values(), key=lambda approval: approval.created_at)

    def put(self, approval: PendingGroupApproval) -> None:
        self.pending[approval.group_id] = approval

    def remove(self, group_id: int) -> PendingGroupApproval | None:
        return self.pending.pop(group_id, None)

    def clear(self) -> None:
        self.pending.clear()

    def cool_down_other_choices(self, approver_id: int, *, approval_user_ids: tuple[int, ...], until: float) -> None:
        for user_id in approval_user_ids:
            if user_id != approver_id:
                self.choice_cooldowns[user_id] = until

    async def handle_control(
        self,
        bot: Bot,
        user_id: int,
        control: ApprovalControl,
        *,
        is_admin: bool,
        services: ApprovalStateServices,
    ) -> None:
        candidate: PendingApprovalCandidate | None = None
        async with self.processing_lock:
            approval = self.latest()
            if approval is None:
                await services.send_private_text(bot, user_id, "当前没有待审批候选。")
                return
            self.pending.pop(approval.group_id, None)
            self.cool_down_other_choices(
                user_id,
                approval_user_ids=services.approval_user_ids(),
                until=time.time() + services.cooldown_seconds,
            )
            if control.kind == "select":
                candidate = next(
                    (item for item in approval.candidates if item.index == control.candidate_index),
                    None,
                )
                if control.high_quality and not is_admin:
                    await services.send_private_text(bot, user_id, "你只有基础审批权限，不能标优。")
                    return
            elif control.kind == "approve_first":
                candidate = approval.candidates[0] if approval.candidates else None
            if control.kind == "cancel":
                candidate = None
            if candidate is None:
                owner_reason = control.rejection_reason.strip()
                if control.rejection_requested and owner_reason and is_admin:
                    rejected_candidate = next(
                        (item for item in approval.candidates if item.index == control.candidate_index),
                        None,
                    )
                    services.save_rejection_feedback(
                        approval,
                        owner_reason,
                        reason_user_id=user_id,
                        candidate=rejected_candidate,
                        candidate_index=control.candidate_index,
                    )
                    response_text = "已取消，并记录不准奏原因。"
                elif control.rejection_requested:
                    response_text = "已取消。不准奏原因是空的，没写入反馈。"
                else:
                    response_text = "已取消。"
                services.logger.info(
                    "qq_social_agent group approval canceled: "
                    f"approver={user_id} group={approval.group_id} approval_id={approval.approval_id} "
                    f"text={control.raw_text!r}"
                )
                services.record_metric_event(
                    "approval_canceled",
                    group_id=approval.group_id,
                    user_id=approval.trigger_user_id,
                    stage="approval",
                    action="reject",
                    approver_id=user_id,
                    reason=services.short_notice_text(control.raw_text or "取消", 120),
                    correlation_id=approval.correlation_id,
                )
                try:
                    await services.send_private_message(
                        bot,
                        user_id=user_id,
                        message=Message(response_text),
                    )
                except ActionFailed:
                    pass
                return
        await services.send_approved_group_reply(
            bot,
            approval,
            candidate,
            approver_id=user_id,
            high_quality=control.high_quality,
        )
