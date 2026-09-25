from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .approval_models import PendingApprovalCandidate, PendingGroupApproval
from .approval_state_service import ApprovalStateService


@dataclass(frozen=True)
class ApprovalRequestServices:
    state: ApprovalStateService
    review_enabled: Callable[[], bool]
    auto_send_percent: Callable[[], int]
    auto_send_selected: Callable[[int], bool]
    duplicate_reply_verdict: Callable[..., Awaitable[tuple[bool, str]]]
    send_approved_group_reply: Callable[..., Awaitable[None]]
    record_metric_event: Callable[..., None]
    approval_user_ids: Callable[[], tuple[int, ...]]
    send_private_message: Callable[..., Awaitable[object]]
    member_label: Callable[[int, str], str]
    action_failed_summary: Callable[[Exception], str]
    logger: Any


def format_approval_candidates(approval: PendingGroupApproval) -> str:
    lines = []
    for candidate in approval.candidates:
        style_line = f"\n   style：{candidate.style.strip()}" if candidate.style.strip() else ""
        lines.append(f"{candidate.index}. {candidate.text}{style_line}")
    return "\n\n".join(lines).strip()


def approval_side_reaction(approval: PendingGroupApproval) -> str:
    pipeline_state = approval.pipeline_state
    return str(pipeline_state.decision_side_reaction or "").strip() if pipeline_state is not None else ""


async def request_group_approval(
    bot: Bot,
    approval: PendingGroupApproval,
    *,
    services: ApprovalRequestServices,
) -> None:
    if not services.review_enabled():
        services.state.remove(approval.group_id)
        candidate = approval.candidates[0] if approval.candidates else None
        if candidate is None:
            services.logger.info(
                "qq_social_agent auto approval skipped: "
                f"group={approval.group_id} reason=no_candidate"
            )
            return
        await _auto_send_candidate(bot, approval, candidate, percent=None, services=services)
        return
    auto_send_percent = services.auto_send_percent()
    if auto_send_percent > 0 and services.auto_send_selected(auto_send_percent):
        services.state.remove(approval.group_id)
        candidate = approval.candidates[0] if approval.candidates else None
        if candidate is None:
            services.logger.info(
                "qq_social_agent probabilistic auto approval skipped: "
                f"group={approval.group_id} reason=no_candidate percent={auto_send_percent}"
            )
            return
        await _auto_send_candidate(bot, approval, candidate, percent=auto_send_percent, services=services)
        return

    services.state.put(approval)
    preview = format_approval_candidates(approval)
    evidence_section = (
        f"\n\n联网依据（仅供审批核对）：\n{approval.tool_evidence}"
        if approval.tool_evidence
        else ""
    )
    private_reply_user_id = approval.pipeline_state.private_reply_user_id if approval.pipeline_state is not None else 0
    delivery_line = f"发送位置：私聊 {private_reply_user_id}（仅回复这次群内提问）\n" if private_reply_user_id else ""
    side_reaction = approval_side_reaction(approval)
    side_reaction_line = f"附带表情：{side_reaction}\n" if side_reaction else ""
    message = (
        f"待发群：{approval.group_id}\n"
        f"审批ID：{approval.approval_id}\n"
        f"触发人：{services.member_label(approval.trigger_user_id, approval.trigger_nickname)}\n"
        f"触发消息：{approval.trigger_text}\n"
        f"{delivery_line}{side_reaction_line}\n"
        f"候选：\n{preview}{evidence_section}\n\n"
        "回复：A/B/C 或 1/2/3 发送；D/X/取消 不发；T 工具单。"
    )
    delivered = 0
    approval_user_ids = services.approval_user_ids()
    for approver_id in approval_user_ids:
        try:
            await services.send_private_message(bot, user_id=approver_id, message=Message(message))
            delivered += 1
        except ActionFailed as exc:
            services.logger.warning(
                "qq_social_agent failed sending group approval request: "
                f"approver={approver_id} group={approval.group_id} "
                f"{services.action_failed_summary(exc)}"
            )
    if delivered <= 0:
        services.state.remove(approval.group_id)
        services.record_metric_event(
            "approval_request_failed",
            group_id=approval.group_id,
            user_id=approval.trigger_user_id,
            stage="approval",
            action="send_private_failed",
            candidate_count=len(approval.candidates),
            correlation_id=approval.pipeline_state.correlation_id if approval.pipeline_state is not None else None,
        )
        return
    services.record_metric_event(
        "approval_requested",
        group_id=approval.group_id,
        user_id=approval.trigger_user_id,
        stage="approval",
        action="pending",
        candidate_count=len(approval.candidates),
        delivered=delivered,
        correlation_id=approval.pipeline_state.correlation_id if approval.pipeline_state is not None else None,
    )
    services.logger.info(
        "qq_social_agent group approval pending: "
        f"approvers={approval_user_ids} group={approval.group_id} approval_id={approval.approval_id} "
        f"candidates={len(approval.candidates)}"
    )


async def _auto_send_candidate(
    bot: Bot,
    approval: PendingGroupApproval,
    candidate: PendingApprovalCandidate,
    *,
    percent: int | None,
    services: ApprovalRequestServices,
) -> None:
    duplicate_send, duplicate_reason = await services.duplicate_reply_verdict(approval, candidate)
    if not duplicate_send:
        services.record_metric_event(
            "reply_suppressed",
            group_id=approval.group_id,
            user_id=approval.trigger_user_id,
            stage="pre_send_duplicate",
            action="skipped",
            reason=duplicate_reason,
            correlation_id=approval.pipeline_state.correlation_id if approval.pipeline_state is not None else None,
        )
        services.logger.info(
            "qq_social_agent skipped duplicate group reply: "
            f"group={approval.group_id} approval_id={approval.approval_id} reason={duplicate_reason}"
        )
        return
    stage = "review_disabled" if percent is None else "probability"
    fields = {"auto_send_percent": percent} if percent is not None else {}
    services.record_metric_event(
        "approval_auto_send",
        group_id=approval.group_id,
        user_id=approval.trigger_user_id,
        stage=stage,
        action=candidate.action,
        candidate_count=len(approval.candidates),
        correlation_id=approval.pipeline_state.correlation_id if approval.pipeline_state is not None else None,
        **fields,
    )
    description = "auto approval send" if percent is None else "probabilistic auto approval send"
    percent_text = "" if percent is None else f" percent={percent}"
    services.logger.info(
        f"qq_social_agent {description}: group={approval.group_id} approval_id={approval.approval_id} "
        f"candidate={candidate.index}{percent_text}"
    )
    await services.send_approved_group_reply(
        bot,
        approval,
        candidate,
        approver_id=None,
        high_quality=False,
        notify_success=False,
    )
