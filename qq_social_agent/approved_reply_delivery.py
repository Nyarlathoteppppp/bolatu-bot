from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot, Message
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .approval_models import DeliveryProgress, PendingApprovalCandidate, PendingGroupApproval
from .memory import MemoryStore
from .pipeline_types import PipelineState


@dataclass(frozen=True)
class ApprovedReplyDeliveryServices:
    memory: MemoryStore
    pending_approvals: dict[int, PendingGroupApproval]
    group_inbound_sequences: dict[int, int]
    last_group_mention_targets: dict[int, tuple[int, float]]
    send_private_message: Callable[..., Awaitable[object]]
    send_private_text: Callable[[Bot, int, str], Awaitable[None]]
    send_group_message: Callable[..., Awaitable[int | None]]
    extract_message_id: Callable[[object], int | None]
    record_metric_event: Callable[..., None]
    pipeline_mark_sending: Callable[[PipelineState], None]
    pipeline_mark_sent: Callable[[PipelineState, int | None], None]
    pipeline_mark_completed: Callable[..., None]
    pipeline_mark_failed: Callable[[PipelineState, str], None]
    action_failed_summary: Callable[[Exception], str]
    record_user_reply: Callable[[int, int], None]
    build_delivery_plan: Callable[..., Any]
    message_from_reply_part: Callable[..., Message]
    first_allowed_mention_id: Callable[[str, dict[int, str]], int | None]
    prepare_group_political_send_texts: Callable[..., Awaitable[tuple[str, str, list[str]]]]
    is_group_send_blocked_error: Callable[[Exception], bool]
    notify_owner_political_gag: Callable[..., Awaitable[None]]
    memory_text_from_reply_part: Callable[[str, dict[int, str]], str]
    record_bot_sent_message: Callable[..., None]
    approval_user_ids: Callable[[], tuple[int, ...]]
    short_notice_text: Callable[[str, int], str]
    record_post_reply_followup_window: Callable[..., None]
    maybe_send_group_meme: Callable[..., Awaitable[None]]
    execute_approved_side_reaction: Callable[..., Awaitable[None]]
    save_approved_reply_feedback: Callable[..., None]
    logger: Any


async def send_approved_group_reply_inner(
    bot: Bot,
    approval: PendingGroupApproval,
    candidate: PendingApprovalCandidate,
    *,
    approver_id: int | None,
    high_quality: bool,
    notify_success: bool = True,
    services: ApprovedReplyDeliveryServices,
) -> None:
    send_started_at = time.monotonic()
    pipeline_state = approval.pipeline_state
    if pipeline_state is not None:
        services.pipeline_mark_sending(pipeline_state)
    services.logger.info(
        "qq_social_agent group approval accepted: "
        f"approver={approver_id} group={approval.group_id} candidate={candidate.index} high_quality={high_quality}"
    )
    services.record_metric_event(
        "approval_accepted",
        group_id=approval.group_id,
        user_id=approval.trigger_user_id,
        stage="approval",
        action=candidate.action,
        approver_id=approver_id,
        high_quality=high_quality,
        candidate_index=candidate.index,
        approval_wait_ms=max(0, int((time.time() - approval.created_at) * 1000)),
        correlation_id=pipeline_state.correlation_id if pipeline_state is not None else None,
    )
    private_reply_user_id = pipeline_state.private_reply_user_id if pipeline_state is not None else 0
    if private_reply_user_id:
        private_text = services.memory_text_from_reply_part(candidate.text, approval.mention_targets)
        try:
            result = await services.send_private_message(
                bot,
                user_id=private_reply_user_id,
                message=Message(f"（回复你刚才在群里的提问）\n{private_text}"),
            )
            sent_message_id = services.extract_message_id(result)
            if pipeline_state is not None:
                services.pipeline_mark_sent(pipeline_state, sent_message_id)
                services.pipeline_mark_completed(
                    pipeline_state,
                    elapsed_ms=int((time.monotonic() - send_started_at) * 1000),
                )
            services.record_metric_event(
                "message_sent",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="send",
                action=candidate.action,
                delivery="private_group_question_redirect",
                private_reply_user_id=private_reply_user_id,
                message_count=1,
                elapsed_ms=int((time.monotonic() - send_started_at) * 1000),
                receive_elapsed_ms=(
                    int((time.monotonic() - pipeline_state.received_monotonic) * 1000)
                    if pipeline_state is not None else None
                ),
                correlation_id=pipeline_state.correlation_id if pipeline_state is not None else None,
                approval_id=approval.approval_id,
                pipeline_stages=list(pipeline_state.stage_history) if pipeline_state is not None else [],
            )
        except ActionFailed as exc:
            if pipeline_state is not None:
                services.pipeline_mark_failed(pipeline_state, services.action_failed_summary(exc))
            services.logger.warning(
                "qq_social_agent failed redirecting group answer to private: "
                f"group={approval.group_id} user={private_reply_user_id} {services.action_failed_summary(exc)}"
            )
            services.record_metric_event(
                "private_redirect_failed",
                group_id=approval.group_id,
                user_id=private_reply_user_id,
                stage="send",
                action="action_failed",
                approval_id=approval.approval_id,
                error=services.action_failed_summary(exc),
            )
            if approver_id is not None and approver_id != private_reply_user_id:
                await services.send_private_text(bot, approver_id, f"私聊转发失败：{services.action_failed_summary(exc)}")
            return
        if high_quality:
            services.save_approved_reply_feedback(approval, candidate, approver_id=approver_id or 0)
        if notify_success and approver_id is not None and approver_id != private_reply_user_id:
            await services.send_private_text(bot, approver_id, "已私聊转发。")
        return

    delivery_plan = services.build_delivery_plan(
        reply_text=candidate.text,
        mention_targets=approval.mention_targets,
        trigger_user_id=approval.trigger_user_id,
        trigger_nickname=approval.trigger_nickname,
        trigger_sequence=approval.trigger_sequence,
        current_sequence=services.group_inbound_sequences.get(
            approval.group_id,
            approval.trigger_sequence,
        ),
        max_messages=3,
    )
    effective_mention_targets = delivery_plan.mention_targets
    if delivery_plan.forced_trigger_mention:
        services.record_metric_event(
            "stale_reply_mention",
            group_id=approval.group_id,
            user_id=approval.trigger_user_id,
            stage="send",
            action="force_mention",
            newer_message_count=delivery_plan.sequence_lag,
        )
    progress = approval.delivery_progress.setdefault(candidate.text, DeliveryProgress(parts=delivery_plan.parts))
    if progress.completed:
        if pipeline_state is not None:
            services.pipeline_mark_completed(pipeline_state)
        return
    if progress.uncertain_index is not None:
        if pipeline_state is not None:
            services.pipeline_mark_failed(pipeline_state, "delivery_unknown_requires_verification")
        services.record_metric_event(
            "group_send_failed",
            group_id=approval.group_id,
            stage="send",
            action="unknown_no_retry",
            approval_id=approval.approval_id,
            part_index=progress.uncertain_index,
        )
        if approver_id is not None:
            await services.send_private_text(
                bot,
                approver_id,
                "上一段发送结果未知，已阻止重复发送；请先核对群内是否收到。",
            )
        return

    reply_parts = progress.parts
    sent_mention_user_id: int | None = None
    recorded_user_reply = bool(progress.sent_message_ids)
    for index, part_text in enumerate(reply_parts):
        if index < len(progress.sent_message_ids):
            if sent_mention_user_id is None:
                sent_mention_user_id = services.first_allowed_mention_id(part_text, effective_mention_targets)
            continue
        attempted = False
        acknowledged = False
        try:
            part_mention_user_id = services.first_allowed_mention_id(part_text, effective_mention_targets)
            public_text, memory_text, gag = await services.prepare_group_political_send_texts(
                part_text,
                context=approval.trigger_text + "\n" + candidate.text,
            )
            attempted = True
            sent_message_id = await services.send_group_message(
                bot,
                approval.group_id,
                services.message_from_reply_part(
                    public_text,
                    effective_mention_targets,
                    quote_message_id=approval.source_message_id if index == 0 else "",
                ),
            )
            acknowledged = True
            progress.sent_message_ids.append(sent_message_id)
            if pipeline_state is not None:
                services.pipeline_mark_sent(pipeline_state, sent_message_id)
            if not recorded_user_reply:
                services.record_user_reply(approval.group_id, approval.trigger_user_id)
                recorded_user_reply = True
            if gag:
                await services.notify_owner_political_gag(
                    original=part_text,
                    public=public_text,
                    hits=gag,
                    group_id=approval.group_id,
                    source=candidate.action,
                )
            memory_text = services.memory_text_from_reply_part(memory_text, effective_mention_targets)
            services.record_bot_sent_message(
                group_id=approval.group_id,
                message_id=sent_message_id,
                bot_reply=memory_text,
                trigger_user_id=approval.trigger_user_id,
                trigger_nickname=approval.trigger_nickname,
                trigger_text=approval.trigger_text,
                action=candidate.action,
            )
            if sent_mention_user_id is None and part_mention_user_id is not None:
                sent_mention_user_id = part_mention_user_id
            services.memory.add_message(
                approval.group_id,
                approval.self_id,
                approval.persona_name,
                memory_text,
                is_bot=True,
                source_message_id=sent_message_id,
                source_kind="live",
                correlation_id=approval.correlation_id,
            )
        except asyncio.CancelledError:
            if attempted and not acknowledged:
                progress.uncertain_index = index
            raise
        except Exception as exc:
            blocked = isinstance(exc, ActionFailed) and services.is_group_send_blocked_error(exc)
            unknown = attempted and not acknowledged and (
                not isinstance(exc, ActionFailed) or "timeout" in str(exc).lower()
            )
            if unknown:
                progress.uncertain_index = index
            if pipeline_state is not None:
                services.pipeline_mark_failed(pipeline_state, services.action_failed_summary(exc))
            services.logger.warning(
                "qq_social_agent failed sending approved group reply: "
                f"group={approval.group_id} {services.action_failed_summary(exc)}"
            )
            services.record_metric_event(
                "group_send_failed",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="send",
                action="unknown" if unknown else "blocked_120" if blocked else "action_failed",
                delivered_parts=len(progress.sent_message_ids),
                failed_part_index=index,
                delivery_status="unknown" if unknown else "partial" if progress.sent_message_ids else "failed",
                approval_id=approval.approval_id,
                error=services.action_failed_summary(exc),
                candidate_index=candidate.index,
            )
            if blocked:
                current_mute = float(services.memory.group_state(approval.group_id)["muted_until"] or 0)
                if current_mute <= time.time():
                    services.memory.mute_until(approval.group_id, time.time() + 10 * 60)
                services.pending_approvals[approval.group_id] = approval
                notice = (
                    f"群 {approval.group_id} 发言失败：QQ 内核返回 result=120，"
                    f"通常是机器人被群禁言或发送受限。已确认发送 {len(progress.sent_message_ids)} 段，剩余候选已保留。\n"
                    f"审批ID：{approval.approval_id}\n"
                    f"候选 {candidate.index}：{services.short_notice_text(candidate.text, 180)}\n"
                    "解除禁言后可再次回复对应候选编号发送。"
                )
                for target_id in services.approval_user_ids():
                    await services.send_private_text(bot, target_id, notice)
            try:
                if approver_id is not None and not blocked:
                    await services.send_private_message(
                        bot,
                        user_id=approver_id,
                        message=Message(f"发送失败：{services.action_failed_summary(exc)}"),
                    )
            except Exception:
                pass
            return
        if index < len(reply_parts) - 1:
            await asyncio.sleep(0.9)

    progress.completed = True
    if sent_mention_user_id is not None:
        services.last_group_mention_targets[approval.group_id] = (sent_mention_user_id, time.time())
    else:
        services.last_group_mention_targets.pop(approval.group_id, None)
    services.record_post_reply_followup_window(
        approval.group_id,
        trigger_user_id=approval.trigger_user_id,
        mention_user_id=sent_mention_user_id,
        conversation_engaged=bool(pipeline_state is not None and pipeline_state.addressed),
    )
    await services.maybe_send_group_meme(bot, approval, candidate)
    await services.execute_approved_side_reaction(bot, approval)
    send_elapsed_ms = int((time.monotonic() - send_started_at) * 1000)
    if pipeline_state is not None:
        services.pipeline_mark_completed(pipeline_state, elapsed_ms=send_elapsed_ms)
    services.record_metric_event(
        "message_sent",
        group_id=approval.group_id,
        user_id=approval.trigger_user_id,
        stage="send",
        action=candidate.action,
        message_count=len(reply_parts),
        elapsed_ms=send_elapsed_ms,
        receive_elapsed_ms=(
            int((time.monotonic() - pipeline_state.received_monotonic) * 1000)
            if pipeline_state is not None else None
        ),
        correlation_id=pipeline_state.correlation_id if pipeline_state is not None else None,
        approval_id=approval.approval_id,
        pipeline_stages=list(pipeline_state.stage_history) if pipeline_state is not None else [],
    )
    if high_quality:
        services.save_approved_reply_feedback(approval, candidate, approver_id=approver_id or 0)
    if not notify_success or approver_id is None:
        return
    try:
        await services.send_private_message(bot, user_id=approver_id, message=Message("已发。"))
    except ActionFailed:
        pass
