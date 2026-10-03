from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .approval_models import PendingApprovalCandidate, PendingGroupApproval
from .approval_request_service import approval_side_reaction
from .member_context import member_label
from .onebot_gateway import action_failed_summary
from .social_actions import reaction_from_action

if TYPE_CHECKING:
    from .deepseek_client import LLMTaskClient
    from .memory import MemoryStore
    from .meme_library import PrivateMemeLibrary
    from .social_actions import SocialActionService


@dataclass(frozen=True)
class GroupFollowupPolicy:
    hard_seconds: float
    soft_seconds: float


@dataclass(frozen=True)
class GroupPostSendServices:
    memory: MemoryStore
    client: LLMTaskClient | None
    meme_library: PrivateMemeLibrary
    social_actions: SocialActionService
    followup_windows: dict[tuple[int, int], float]
    followup_policy: GroupFollowupPolicy
    send_group_message: Callable[[Bot, int, Message], Awaitable[int | None]]
    record_metric_event: Callable[..., None]
    memory_text_from_reply_part: Callable[..., str]
    logger: Any


def group_meme_context_eligible(approval: PendingGroupApproval, candidate: PendingApprovalCandidate) -> bool:
    if candidate.action in {"market_check", "fresh_context"}:
        return False
    if approval.tool_evidence.strip():
        return False
    return 0 < len(candidate.text.strip()) <= 150


class GroupPostSendService:
    """Effects around successful delivery; the delivery flow owns call ordering."""

    def __init__(self, services: GroupPostSendServices) -> None:
        self.services = services

    async def maybe_send_meme(
        self,
        bot: Bot,
        approval: PendingGroupApproval,
        candidate: PendingApprovalCandidate,
    ) -> None:
        if self.services.client is None or not group_meme_context_eligible(approval, candidate):
            return
        gate = self.services.meme_library.group_gate(approval.group_id)
        if not gate.allowed:
            self.services.record_metric_event(
                "group_meme_selector",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="eligibility",
                action="skipped",
                gate_reason=gate.reason,
            )
            return
        candidates = self.services.meme_library.group_candidates(
            approval.group_id,
            query=f"{approval.trigger_text}\n{candidate.text}",
        )
        if not candidates:
            return
        try:
            choice = await self.services.client.select_private_meme(
                current_text=approval.trigger_text,
                reply_text=self.services.memory_text_from_reply_part(candidate.text, approval.mention_targets),
                candidates=self.services.meme_library.candidate_text(candidates),
            )
        except Exception as exc:
            self.services.logger.warning(
                "qq_social_agent group meme selector failed: "
                f"group={approval.group_id} error={exc}"
            )
            return
        candidate_ids = {asset.id for asset in candidates}
        if not choice.send or choice.meme_id not in candidate_ids:
            self.services.record_metric_event(
                "group_meme_selector",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="selection",
                action="skipped",
                gate_reason=gate.reason,
                reason=choice.reason,
            )
            return
        image_ref = self.services.meme_library.image_base64_ref(choice.meme_id)
        if not image_ref:
            return
        try:
            message_id = await self.services.send_group_message(
                bot,
                approval.group_id,
                Message(MessageSegment.image(file=image_ref)),
            )
        except ActionFailed as exc:
            self.services.logger.warning(
                "qq_social_agent failed sending group meme: "
                f"group={approval.group_id} meme={choice.meme_id} {action_failed_summary(exc)}"
            )
            self.services.record_metric_event(
                "group_meme_selector",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="delivery",
                action="failed",
                meme_id=choice.meme_id,
                error=action_failed_summary(exc),
            )
            return
        self.services.meme_library.mark_group_sent(approval.group_id, choice.meme_id)
        asset = self.services.memory.meme_asset(choice.meme_id)
        self.services.memory.add_message(
            approval.group_id,
            approval.self_id,
            approval.persona_name,
            f"[风雪附了一张已授权表情包：{asset.description if asset else choice.meme_id}]",
            is_bot=True,
            source_message_id=message_id,
            source_kind="live",
            correlation_id=approval.correlation_id,
        )
        if message_id is not None:
            self.services.memory.interactions.observe_sent(
                group_id=approval.group_id,
                source_message_id=str(message_id),
                trigger_source_id=approval.source_message_id,
                action="meme",
                context_at=approval.pipeline_state.interaction_context_at if approval.pipeline_state is not None else None,
            )
        self.services.record_metric_event(
            "group_meme_selector",
            group_id=approval.group_id,
            user_id=approval.trigger_user_id,
            stage="delivery",
            action="sent",
            meme_id=choice.meme_id,
            reason=choice.reason,
        )

    def record_followup_window(
        self,
        group_id: int,
        *,
        trigger_user_id: int,
        mention_user_id: int | None = None,
        conversation_engaged: bool = True,
    ) -> None:
        if not conversation_engaged:
            return
        now = time.time()
        target_user_ids = {int(trigger_user_id or 0)}
        if mention_user_id is not None:
            target_user_ids.add(int(mention_user_id or 0))
        target_user_ids.discard(0)
        opened_user_ids: list[int] = []
        refreshed_skipped: list[int] = []
        for target_user_id in sorted(target_user_ids):
            key = (group_id, target_user_id)
            opened_at = self.services.followup_windows.get(key, 0.0)
            if opened_at and now - opened_at <= self.services.followup_policy.soft_seconds:
                refreshed_skipped.append(target_user_id)
                continue
            self.services.followup_windows[key] = now
            opened_user_ids.append(target_user_id)
        if opened_user_ids or refreshed_skipped:
            self.services.record_metric_event(
                "followup_window_opened",
                group_id=group_id,
                user_id=trigger_user_id,
                stage="send",
                action="post_reply",
                target_user_ids=sorted(target_user_ids),
                opened_user_ids=opened_user_ids,
                skipped_refresh_user_ids=refreshed_skipped,
                window_seconds=self.services.followup_policy.hard_seconds,
                soft_window_seconds=self.services.followup_policy.soft_seconds,
            )

    async def execute_side_reaction(self, bot: Bot, approval: PendingGroupApproval) -> None:
        pipeline_state = approval.pipeline_state
        side_reaction = approval_side_reaction(approval)
        if pipeline_state is None or not side_reaction:
            return
        target_message_id = str(pipeline_state.source_message_id or "").strip()
        if not target_message_id.isdigit():
            self.services.logger.info(
                "qq_social_agent approved side reaction skipped: "
                f"group={approval.group_id} reason=missing_message_id reaction={side_reaction}"
            )
            return
        reaction = reaction_from_action(pipeline_state.decision_action, side_reaction)
        try:
            result = await self.services.social_actions.react_to_message(
                bot,
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                message_id=target_message_id,
                reaction=reaction,
                target_label=member_label(approval.trigger_user_id, approval.trigger_nickname),
            )
        except ActionFailed as exc:
            self.services.logger.warning(
                "qq_social_agent approved side reaction failed: "
                f"group={approval.group_id} message_id={target_message_id} {action_failed_summary(exc)}"
            )
            self.services.record_metric_event(
                "social_action_failed",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="approved_side_reaction",
                action="react",
                reaction=reaction,
                error=action_failed_summary(exc),
            )
            return
        except Exception as exc:
            self.services.logger.warning(
                "qq_social_agent approved side reaction failed: "
                f"group={approval.group_id} message_id={target_message_id} error={exc}"
            )
            self.services.record_metric_event(
                "social_action_failed",
                group_id=approval.group_id,
                user_id=approval.trigger_user_id,
                stage="approved_side_reaction",
                action="react",
                reaction=reaction,
                error=str(exc)[:160],
            )
            return
        self.services.record_metric_event(
            "social_action",
            group_id=approval.group_id,
            user_id=approval.trigger_user_id,
            stage="approved_side_reaction",
            action="react",
            reaction=result.reaction,
            reason=result.reason,
            emoji_id=result.emoji_id,
            sent=result.sent,
            approval_id=approval.approval_id,
        )
        if result.sent:
            self.services.logger.info(
                "qq_social_agent approved side reaction sent: "
                f"group={approval.group_id} message_id={target_message_id} "
                f"reaction={result.reaction} emoji_id={result.emoji_id}"
            )
        else:
            self.services.logger.info(
                "qq_social_agent approved side reaction skipped: "
                f"group={approval.group_id} message_id={target_message_id} reason={result.reason}"
            )

    def record_bot_sent_message(
        self,
        *,
        group_id: int,
        message_id: int | None,
        bot_reply: str,
        trigger_user_id: int,
        trigger_nickname: str,
        trigger_text: str,
        action: str,
    ) -> None:
        if message_id is None:
            self.services.logger.warning(
                "qq_social_agent bot sent message missing message_id: "
                f"group={group_id} action={action}"
            )
            return
        self.services.memory.add_bot_sent_message(
            group_id=group_id,
            message_id=message_id,
            bot_reply=bot_reply,
            trigger_user_id=trigger_user_id,
            trigger_nickname=trigger_nickname,
            trigger_text=trigger_text,
            action=action,
        )
