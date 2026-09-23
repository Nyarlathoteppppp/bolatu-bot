from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .approval_models import PendingApprovalCandidate
from .deepseek_client import DeepSeekClient, ReplyDecision
from .discourse_effects import RepairResolution
from .discourse_state import DiscourseState, draft_violates_media_gate
from .ellipsis_resolver import EllipsisResolution
from .jev_policy import GroupReplyBudget
from .memory import ChatMessage
from .pipeline_types import ContextPacket, PipelineMode
from .pre_send_critic import (
    CriticResult,
    apply_jev_critic_judgement,
    critic_prefers_clarify,
    format_critic_feedback,
    next_critic_action,
)
from .pronoun_guard import (
    apply_jev_pronoun_judgement,
    draft_has_person_pronoun,
    format_pronoun_feedback,
)
from .reference_resolver import ReferenceResolution
from .resolver_result import RESOLVED


@dataclass(frozen=True)
class GeneratedGroupReply:
    decision: ReplyDecision
    candidates: tuple[PendingApprovalCandidate, ...]
    critic: CriticResult
    regenerated: bool
    direct_single_reply: bool
    prompt_flow: str
    elapsed_ms: int


async def generate_group_reply(
    *,
    client: DeepSeekClient,
    decision: ReplyDecision,
    persona: Any,
    recent_messages: list[ChatMessage],
    text: str,
    nickname: str,
    current_label: str,
    addressed_bot: bool,
    addressed_repeat_count: int,
    cue_repeat_context: str,
    market_context: str,
    fresh_context: str,
    context_packet: ContextPacket,
    mode: PipelineMode,
    mention_targets_context: str,
    priority_context: str,
    speaker_context: str,
    memory_context: str,
    reference_resolution: ReferenceResolution,
    ellipsis_resolution: EllipsisResolution,
    repair_resolution: RepairResolution,
    discourse_state: DiscourseState,
    direct_single_reply: bool,
    market_report: str,
    reply_budget: GroupReplyBudget,
    group_id: int,
    user_id: int,
    build_approval_candidates: Callable[..., list[PendingApprovalCandidate]],
    combine_text_sections: Callable[..., str],
    record_metric_event: Callable[..., None],
    logger: Any,
) -> GeneratedGroupReply | None:
    tool_answer_mode = mode in {
        PipelineMode.SEARCH,
        PipelineMode.MARKET,
        PipelineMode.DEEP_URL,
        PipelineMode.PROBABILITY,
    }
    reply_candidate_limit = 1 if direct_single_reply or tool_answer_mode else 3
    prompt_flow = (
        "search_answer"
        if tool_answer_mode
        else "reply_direct"
        if direct_single_reply
        else "reply_candidates"
    )
    task_name = "search_answer" if tool_answer_mode else prompt_flow
    generation_started_at = time.monotonic()
    critic_feedback = ""
    critic_result = None
    approval_candidates: list[PendingApprovalCandidate] = []
    for attempt in range(2):
        effective_speaker_context = speaker_context
        if critic_feedback:
            effective_speaker_context = combine_text_sections(speaker_context, critic_feedback)
            if critic_prefers_clarify(critic_result) and attempt > 0:
                decision = replace(decision, action="clarify")
        try:
            reply_candidates = await client.reply_candidates(
                persona=persona,
                recent_messages=recent_messages,
                current_text=text,
                current_nickname=current_label,
                mentioned=addressed_bot,
                addressed_repeat_count=addressed_repeat_count,
                cue_repeat_context=cue_repeat_context,
                action=decision.action,
                chat_label="QQ 群聊",
                market_context=market_context,
                fresh_context=fresh_context,
                context_packet=context_packet,
                mention_targets=mention_targets_context,
                priority_context=priority_context,
                include_bot_history=tool_answer_mode,
                context_message_limit=8 if tool_answer_mode else None,
                candidate_count=reply_candidate_limit,
                prompt_flow=prompt_flow,
                task_name=task_name,
                speaker_context=effective_speaker_context,
            )
        except Exception as exc:
            logger.warning(
                "qq_social_agent reply candidate generation failed: "
                f"group={group_id} addressed={addressed_bot} error={exc}"
            )
            return
        if not reply_candidates:
            if direct_single_reply:
                logger.info(
                    "qq_social_agent skipped group reply: "
                    f"group={group_id} reason=empty_model_reply_direct_single addressed={addressed_bot}"
                )
                record_metric_event(
                    "reply_suppressed",
                    group_id=group_id,
                    user_id=user_id,
                    stage="generation",
                    action="empty_model_reply_direct_single",
                    addressed=addressed_bot,
                )
                return
            logger.info(f"qq_social_agent skipped group={group_id}: empty_model_reply")
            return
        approval_candidates = build_approval_candidates(
            reply_candidates,
            market_report=market_report,
            limit=reply_candidate_limit,
            allow_questions=addressed_bot,
        )
        if not approval_candidates:
            logger.info(f"qq_social_agent skipped group={group_id}: empty_candidate_after_guard")
            return
        judged_pronoun, judged_critic = None, None
        if client is not None:
            judged_pronoun, judged_critic = await client.review_draft(
                draft=approval_candidates[0].text,
                current_text=text,
                current_label=current_label,
                action=decision.action,
                speaker_context=speaker_context,
                recent_messages=recent_messages,
                memory_context=memory_context,
                tool_context=combine_text_sections(fresh_context, market_context),
                reference=reference_resolution,
                ellipsis=ellipsis_resolution,
                repair=repair_resolution,
                discourse=discourse_state,
            )
        pronoun_result = apply_jev_pronoun_judgement(
            judged_pronoun,
            has_pronoun=draft_has_person_pronoun(approval_candidates[0].text),
        )
        record_metric_event(
            "pronoun_guard",
            group_id=group_id,
            user_id=user_id,
            stage="pronoun",
            action="fix" if pronoun_result.needs_fix else "pass",
            pronoun_status=pronoun_result.status,
            pronoun_issue=pronoun_result.issue,
            pronoun_noul=pronoun_result.noul,
            attempt=attempt,
        )
        if pronoun_result.needs_fix and attempt <= 0:
            critic_feedback = combine_text_sections(
                format_pronoun_feedback(pronoun_result),
                "【待修人称原草稿】\n" + approval_candidates[0].text,
            )
            continue
        critic_result = apply_jev_critic_judgement(judged_critic)
        if draft_violates_media_gate(approval_candidates[0].text, discourse_state):
            failures = tuple(dict.fromkeys([*critic_result.failures, "context_consistent"]))
            critic_result = CriticResult(
                intent_covered=critic_result.intent_covered or "YES",
                referent_consistent=critic_result.referent_consistent or "YES",
                context_consistent="NO",
                unsupported_claim=critic_result.unsupported_claim or "NO",
                reason="media_gate",
                status=RESOLVED,
                source=critic_result.source,
                failures=failures,
                regenerated=attempt > 0,
            )
        critic_result = replace(critic_result, regenerated=attempt > 0)
        regenerated = attempt > 0
        action = next_critic_action(
            critic_result, attempt=attempt, addressed=addressed_bot,
        )
        if action == "regenerate" and reply_budget.skip("critic_retry", time.monotonic()):
            action = "send"
            record_metric_event(
                "reply_budget",
                group_id=group_id,
                user_id=user_id,
                stage="critic",
                action="skip_critic_retry",
                remaining_ms=int(reply_budget.remaining(time.monotonic()) * 1000),
            )
        record_metric_event(
            "pre_send_critic",
            group_id=group_id,
            user_id=user_id,
            stage="critic",
            action=action,
            critic_status=critic_result.status,
            critic_failed=critic_result.failed,
            critic_unavailable=list(critic_result.unavailable),
            critic_failures=list(critic_result.failures),
            regenerated=regenerated,
            attempt=attempt,
        )
        if action != "regenerate":
            break
        critic_feedback = format_critic_feedback(critic_result)
        regenerated = True
    generation_elapsed_ms = int((time.monotonic() - generation_started_at) * 1000)
    return GeneratedGroupReply(
        decision=decision,
        candidates=tuple(approval_candidates),
        critic=critic_result,
        regenerated=regenerated,
        direct_single_reply=direct_single_reply,
        prompt_flow=prompt_flow,
        elapsed_ms=generation_elapsed_ms,
    )
