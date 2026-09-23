from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .deepseek_client import DeepSeekClient
from .discourse_effects import (
    MemoryCandidate,
    MemoryEffectResolution,
    apply_jev_memory_judgement,
    related_memories_for_candidate,
    should_ask_jev_memory_effect,
)
from .discourse_state import DiscourseState, discourse_decision_trace, resolve_group_discourse
from .member_context import related_member_user_ids as _related_member_user_ids
from .memory import ChatMessage, MemoryStore
from .rag_retriever import RAGService
from .reference_resolver import ReplyHint
from .resolver_result import AMBIGUOUS, ERROR, NOT_APPLICABLE, RESOLVED, UNAVAILABLE
from .speaker_context import _format_speaker_reference_context, _message_relation_facts


@dataclass(frozen=True)
class GroupDiscourseContext:
    state: DiscourseState
    speaker_context: str
    memory_effect: MemoryEffectResolution
    memory_candidate: MemoryCandidate | None
    invalidated_layers: list[str]
    recomputed_layers: list[str]


async def resolve_group_discourse_context(
    *,
    group_id: int,
    user_id: int,
    nickname: str,
    text: str,
    normalized_text: str,
    self_id: int,
    recent_messages: list[ChatMessage],
    reply_hint: ReplyHint,
    at_user_ids: tuple[int, ...],
    current_has_media: bool,
    reply_has_media: bool,
    mentioned: bool,
    replied_to_bot: bool,
    addressed_bot: bool,
    followup_addressed: bool,
    followup_soft: bool,
    source_message_id: str,
    client: DeepSeekClient,
    memory: MemoryStore,
    rag_service: RAGService,
    record_metric_event: Callable[..., None],
    logger: Any,
) -> GroupDiscourseContext:
    related_member_user_ids = _related_member_user_ids(recent_messages, current_user_id=user_id)
    named_resolver = lambda candidate: rag_service.resolve_named_user_ids(group_id, candidate)
    discourse_state = await resolve_group_discourse(
        current_text=normalized_text,
        current_user_id=user_id,
        current_nickname=nickname,
        self_id=self_id,
        recent_messages=recent_messages,
        reply=reply_hint,
        at_user_ids=at_user_ids,
        named_resolver=named_resolver,
        jev=client,
        current_has_media=current_has_media,
        reply_has_media=reply_has_media,
        relation_user_ids=related_member_user_ids,
    )
    reference_resolution = discourse_state.reference
    ellipsis_resolution = discourse_state.ellipsis
    repair_resolution = discourse_state.repair
    ambiguity_resolution = discourse_state.ambiguity_resolution
    invalidated_layers = list(discourse_state.invalidated_layers)
    recomputed_layers = list(discourse_state.recomputed_layers)
    memory_effect_resolution = MemoryEffectResolution()
    if invalidated_layers:
        logger.info(
            "qq_social_agent repair invalidation: "
            f"group={group_id} kind={repair_resolution.kind} "
            f"target={repair_resolution.target_key} layers={','.join(invalidated_layers)} "
            f"recomputed={','.join(recomputed_layers)}"
        )
    memory_candidate = None
    related_memories: list = []
    subject_id = None
    if "referent" in invalidated_layers and "referent" not in recomputed_layers:
        recomputed_layers.append("referent")
    blocked_memory_statuses = {AMBIGUOUS, UNAVAILABLE, ERROR}
    if (
        reference_resolution.status not in blocked_memory_statuses
        and repair_resolution.status not in blocked_memory_statuses
    ):
        if (
            reference_resolution.status == RESOLVED
            and reference_resolution.kind == "PERSON"
            and reference_resolution.user_ids
        ):
            subject_id = reference_resolution.user_ids[0]
        elif reference_resolution.kind in {"", "NONE", "NON_PERSON"} or reference_resolution.status == NOT_APPLICABLE:
            subject_id = user_id
    fact_like = repair_resolution.kind in {"FACT", "RETRACTION"} and repair_resolution.status == RESOLVED
    if subject_id is not None and (
        fact_like
        or should_ask_jev_memory_effect(
            normalized_text,
            repair=repair_resolution,
            candidate=MemoryCandidate(subject_user_id=subject_id, content=normalized_text),
        )
    ):
        memory_candidate = MemoryCandidate(
            subject_user_id=subject_id,
            content=normalized_text.strip()[:180],
            source_message_id=source_message_id,
            speaker=nickname,
        )
    if should_ask_jev_memory_effect(
        normalized_text,
        repair=repair_resolution,
        candidate=memory_candidate,
    ) and memory_candidate is not None:
        related_memories = related_memories_for_candidate(
            memory,
            group_id=group_id,
            candidate=memory_candidate,
            speaker_user_id=user_id,
        )
        judged_memory = None
        if client is not None:
            judged_memory = await client.resolve_memory_effect(
                current_text=normalized_text,
                candidate=memory_candidate,
                related=related_memories,
            )
        memory_effect_resolution = apply_jev_memory_judgement(
            judged_memory,
            related_memories,
            candidate=memory_candidate,
        )
        if "memory" in invalidated_layers and "memory" not in recomputed_layers:
            recomputed_layers.append("memory")
    relation_facts = _message_relation_facts(
        current_user_id=user_id,
        current_nickname=nickname,
        current_text=text,
        reference_resolution=reference_resolution,
        mentioned=mentioned,
        replied_to_bot=replied_to_bot,
        addressed_bot=addressed_bot,
        followup_addressed=followup_addressed,
        self_id=self_id,
    )
    speaker_context = _format_speaker_reference_context(
        current_user_id=user_id,
        current_nickname=nickname,
        current_text=text,
        recent_messages=recent_messages,
        reference_resolution=reference_resolution,
        mentioned=mentioned,
        replied_to_bot=replied_to_bot,
        addressed_bot=addressed_bot,
        followup_addressed=followup_addressed,
        followup_soft=followup_soft,
        self_id=self_id,
        relation_facts=relation_facts,
        ellipsis_resolution=ellipsis_resolution,
        repair_resolution=repair_resolution,
        ambiguity_resolution=ambiguity_resolution,
        discourse_state=discourse_state,
    )
    record_metric_event(
        "discourse_decision_trace",
        group_id=group_id,
        user_id=user_id,
        stage="discourse",
        action=discourse_state.state_audit,
        **discourse_decision_trace(discourse_state),
    )
    record_metric_event(
        "message_relation",
        group_id=group_id,
        user_id=user_id,
        stage="relation",
        action=relation_facts.target_scope,
        target_note=relation_facts.target_note,
        reply_target=relation_facts.reply_target_label,
        reply_target_is_bot=relation_facts.reply_target_is_bot,
        ambiguous_reference=relation_facts.ambiguous_reference,
        reference_user_ids=list(relation_facts.reference_user_ids),
        reference_reason=relation_facts.reference_reason,
        reference_confidence=relation_facts.reference_confidence,
        reference_status=reference_resolution.status,
        reference_source=reference_resolution.source,
        reference_value=reference_resolution.value,
        ellipsis_kind=ellipsis_resolution.kind,
        ellipsis_status=ellipsis_resolution.status,
        ellipsis_source=ellipsis_resolution.source,
        ellipsis_value=ellipsis_resolution.value,
        ellipsis_unresolved=ellipsis_resolution.unresolved,
        ellipsis_reason=ellipsis_resolution.reason,
        repair_kind=repair_resolution.kind,
        repair_status=repair_resolution.status,
        repair_source=repair_resolution.source,
        repair_value=repair_resolution.value,
        repair_target=repair_resolution.target_key,
        repair_unresolved=repair_resolution.unresolved,
        repair_reason=repair_resolution.reason,
        repair_invalidates=list(repair_resolution.invalidates),
        invalidated_states=list(invalidated_layers),
        recomputed_states=list(recomputed_layers),
        ambiguity_kind=ambiguity_resolution.kind,
        ambiguity_status=ambiguity_resolution.status,
        ambiguity_source=ambiguity_resolution.source,
        ambiguity_unresolved=ambiguity_resolution.unresolved,
        memory_action=memory_effect_resolution.action,
        memory_status=memory_effect_resolution.status,
        memory_source=memory_effect_resolution.source,
        memory_unresolved=memory_effect_resolution.unresolved,
        memory_applied=memory_effect_resolution.applied,
    )
    return GroupDiscourseContext(
        state=discourse_state,
        speaker_context=speaker_context,
        memory_effect=memory_effect_resolution,
        memory_candidate=memory_candidate,
        invalidated_layers=invalidated_layers,
        recomputed_layers=recomputed_layers,
    )
