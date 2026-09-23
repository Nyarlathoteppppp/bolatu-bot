from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .memory import MemoryStore
from .private_message_types import PrivateGenerationContext, PrivateToolStage


@dataclass(frozen=True)
class PrivateGenerationContextServices:
    memory: MemoryStore
    format_memory_context: Callable[[list[Any]], str]
    merge_rag_and_summary_context: Callable[..., str]
    member_memory_user_ids: Callable[..., list[int]]
    is_self_memory_query: Callable[[str], bool]
    format_member_context: Callable[..., str]
    format_memory_atom_context: Callable[[list[Any]], str]
    format_style_context: Callable[[list[Any]], str]
    format_raw_corpus_context: Callable[[list[Any]], str]
    selected_group_jargon_context: Callable[..., Awaitable[str]]
    format_recall_feedback_context: Callable[[list[Any]], str]
    private_priority_context: Callable[[int], str]
    combine_text_sections: Callable[..., str]
    record_metric_event: Callable[..., None]
    mid_memory_keep_summaries: int
    mid_memory_summary_appendix_chars: int
    member_impression_context_limit: int
    memory_atom_context_limit: int
    style_rule_context_limit: int
    raw_corpus_context_limit: int
    raw_corpus_candidate_limit: int
    raw_corpus_context_radius: int
    recall_feedback_context_limit: int


async def build_private_generation_context(
    stage: PrivateToolStage,
    *,
    services: PrivateGenerationContextServices,
) -> PrivateGenerationContext:
    turn = stage.turn
    rag_result = await stage.rag_task
    summary_context = services.format_memory_context(
        services.memory.relevant_memory_summaries(
            turn.chat_id,
            stage.context_query,
            limit=services.mid_memory_keep_summaries,
        )
    )
    memory_context = services.merge_rag_and_summary_context(
        rag_result.context,
        summary_context,
        summary_char_limit=services.mid_memory_summary_appendix_chars,
    )
    related_user_ids = services.member_memory_user_ids(
        stage.context_recent,
        current_user_id=turn.user_id,
        current_text=turn.text,
    )
    member_context = services.format_member_context(
        services.memory.member_impressions_for_context(
            turn.chat_id,
            related_user_ids,
            limit=services.member_impression_context_limit,
        ),
        current_user_id=turn.user_id,
    )
    atom_query = f"{turn.nickname}\n{turn.text}" if services.is_self_memory_query(turn.text) else stage.context_query
    memory_atoms_context = services.format_memory_atom_context(
        services.memory.relevant_memory_atoms(
            turn.chat_id,
            atom_query,
            subject_user_ids=related_user_ids,
            speaker_user_id=turn.user_id,
            relationship_user_ids=related_user_ids,
            limit=services.memory_atom_context_limit,
        )
    )
    style_context = services.format_style_context(
        services.memory.relevant_style_rules(
            turn.chat_id,
            stage.context_query,
            limit=services.style_rule_context_limit,
            speaker_user_id=turn.user_id,
        )
    )
    raw_corpus_context = services.format_raw_corpus_context(
        services.memory.relevant_raw_corpus_examples(
            turn.chat_id,
            stage.context_query,
            limit=services.raw_corpus_context_limit,
            candidate_limit=services.raw_corpus_candidate_limit,
            context_radius=services.raw_corpus_context_radius,
            exclude_user_id=turn.user_id,
            exclude_text=turn.text,
            preferred_user_id=turn.user_id,
            preferred_limit=2,
            preferred_score_multiplier=1.1,
            preferred_score_bonus=0.5,
            per_user_limit=1,
        )
    )
    jargon_context = await services.selected_group_jargon_context(
        turn.chat_id,
        list(stage.context_recent),
        current_text=stage.context_query,
        current_nickname=turn.nickname,
        chat_label="QQ 私聊",
    )
    recall_feedback_context = services.format_recall_feedback_context(
        services.memory.recent_recalled_reply_feedback(turn.chat_id, services.recall_feedback_context_limit)
    )
    services.record_metric_event(
        "rag_retrieval",
        group_id=turn.chat_id,
        user_id=turn.user_id,
        stage="private_generation_context",
        action="injected" if rag_result.context else "empty",
        route=rag_result.plan.route,
        lexical_count=rag_result.lexical_count,
        semantic_count=rag_result.semantic_count,
        injected_count=len(rag_result.hits),
        elapsed_ms=rag_result.elapsed_ms,
        error=rag_result.error,
    )
    priority_context = services.combine_text_sections(
        services.private_priority_context(turn.user_id),
        stage.private_state_context,
        turn.forced_once_context,
    )
    return PrivateGenerationContext(
        persona=stage.persona,
        recent_messages=stage.context_recent,
        current_text=turn.text,
        current_nickname=turn.nickname,
        decision=stage.decision,
        market_context=stage.market_context,
        fresh_context=stage.fresh_context,
        memory_context=memory_context,
        member_context=member_context,
        memory_atoms_context=memory_atoms_context,
        style_context=style_context,
        raw_corpus_context=raw_corpus_context,
        jargon_context=jargon_context,
        recall_feedback_context=recall_feedback_context,
        speaker_context=stage.speaker_context,
        priority_context=priority_context,
    )
