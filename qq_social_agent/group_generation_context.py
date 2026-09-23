from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .context_assembler import assemble_generation_context, merge_rag_and_summary_context
from .member_context import (
    format_member_context,
    is_self_memory_query,
    member_memory_user_ids,
)
from .memory import ChatMessage, MemoryStore
from .pipeline_types import ContextPacket, PipelineMode
from .rag_retriever import RAGRetrievalResult, RAGService
from .reference_resolver import ReferenceResolution
from .social_actions import SocialActionService
from .jev_policy import GroupReplyBudget


@dataclass(frozen=True)
class GroupContextLimits:
    keep_summaries: int
    summary_appendix_chars: int
    member_impressions: int
    memory_atoms: int
    style_rules: int
    raw_corpus: int
    raw_corpus_candidates: int
    raw_corpus_radius: int
    recall_feedback: int
    positive_feedback: int


async def build_group_generation_context(
    *,
    group_id: int,
    user_id: int,
    nickname: str,
    text: str,
    self_id: int,
    context_query: str,
    recent_messages: list[ChatMessage],
    reference_resolution: ReferenceResolution,
    addressed_bot: bool,
    mode: PipelineMode,
    reply_budget: GroupReplyBudget,
    memory: MemoryStore,
    rag_service: RAGService,
    social_action_service: SocialActionService,
    limits: GroupContextLimits,
    select_jargon_context: Callable[..., Awaitable[str]],
    format_memory_context: Callable[..., str],
    format_memory_atom_context: Callable[..., str],
    format_style_context: Callable[..., str],
    format_raw_corpus_context: Callable[..., str],
    format_recall_feedback_context: Callable[..., str],
    format_positive_feedback_context: Callable[..., str],
    record_metric_event: Callable[..., None],
) -> ContextPacket:
    memory_context = ""
    member_context = ""
    memory_atoms_context = ""
    style_context = ""
    jargon_context = ""
    recall_feedback_context = ""
    positive_feedback_context = ""
    rag_result: RAGRetrievalResult | None = None
    rag_context_applied = False
    if not memory_context:
        memory_context = format_memory_context(
            memory.relevant_memory_summaries(
                group_id,
                f"{nickname}\n{text}" if is_self_memory_query(text) else context_query,
                limit=limits.keep_summaries,
            )
        )
    if not rag_context_applied and is_self_memory_query(text):
        rag_context_applied = True
        record_metric_event(
            "rag_retrieval",
            group_id=group_id,
            user_id=user_id,
            stage="generation_context",
            action="skip_self_memory_rag",
        )
    if not rag_context_applied and reply_budget.skip("optional_rag", time.monotonic()):
        rag_context_applied = True
        record_metric_event(
            "reply_budget",
            group_id=group_id,
            user_id=user_id,
            stage="generation_context",
            action="skip_optional_rag",
            remaining_ms=int(reply_budget.remaining(time.monotonic()) * 1000),
        )
    if not rag_context_applied:
        rag_result = await rag_service.retrieve(
            group_id=group_id,
            query=reference_resolution.expanded_query or text,
            addressed=addressed_bot,
            related_user_ids=list(reference_resolution.user_ids),
            excluded_user_ids=[self_id],
        )
        rag_context_applied = True
        summary_context = memory_context
        memory_context = merge_rag_and_summary_context(
            rag_result.context,
            summary_context,
            summary_char_limit=limits.summary_appendix_chars,
        )
        record_metric_event(
            "rag_retrieval",
            group_id=group_id,
            user_id=user_id,
            stage="generation_context",
            action="merged" if rag_result.context and summary_context else (
                "injected" if rag_result.context else "empty"
            ),
            route=rag_result.plan.route,
            lexical_count=rag_result.lexical_count,
            semantic_count=rag_result.semantic_count,
            injected_count=len(rag_result.hits),
            elapsed_ms=rag_result.elapsed_ms,
            error=rag_result.error,
            resolved_user_ids=[item.user_id for item in rag_result.resolved_members],
            resolved_names=[item.matched_name for item in rag_result.resolved_members],
            reference_reason=reference_resolution.reason,
            reference_confidence=reference_resolution.confidence,
            normalized_query=rag_result.normalized_query,
            focused_topic=rag_result.focused_topic,
            reply_envelope_removed=rag_result.reply_envelope_removed,
            hit_document_ids=[hit.document.id for hit in rag_result.hits],
            hit_doc_types=[hit.document.doc_type for hit in rag_result.hits],
            hit_scores=[round(hit.score, 4) for hit in rag_result.hits],
            hit_reasons=[list(hit.reasons) for hit in rag_result.hits],
            hit_sources=[f"{hit.document.source_name}:{hit.document.source_row_id}" for hit in rag_result.hits],
        )
    memory_focus_ids = member_memory_user_ids(
        recent_messages,
        current_user_id=user_id,
        current_text=text,
    )
    self_memory_query = is_self_memory_query(text)
    memory_focus_query = f"{nickname}\n{text}" if self_memory_query else context_query
    if not member_context:
        member_context = format_member_context(
            memory.member_impressions_for_context(
                group_id,
                memory_focus_ids,
                limit=limits.member_impressions,
            ),
            current_user_id=user_id,
        )
    if not memory_atoms_context:
        memory_atoms_context = format_memory_atom_context(
            memory.relevant_memory_atoms(
                group_id,
                memory_focus_query,
                subject_user_ids=memory_focus_ids,
                speaker_user_id=user_id,
                relationship_user_ids=memory_focus_ids,
                limit=limits.memory_atoms,
            )
        )
    if not style_context:
        style_context = format_style_context(
            memory.relevant_style_rules(
                group_id,
                context_query,
                limit=limits.style_rules,
                speaker_user_id=user_id,
            )
        )
    raw_corpus_context = format_raw_corpus_context(
        memory.relevant_raw_corpus_examples(
            group_id,
            context_query,
            limit=limits.raw_corpus,
            candidate_limit=limits.raw_corpus_candidates,
            context_radius=limits.raw_corpus_radius,
            exclude_user_id=user_id,
            exclude_text=text,
            preferred_user_id=user_id,
            preferred_limit=2,
            preferred_score_multiplier=1.1,
            preferred_score_bonus=0.5,
            per_user_limit=1,
        )
    )
    if not jargon_context:
        jargon_context = await select_jargon_context(
            group_id,
            recent_messages,
            current_text=text,
            current_nickname=nickname,
        )
    recall_feedback_context = format_recall_feedback_context(
        memory.recent_recalled_reply_feedback(group_id, limits.recall_feedback)
    )
    positive_feedback_context = format_positive_feedback_context(
        memory.recent_approved_reply_feedback(group_id, limits.positive_feedback)
    )
    social_action_context = social_action_service.recent_reaction_context(group_id)
    context_packet = assemble_generation_context(
        memory_context=memory_context,
        member_context=member_context,
        memory_atoms_context=memory_atoms_context,
        style_context=style_context,
        raw_corpus_context=raw_corpus_context,
        jargon_context=jargon_context,
        recall_feedback_context=recall_feedback_context,
        positive_feedback_context=positive_feedback_context,
        social_action_context=social_action_context,
        rag_document_ids=tuple(
            hit.document.id for hit in rag_result.hits
        ) if rag_result is not None else (),
        rag_document_types=tuple(
            hit.document.doc_type for hit in rag_result.hits
        ) if rag_result is not None else (),
        mode=mode,
    )
    record_metric_event(
        "context_assembled",
        group_id=group_id,
        user_id=user_id,
        stage="generation_context",
        action="ready",
        section_names=[section.name for section in context_packet.sections],
        section_chars={section.name: len(section.content) for section in context_packet.sections},
        pipeline_mode=context_packet.mode.value,
        dropped_sections=list(context_packet.dropped_sections),
        rag_document_ids=list(context_packet.rag_document_ids),
        rag_document_types=list(context_packet.rag_document_types),
    )

    return context_packet
