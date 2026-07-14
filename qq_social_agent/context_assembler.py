from __future__ import annotations

import re

from .pipeline_types import ContextPacket, ContextSection, PipelineMode


STRUCTURED_RAG_TYPES = frozenset({"memory_atom", "member"})

_MODE_SECTION_BUDGETS: dict[PipelineMode, dict[str, int]] = {
    PipelineMode.CHAT: {
        "memory": 2200,
        "memory_atoms": 1200,
        "member": 1000,
        "jargon": 700,
        "recall_feedback": 700,
        "positive_feedback": 600,
        "style": 700,
        "raw_corpus": 1500,
    },
    # Fresh facts must not be overridden by old chat summaries or style
    # examples. The recent message window and the fresh fact pack are supplied
    # separately by the generation flow; only matched group jargon is useful.
    PipelineMode.SEARCH: {"jargon": 500},
    PipelineMode.MARKET: {"jargon": 500},
    PipelineMode.DEEP_URL: {"jargon": 500},
}

_MODE_TOTAL_BUDGETS: dict[PipelineMode, int] = {
    PipelineMode.CHAT: 6500,
    PipelineMode.SEARCH: 500,
    PipelineMode.MARKET: 500,
    PipelineMode.DEEP_URL: 500,
}


def assemble_generation_context(
    *,
    memory_context: str = "",
    member_context: str = "",
    memory_atoms_context: str = "",
    style_context: str = "",
    raw_corpus_context: str = "",
    jargon_context: str = "",
    recall_feedback_context: str = "",
    positive_feedback_context: str = "",
    rag_document_ids: tuple[int, ...] = (),
    rag_document_types: tuple[str, ...] = (),
    mode: PipelineMode | str = PipelineMode.CHAT,
) -> ContextPacket:
    """Build a typed, mode-aware context packet.

    Chat mode keeps separately ranked memory evidence. Tool-answer modes exclude
    old conversational evidence so fresh facts cannot be overwritten by a stale
    summary, profile, style example, or raw quote.
    """

    normalized_mode = _pipeline_mode(mode)
    section_budgets = _MODE_SECTION_BUDGETS[normalized_mode]
    total_budget = _MODE_TOTAL_BUDGETS[normalized_mode]
    candidates = [
        ContextSection("memory", memory_context, "rag_or_summary", 90),
        ContextSection("member", member_context, "member_selector", 80),
        ContextSection("memory_atoms", memory_atoms_context, "memory_atom_selector", 85),
        ContextSection("style", style_context, "style_rules", 55),
        ContextSection("raw_corpus", raw_corpus_context, "raw_corpus", 50),
        ContextSection("jargon", jargon_context, "jargon", 75),
        ContextSection("recall_feedback", recall_feedback_context, "approval_feedback", 70),
        ContextSection("positive_feedback", positive_feedback_context, "approval_feedback", 65),
    ]
    candidates.sort(key=lambda item: item.priority, reverse=True)
    sections: list[ContextSection] = []
    dropped: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for section in candidates:
        content = section.content.strip()
        if not content:
            continue
        section_budget = section_budgets.get(section.name, 0)
        if section_budget <= 0:
            dropped.append(section.name)
            continue
        fingerprint = _content_fingerprint(content)
        if fingerprint in seen:
            dropped.append(f"{section.name}:duplicate")
            continue
        seen.add(fingerprint)
        remaining = max(0, total_budget - used_chars)
        allowed = min(section_budget, remaining)
        if allowed <= 0:
            dropped.append(f"{section.name}:budget")
            continue
        trimmed = _trim_context(content, allowed)
        if len(trimmed) < len(content):
            dropped.append(f"{section.name}:truncated")
        sections.append(ContextSection(section.name, trimmed, section.source, section.priority))
        used_chars += len(trimmed)
    sections.sort(key=lambda item: item.priority, reverse=True)
    return ContextPacket(
        mode=normalized_mode,
        sections=tuple(sections),
        rag_document_ids=rag_document_ids,
        rag_document_types=rag_document_types,
        dropped_sections=tuple(dropped),
    )


def _content_fingerprint(content: str) -> str:
    return re.sub(r"[\W_]+", "", content, flags=re.UNICODE).casefold()


def _pipeline_mode(mode: PipelineMode | str) -> PipelineMode:
    if isinstance(mode, PipelineMode):
        return mode
    try:
        return PipelineMode(str(mode).strip().casefold())
    except ValueError:
        return PipelineMode.CHAT


def _trim_context(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    if limit <= 1:
        return content[:limit]
    return content[: limit - 1].rstrip() + "…"
