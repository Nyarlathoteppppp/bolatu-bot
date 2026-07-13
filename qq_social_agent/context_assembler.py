from __future__ import annotations

import re

from .pipeline_types import ContextPacket, ContextSection


STRUCTURED_RAG_TYPES = frozenset({"memory_atom", "member"})


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
) -> ContextPacket:
    """Build one typed context packet and suppress only duplicate evidence paths.

    Memory is not removed. Exact duplicate sections are collapsed, while RAG,
    profiles and memory atoms remain separate evidence categories so one RAG hit
    cannot accidentally hide other relevant people or facts.
    """

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
    sections: list[ContextSection] = []
    seen: set[str] = set()
    for section in candidates:
        content = section.content.strip()
        if not content:
            continue
        fingerprint = _content_fingerprint(content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        sections.append(ContextSection(section.name, content, section.source, section.priority))
    sections.sort(key=lambda item: item.priority, reverse=True)
    return ContextPacket(
        sections=tuple(sections),
        rag_document_ids=rag_document_ids,
        rag_document_types=rag_document_types,
    )


def _content_fingerprint(content: str) -> str:
    return re.sub(r"[\W_]+", "", content, flags=re.UNICODE).casefold()
