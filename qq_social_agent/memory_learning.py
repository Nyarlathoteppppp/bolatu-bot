from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Iterable

from .deepseek_client import DailyReviewDraft, MemoryFactDraft, MidMemoryDraft
from .memory import ChatMessage, MemoryStore


_ATOM_TYPE_BY_KIND = {
    "event": "event",
    "fact": "fact",
    "preference": "preference",
    "relationship": "relation",
    "relation": "relation",
    "identity": "identity",
    "promise": "promise",
    "recurring_behavior": "behavior",
    "member_delta": "profile_delta",
    "jargon_candidate": "jargon_candidate",
    "open_thread": "open_thread",
    "feedback_lesson": "feedback",
    "style_observation": "style",
}


def persist_mid_memory_learning(
    memory: MemoryStore,
    *,
    group_id: int,
    draft: MidMemoryDraft,
    messages: list[ChatMessage],
) -> tuple[int, ...]:
    facts = (
        *draft.facts,
        *draft.member_deltas,
        *draft.jargon_candidates,
        *draft.open_threads,
    )
    return persist_fact_drafts(
        memory,
        group_id=group_id,
        facts=facts,
        source_prefix="mid_summary",
        messages=messages,
        require_message_evidence=True,
    )


def persist_daily_review_learning(
    memory: MemoryStore,
    *,
    group_id: int,
    review_label: str,
    draft: DailyReviewDraft,
    messages: list[ChatMessage],
) -> tuple[int, ...]:
    message_facts = (*draft.events, *draft.member_changes, *draft.jargon_candidates)
    learned = list(
        persist_fact_drafts(
            memory,
            group_id=group_id,
            facts=message_facts,
            source_prefix=f"daily_review:{review_label}",
            messages=messages,
            require_message_evidence=True,
        )
    )
    learned.extend(
        persist_fact_drafts(
            memory,
            group_id=group_id,
            facts=(*draft.feedback_lessons, *draft.style_observations),
            source_prefix=f"daily_review_event:{review_label}",
            messages=messages,
            require_message_evidence=False,
        )
    )
    snapshot = {
        "review_label": review_label,
        "public_reply": draft.public_reply,
        "events": [asdict(item) for item in draft.events],
        "member_changes": [asdict(item) for item in draft.member_changes],
        "jargon_candidates": [asdict(item) for item in draft.jargon_candidates],
        "feedback_lessons": [asdict(item) for item in draft.feedback_lessons],
        "style_observations": [asdict(item) for item in draft.style_observations],
        "memory_atom_ids": learned,
        "saved_at": time.time(),
    }
    memory.app_kv_set(
        f"daily_review_structured:{group_id}:{review_label}",
        json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
    )
    return tuple(learned)


def persist_fact_drafts(
    memory: MemoryStore,
    *,
    group_id: int,
    facts: Iterable[MemoryFactDraft],
    source_prefix: str,
    messages: list[ChatMessage],
    require_message_evidence: bool,
    minimum_confidence: float = 0.55,
) -> tuple[int, ...]:
    by_id = {message.id: message for message in messages if message.id > 0}
    atom_ids: list[int] = []
    for fact in facts:
        evidence_ids = tuple(message_id for message_id in fact.evidence_message_ids if message_id in by_id)
        if require_message_evidence and not evidence_ids:
            continue
        if fact.confidence < minimum_confidence:
            continue
        observed_at = max(
            (by_id[message_id].created_at for message_id in evidence_ids),
            default=time.time(),
        )
        valid_to = None
        if fact.valid_for_days is not None:
            valid_to = observed_at + max(1, fact.valid_for_days) * 24 * 60 * 60
        evidence_label = ",".join(str(item) for item in evidence_ids[:4])
        source = f"{source_prefix}:{evidence_label}" if evidence_label else source_prefix
        atom_id = memory.upsert_memory_atom(
            atom_type=_ATOM_TYPE_BY_KIND.get(fact.kind, "note"),
            group_id=group_id,
            subject_user_id=fact.subject_user_id,
            object_user_id=fact.object_user_id,
            content=fact.content,
            source=source[:80],
            evidence_type="message" if evidence_ids else "event",
            source_message_id=f"db:{evidence_ids[0]}" if evidence_ids else None,
            observed_at=observed_at,
            valid_from=observed_at,
            valid_to=valid_to,
            confidence=fact.confidence,
            importance=fact.importance,
        )
        if atom_id and atom_id not in atom_ids:
            atom_ids.append(atom_id)
    return tuple(atom_ids)
