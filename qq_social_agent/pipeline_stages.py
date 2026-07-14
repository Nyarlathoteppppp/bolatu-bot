from __future__ import annotations

from typing import Iterable

from .pipeline_types import (
    ContextPacket,
    GeneratedCandidate,
    OutputChannel,
    PipelineStage,
    PipelineState,
    SocialIntent,
)


def mark_gated(state: PipelineState) -> None:
    state.transition(PipelineStage.GATED)


def mark_understood(state: PipelineState) -> None:
    state.transition(PipelineStage.UNDERSTOOD)


def apply_decision(
    state: PipelineState,
    *,
    should_reply: bool,
    action: str,
    reason: str,
    confidence: float,
    elapsed_ms: int | None = None,
) -> None:
    state.decision_action = action
    state.decision_reason = reason
    state.decision_confidence = confidence
    if not should_reply:
        state.output_channel = OutputChannel.SILENT
    elif action == "react":
        state.output_channel = OutputChannel.REACT
    elif action == "poke":
        state.output_channel = OutputChannel.POKE
    else:
        state.output_channel = OutputChannel.TEXT
    state.social_intent = {
        "answer": SocialIntent.ANSWER,
        "care": SocialIntent.CARE,
        "tease": SocialIntent.PLAY,
        "agree": SocialIntent.AGREE,
    }.get(action, SocialIntent.CHAT)
    state.transition(PipelineStage.DECIDED, elapsed_ms=elapsed_ms)


def apply_context(state: PipelineState, packet: ContextPacket, *, elapsed_ms: int | None = None) -> None:
    state.context = packet
    state.transition(PipelineStage.CONTEXT_READY, elapsed_ms=elapsed_ms)


def apply_candidates(state: PipelineState, candidates: Iterable[object], *, elapsed_ms: int | None = None) -> None:
    state.candidates = tuple(
        GeneratedCandidate(
            int(getattr(candidate, "index")),
            str(getattr(candidate, "text")),
            str(getattr(candidate, "action")),
            str(getattr(candidate, "style")),
        )
        for candidate in candidates
    )
    state.transition(PipelineStage.GENERATED, elapsed_ms=elapsed_ms)


def mark_approval_pending(state: PipelineState, approval_id: str) -> None:
    state.approval_id = approval_id
    state.transition(PipelineStage.APPROVAL_PENDING)


def mark_sending(state: PipelineState) -> None:
    state.failure = ""
    state.transition(PipelineStage.SENDING)


def mark_sent(state: PipelineState, message_id: int | str | None) -> None:
    state.add_sent_message(message_id)


def mark_completed(state: PipelineState, *, elapsed_ms: int | None = None) -> None:
    state.transition(PipelineStage.COMPLETED, elapsed_ms=elapsed_ms)


def mark_failed(state: PipelineState, reason: str) -> None:
    state.fail(reason)
