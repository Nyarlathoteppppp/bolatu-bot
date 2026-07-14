from __future__ import annotations

from dataclasses import dataclass

from .reply_splitter import split_reply_messages


@dataclass(frozen=True)
class DeliveryPlan:
    parts: tuple[str, ...]
    mention_targets: dict[int, str]
    sequence_lag: int
    forced_trigger_mention: bool


def build_delivery_plan(
    *,
    reply_text: str,
    mention_targets: dict[int, str],
    trigger_user_id: int,
    trigger_nickname: str,
    trigger_sequence: int,
    current_sequence: int,
    max_messages: int = 3,
) -> DeliveryPlan:
    effective_targets = dict(mention_targets)
    sequence_lag = max(0, current_sequence - trigger_sequence)
    force_mention = trigger_sequence > 0 and sequence_lag >= 3
    prepared_text = reply_text
    if force_mention:
        effective_targets[trigger_user_id] = (
            trigger_nickname.strip() or str(trigger_user_id)
        )[:24]
        marker = f"[[at:{trigger_user_id}]]"
        if marker not in prepared_text:
            prepared_text = f"{marker} {prepared_text}".strip()
    return DeliveryPlan(
        parts=tuple(split_reply_messages(prepared_text, max_messages=max_messages)),
        mention_targets=effective_targets,
        sequence_lag=sequence_lag,
        forced_trigger_mention=force_mention,
    )
