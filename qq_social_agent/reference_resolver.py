from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable


PRONOUN_RE = re.compile(r"(?:^|[，。！？\s])(他|她|这个人|那个人|这位群友|那位群友)(?:的|呢|之前|以前|过去|现在|后来|说|做|想|喜|$)")
ELLIPTICAL_FOLLOWUP_RE = re.compile(
    r"^(?:那|然后|所以|那么)?(?:后来|现在|以前|之后|再后来)?(?:呢|怎么样了?|还.+吗|又.+吗|接着呢)[？?~～。！!]*$"
)


@dataclass(frozen=True)
class ReferenceResolution:
    user_ids: tuple[int, ...] = ()
    expanded_query: str = ""
    reason: str = "none"
    confidence: float = 0.0


def resolve_context_reference(
    text: str,
    recent_messages: Iterable[object],
    *,
    current_user_id: int,
    resolve_named_users: Callable[[str], Iterable[int]] | None = None,
) -> ReferenceResolution:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    pronoun = PRONOUN_RE.search(f" {clean}") is not None
    elliptical = ELLIPTICAL_FOLLOWUP_RE.match(clean) is not None
    if not pronoun and not elliptical:
        return ReferenceResolution()

    messages = tuple(recent_messages)
    if resolve_named_users is not None:
        for message in reversed(messages):
            if bool(getattr(message, "is_bot", False)):
                continue
            previous_text = str(getattr(message, "text", "") or "").strip()
            if not previous_text or previous_text == clean:
                continue
            resolved = tuple(dict.fromkeys(int(value) for value in resolve_named_users(previous_text) if int(value) > 0))
            if len(resolved) == 1:
                return ReferenceResolution(
                    resolved,
                    _expanded_query(clean, previous_text),
                    "previous_named_member",
                    0.9,
                )

    # A pronoun in a direct reply commonly refers to the latest other human
    # speaker. Use this only when there is one unambiguous nearest candidate.
    for message in reversed(messages):
        if bool(getattr(message, "is_bot", False)):
            continue
        user_id = int(getattr(message, "user_id", 0) or 0)
        previous_text = str(getattr(message, "text", "") or "").strip()
        if user_id <= 0 or user_id == current_user_id or not previous_text:
            continue
        return ReferenceResolution(
            (user_id,),
            _expanded_query(clean, previous_text),
            "latest_other_speaker",
            0.72 if pronoun else 0.62,
        )
    return ReferenceResolution(reason="ambiguous", confidence=0.0)


def _expanded_query(current: str, previous: str) -> str:
    previous = re.sub(r"\s+", " ", previous).strip()
    if len(previous) > 140:
        previous = previous[-140:]
    return f"{current}（承接前文：{previous}）"
