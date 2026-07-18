from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .pipeline_types import OutputChannel, SocialIntent

if TYPE_CHECKING:
    from .deepseek_client import ReplyDecision


INTENT_TO_ACTION = {
    SocialIntent.ANSWER: "answer",
    SocialIntent.CARE: "care",
    SocialIntent.PLAY: "tease",
    SocialIntent.AGREE: "agree",
    SocialIntent.CHAT: "reply",
}


@dataclass(frozen=True)
class TimingDecision:
    channel: OutputChannel
    intent: SocialIntent = SocialIntent.CHAT
    confidence: float = 0.0
    reason: str = ""
    reaction: str = ""
    side_reaction: str = ""

    def to_reply_decision(self) -> ReplyDecision:
        from .deepseek_client import ReplyDecision

        if self.channel == OutputChannel.SILENT:
            return ReplyDecision(False, self.confidence, self.reason, action="ignore")
        if self.channel == OutputChannel.REACT:
            return ReplyDecision(
                True,
                self.confidence,
                self.reason,
                mode="chat",
                action="react",
                reaction=self.reaction,
            )
        if self.channel == OutputChannel.POKE:
            return ReplyDecision(True, self.confidence, self.reason, mode="chat", action="poke")
        return ReplyDecision(
            True,
            self.confidence,
            self.reason,
            mode="chat",
            action=INTENT_TO_ACTION.get(self.intent, "reply"),
            side_reaction=self.side_reaction,
        )


def parse_timing_decision(raw: object) -> TimingDecision:
    data = raw if isinstance(raw, dict) else {}
    try:
        channel = OutputChannel(str(data.get("channel", "silent")).strip().lower())
    except ValueError:
        channel = OutputChannel.SILENT
    try:
        intent = SocialIntent(str(data.get("intent", "chat")).strip().lower())
    except ValueError:
        intent = SocialIntent.CHAT
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reaction = str(data.get("reaction", "") or "").strip().lower()
    if reaction not in {"agree", "care", "laugh", "tease", "surprise", "question", "applause", "heart"}:
        reaction = ""
    side_reaction = str(
        data.get("side_reaction", "")
        or data.get("sideReaction", "")
        or data.get("emoji_reaction", "")
        or ""
    ).strip().lower()
    if side_reaction not in {"agree", "care", "laugh", "tease", "surprise", "question", "applause", "heart"}:
        side_reaction = ""
    if channel == OutputChannel.REACT:
        if not reaction and side_reaction:
            reaction = side_reaction
        side_reaction = ""
    elif reaction and not side_reaction:
        side_reaction = reaction
        reaction = ""
    return TimingDecision(
        channel=channel,
        intent=intent,
        confidence=confidence,
        reason=str(data.get("reason", "") or "")[:40],
        reaction=reaction,
        side_reaction=side_reaction,
    )
