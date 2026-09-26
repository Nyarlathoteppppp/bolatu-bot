"""Pinned Jev model, thresholds, and group-reply time budget.

Upgrade only with the anonymized correctness cases. Noul thresholds are not
interchangeable with Choice confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


JEV_MODEL_VERSION = "1.13.0"
TYPESAFE_JEV_MODEL = "jev-1.13.0"
OPENROUTER_JEV_MODEL = "typesafe/jev-1.13"

JEV_ADDRESSEE_CONFIDENCE_MIN = 0.35
JEV_AUDIT_CONFIDENCE_MIN = 0.55
JEV_ELLIPSIS_CONFIDENCE_MIN = 0.35
JEV_PERSON_CONFIDENCE_MIN = 0.55
JEV_REPAIR_CONFIDENCE_MIN = 0.42
JEV_MEMORY_CONFIDENCE_MIN = 0.60
JEV_AMBIGUITY_CONFIDENCE_MIN = 0.60
JEV_TOOL_NOUL_MIN = 0.40
JEV_TOOL_CHOICE_CONFIDENCE_MIN = 0.50

# Group timing: a Noul is evidence for its own proposition, not a reply
# probability. Keep the routing thresholds together for replay calibration.
JEV_TIMING_TO_OTHER_MIN = 0.65
JEV_TIMING_CARE_MIN = 0.58
JEV_TIMING_ANSWER_INTENT_MIN = 0.50
JEV_TIMING_SEMANTIC_REQUEST_MIN = 0.85
JEV_TIMING_ANSWER_CHOICE_MIN = 0.40
JEV_TIMING_ANSWER_SILENT_MAX = 0.40
JEV_TIMING_SOCIAL_CHOICE_MIN = 0.60
JEV_TIMING_SOCIAL_SILENT_MAX = 0.25

CRITIC_FAIL_THRESHOLD = 0.75
CRITIC_ANSWER_INTENT_FAIL_THRESHOLD = 0.80
CRITIC_INTENT_FAIL_THRESHOLD = 0.90
CRITIC_PASS_THRESHOLD = 0.45

GROUP_REPLY_BUDGET_SECONDS = 28.0
GENERATION_RESERVE_SECONDS = 12.0
STAGE_NEED_SECONDS = {
    "speaking_action": 3.0,
    "ask_back": 3.0,
    "optional_rag": 4.0,
    "critic_retry": 8.0,
}


@dataclass
class GroupReplyBudget:
    started_at: float
    deadline: float
    skipped: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, now: float, *, seconds: float = GROUP_REPLY_BUDGET_SECONDS) -> "GroupReplyBudget":
        return cls(started_at=now, deadline=now + max(1.0, float(seconds)))

    def remaining(self, now: float) -> float:
        return self.deadline - now

    def skip(self, stage: str, now: float) -> bool:
        need = STAGE_NEED_SECONDS[stage]
        if self.remaining(now) < need + GENERATION_RESERVE_SECONDS:
            self.skipped.append(stage)
            return True
        return False
