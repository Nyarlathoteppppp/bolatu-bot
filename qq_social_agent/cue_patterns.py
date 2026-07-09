from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CueRepeatState:
    kind: str
    label: str
    count: int


class CuePatternTracker:
    def __init__(self, *, window_seconds: int = 10 * 60):
        self.window_seconds = window_seconds
        self._events: dict[tuple[int, int, str], list[float]] = {}

    def record(
        self,
        *,
        group_id: int,
        user_id: int,
        text: str,
        addressed: bool,
        now: float | None = None,
    ) -> CueRepeatState | None:
        if not addressed:
            return None
        kind = classify_cue_pattern(text)
        if kind is None:
            return None
        timestamp = now or time.time()
        key = (group_id, user_id, kind)
        recent = [
            ts for ts in self._events.get(key, []) if timestamp - ts <= self.window_seconds
        ]
        recent.append(timestamp)
        self._events[key] = recent
        return CueRepeatState(kind=kind, label=CUE_LABELS[kind], count=len(recent))


CUE_LABELS = {
    "evaluation": "连续问评价类问题",
    "comparison": "连续问谁厉害/谁更强",
    "command": "连续命令式 cue",
}


def classify_cue_pattern(text: str) -> str | None:
    compact = re.sub(r"[\s，。！？!?、,.]+", "", text.lower())
    if not compact:
        return None

    if _is_comparison_cue(compact):
        return "comparison"
    if _is_command_cue(compact):
        return "command"
    if _is_evaluation_cue(compact):
        return "evaluation"
    return None


def _is_evaluation_cue(text: str) -> bool:
    patterns = (
        r"评价(一下)?",
        r"锐评(一下)?",
        r"点评(一下)?",
        r"怎么看",
        r"怎么评价",
        r"如何评价",
        r"如何看待",
        r"你觉得.*(怎么样|如何|咋样)",
        r"谈谈.*看法",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_comparison_cue(text: str) -> bool:
    patterns = (
        r"谁(更)?(厉害|强|牛逼|牛|吊|猛)",
        r"哪个(更)?(厉害|强|好|牛逼|牛|吊|猛)",
        r"哪边(更)?(厉害|强|好|牛逼|牛|吊|猛)",
        r".+和.+谁(更)?(厉害|强|牛逼|牛|吊|猛)",
        r".+跟.+谁(更)?(厉害|强|牛逼|牛|吊|猛)",
        r".+(打得过|能不能打过|能打过).+",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_command_cue(text: str) -> bool:
    if len(text) <= 1:
        return False
    direct_commands = {
        "快点",
        "说",
        "讲",
        "念",
        "回答",
        "快说",
        "快讲",
        "快念",
        "来评",
        "快评",
        "开评",
        "继续",
    }
    if text in direct_commands:
        return True
    prefixes = (
        "你来说",
        "你来讲",
        "你来评",
        "你评价",
        "你锐评",
        "给我说",
        "给我讲",
        "给我评价",
        "给我锐评",
        "帮我评价",
        "帮我锐评",
        "快评价",
        "快锐评",
    )
    return text.startswith(prefixes)
