"""Actual prior contributions, not inferred emotions or invented internal monologues."""

_FEEDBACK_VERBS = {
    "correction": "纠正了你",
    "tone_feedback": "对你的语气有意见",
    "closed": "表示这个话题结束了",
}


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_self_interaction_context(contributions: list[dict[str, object]]) -> str:
    if not contributions:
        return ""
    lines = ["你在这段互动里实际发出去的话（旧到新）："]
    for item in contributions:
        trigger = _clip(item.get("trigger_said"), 40)
        speaker = str(item.get("trigger_speaker") or "群友")
        cue = f"回应 {speaker}「{trigger}」" if trigger else f"回应 {speaker}"
        lines.append(f"- {cue}，你说：「{_clip(item.get('said'), 160)}」")
        feedback_items = item.get("subsequent_feedback")
        for feedback in feedback_items if isinstance(feedback_items, list) else ():
            if not isinstance(feedback, dict):
                continue
            verb = _FEEDBACK_VERBS.get(str(feedback.get("kind") or ""), "回应了你")
            lines.append(f"  之后 {feedback.get('speaker') or '群友'} {verb}：「{_clip(feedback.get('said'), 80)}」")
    return "<self_interaction_context>\n" + "\n".join(lines) + "\n</self_interaction_context>"
