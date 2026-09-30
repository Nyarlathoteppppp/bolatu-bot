"""Actual prior contributions, not inferred emotions or invented internal monologues."""

import json


def format_self_interaction_context(contributions: list[dict[str, object]]) -> str:
    if not contributions:
        return ""
    return "<self_interaction_context>\n" + json.dumps(
        {"own_contributions": contributions}, ensure_ascii=False,
    ) + "\n</self_interaction_context>"
