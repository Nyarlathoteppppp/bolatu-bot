"""Offline correctness cases. No live Jev, no QQ send."""
import json
from pathlib import Path

from qq_social_agent.discourse_state import should_ask_jev_addressee
from qq_social_agent.ellipsis_resolver import apply_jev_ellipsis_judgement, should_ask_jev_ellipsis
from qq_social_agent.jev_client import _timing_looks_like_reply_to_other
from qq_social_agent.reference_resolver import ReferenceResolution, ReplyHint, apply_jev_referent_judgement
from qq_social_agent.tool_router import route_tools

BOT_ALIASES = ("张风雪", "风雪")

CASES = json.loads((Path(__file__).parent / "fixtures" / "dialogue_correctness_cases.json").read_text())


def test_cases_cover_required_scenes():
    scenes = {case["scene"] for case in CASES}
    assert {"ellipsis", "addressed", "tool", "timing", "unavailable", "addressee"} <= scenes


def test_offline_correctness_expectations():
    for case in CASES:
        text = case["current_text"]
        expect = case["expect"]
        if "ask_ellipsis" in expect:
            assert should_ask_jev_ellipsis(text) is expect["ask_ellipsis"], case["id"]
        if "addressed_by_name" in expect:
            assert any(alias in text for alias in BOT_ALIASES) is expect["addressed_by_name"], case["id"]
        if "force_probability_tool" in expect:
            plan = route_tools(text, market_intents=[], fresh_intent=None, addressed=True)
            has_prob = any(req.kind.value == "probability" for req in plan.requests)
            assert has_prob is expect["force_probability_tool"], case["id"]
        if "timing_reply_to_other" in expect:
            assert _timing_looks_like_reply_to_other(text) is expect["timing_reply_to_other"], case["id"]
        if expect.get("unavailable_is_not_none"):
            ellipsis = apply_jev_ellipsis_judgement(None, [], current_text=text)
            referent = apply_jev_referent_judgement(
                None, [], current_text=text, fallback=ReferenceResolution()
            )
            assert ellipsis.status == "UNAVAILABLE"
            assert referent.status == "UNAVAILABLE"
            assert ellipsis.status != "NOT_APPLICABLE"
            assert referent.kind != "NONE" or referent.status == "UNAVAILABLE"
        if expect.get("isolate_addressee"):
            reply = ReplyHint(exists=True, author_id=111, author_label="U1", text="hi")
            assert should_ask_jev_addressee(text, reply=reply, at_user_ids=(1801507496,))
