from qq_social_agent.delivery import build_delivery_plan


def test_delivery_plan_forces_trigger_mention_after_three_new_messages() -> None:
    plan = build_delivery_plan(
        reply_text="这个要看具体数据。",
        mention_targets={},
        trigger_user_id=12345,
        trigger_nickname="提问人",
        trigger_sequence=10,
        current_sequence=13,
    )

    assert plan.forced_trigger_mention
    assert plan.sequence_lag == 3
    assert plan.mention_targets == {12345: "提问人"}
    assert plan.parts[0].startswith("[[at:12345]]")


def test_delivery_plan_does_not_force_mention_for_current_reply() -> None:
    plan = build_delivery_plan(
        reply_text="现在就回。",
        mention_targets={},
        trigger_user_id=12345,
        trigger_nickname="提问人",
        trigger_sequence=10,
        current_sequence=11,
    )

    assert not plan.forced_trigger_mention
    assert plan.parts == ("现在就回。",)
