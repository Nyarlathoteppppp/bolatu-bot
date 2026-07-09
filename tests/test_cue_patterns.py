from qq_social_agent.cue_patterns import CuePatternTracker, classify_cue_pattern


def test_classify_evaluation_cue() -> None:
    assert classify_cue_pattern("评价一下未明子") == "evaluation"
    assert classify_cue_pattern("你怎么看米哈游校招") == "evaluation"


def test_classify_comparison_cue() -> None:
    assert classify_cue_pattern("詹姆斯和东契奇谁厉害") == "comparison"
    assert classify_cue_pattern("哪个更强") == "comparison"


def test_classify_command_cue() -> None:
    assert classify_cue_pattern("快点") == "command"
    assert classify_cue_pattern("给我锐评") == "command"


def test_tracker_only_counts_addressed_messages() -> None:
    tracker = CuePatternTracker(window_seconds=600)

    assert tracker.record(
        group_id=1,
        user_id=2,
        text="评价一下未明子",
        addressed=False,
        now=100,
    ) is None

    first = tracker.record(
        group_id=1,
        user_id=2,
        text="评价一下未明子",
        addressed=True,
        now=101,
    )
    second = tracker.record(
        group_id=1,
        user_id=2,
        text="怎么看米哈游校招",
        addressed=True,
        now=102,
    )
    third = tracker.record(
        group_id=1,
        user_id=2,
        text="锐评一下张雪峰",
        addressed=True,
        now=103,
    )

    assert first is not None and first.count == 1
    assert second is not None and second.count == 2
    assert third is not None
    assert third.kind == "evaluation"
    assert third.count == 3
