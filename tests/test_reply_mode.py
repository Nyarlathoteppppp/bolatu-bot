from qq_social_agent.reply_mode import ASSOCIATE, FOLLOW, FREE, pick_reply_mode, reply_mode_guide


def test_modes_are_drawn_evenly() -> None:
    assert pick_reply_mode(lambda: 0.1) == FOLLOW
    assert pick_reply_mode(lambda: 0.5) == ASSOCIATE
    assert pick_reply_mode(lambda: 0.9) == FREE


def test_direct_questions_are_answered_first_then_the_drawn_mode() -> None:
    guide = reply_mode_guide(ASSOCIATE, addressed_question=True)
    assert guide.startswith("本轮接法：对方问了你具体的问题，先把它答清楚；之后联想展开")
    assert "先把它答清楚" not in reply_mode_guide(ASSOCIATE, addressed_question=False)
