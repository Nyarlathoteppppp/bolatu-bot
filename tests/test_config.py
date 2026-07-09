from qq_social_agent.config import AppConfig


def test_deepseek_split_models_default_to_flash_for_utility() -> None:
    config = AppConfig({"deepseek": {"model": "deepseek-v4-pro"}}).deepseek

    assert config.model == "deepseek-v4-pro"
    assert config.reply_model == "deepseek-v4-pro"
    assert config.decision_model == "deepseek-v4-flash"
    assert config.utility_model == "deepseek-v4-flash"


def test_deepseek_split_models_can_be_configured() -> None:
    config = AppConfig(
        {
            "deepseek": {
                "model": "base",
                "decision_model": "decision",
                "reply_model": "reply",
                "utility_model": "utility",
            }
        }
    ).deepseek

    assert config.decision_model == "decision"
    assert config.reply_model == "reply"
    assert config.utility_model == "utility"


def test_user_reply_cooldowns_can_be_configured() -> None:
    config = AppConfig(
        {
            "rate_control": {
                "user_reply_cooldowns": {
                    "3370998238": 120,
                    123: "30",
                }
            }
        }
    )

    assert config.user_reply_cooldowns == {3370998238: 120, 123: 30}
