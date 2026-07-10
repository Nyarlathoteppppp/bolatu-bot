from qq_social_agent.config import AppConfig, PROJECT_ROOT, parse_llm_model_route
from qq_social_agent.persona import PersonaRegistry
from qq_social_agent.prompts import PromptRegistry


def test_deepseek_split_models_default_to_flash_for_utility() -> None:
    config = AppConfig({"deepseek": {"model": "deepseek-v4-pro"}}).deepseek

    assert config.model == "deepseek-v4-pro"
    assert config.reply_model == "deepseek-v4-pro"
    assert config.decision_model == "deepseek-v4-flash"
    assert config.utility_model == "deepseek-v4-flash"
    assert config.jargon_model == "deepseek-v4-flash"
    assert config.memory_model == "deepseek-v4-flash"
    assert config.style_model == "deepseek-v4-flash"


def test_deepseek_split_models_can_be_configured() -> None:
    config = AppConfig(
        {
            "deepseek": {
                "model": "base",
                "decision_model": "decision",
                "reply_model": "reply",
                "utility_model": "utility",
                "jargon_model": "jargon",
                "memory_model": "memory",
                "style_model": "style",
                "model_catalog": ["deepseek-v4-flash", "deepseek/deepseek-v4-pro"],
            }
        }
    ).deepseek

    assert config.decision_model == "decision"
    assert config.reply_model == "reply"
    assert config.utility_model == "utility"
    assert config.jargon_model == "jargon"
    assert config.memory_model == "memory"
    assert config.style_model == "style"
    assert tuple(route.label for route in config.model_catalog) == (
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    )


def test_llm_provider_routes_can_use_multislash_model_names() -> None:
    config = AppConfig(
        {
            "deepseek": {
                "providers": {
                    "siliconflow": {
                        "base_url": "https://api.siliconflow.cn/v1",
                        "api_key_env": "SILICONFLOW_API_KEY",
                    }
                },
                "decision_model": "siliconflow/Qwen/Qwen3.5-35B-A3B",
                "reply_model": "siliconflow/MiniMaxAI/MiniMax-M2.5",
                "fallback_models": {
                    "reply": "deepseek/deepseek-v4-flash",
                },
            }
        }
    ).deepseek

    assert config.routes["decision"].provider == "siliconflow"
    assert config.routes["decision"].model == "Qwen/Qwen3.5-35B-A3B"
    assert config.routes["reply"].provider == "siliconflow"
    assert config.routes["reply"].model == "MiniMaxAI/MiniMax-M2.5"
    assert config.fallback_routes["reply"].label == "deepseek/deepseek-v4-flash"


def test_llm_route_catalog_and_split_utility_flows() -> None:
    config = AppConfig(
        {
            "deepseek": {
                "providers": {
                    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
                },
                "decision_model": "siliconflow/Qwen/Qwen3.5-35B-A3B",
                "reply_model": "siliconflow/MiniMaxAI/MiniMax-M2.5",
                "jargon_model": "siliconflow/Qwen/Qwen3.5-35B-A3B",
                "memory_model": "deepseek/deepseek-v4-flash",
                "style_model": "siliconflow/MiniMaxAI/MiniMax-M2.5",
                "model_catalog": [
                    "deepseek/deepseek-v4-flash",
                    "siliconflow/Pro/moonshotai/Kimi-K2.6",
                ],
            }
        }
    ).deepseek

    assert set(config.routes) >= {"decision", "reply", "jargon", "memory", "style", "member_profile"}
    assert config.routes["jargon"].label == "siliconflow/Qwen/Qwen3.5-35B-A3B"
    assert config.routes["memory"].label == "deepseek/deepseek-v4-flash"
    assert config.routes["style"].label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert config.routes["member_profile"].label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert tuple(route.label for route in config.model_catalog) == (
        "deepseek/deepseek-v4-flash",
        "siliconflow/Pro/moonshotai/Kimi-K2.6",
    )


def test_parse_llm_model_route_defaults_to_given_provider() -> None:
    config = AppConfig(
        {
            "deepseek": {
                "providers": {
                    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
                }
            }
        }
    ).deepseek

    route = parse_llm_model_route("MiniMaxAI/MiniMax-M2.5", config.providers, default_provider="siliconflow")

    assert route.label == "siliconflow/MiniMaxAI/MiniMax-M2.5"


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


def test_usage_tracking_can_be_disabled() -> None:
    config = AppConfig({"deepseek": {"usage_tracking_enabled": False}}).deepseek

    assert not config.usage_tracking_enabled


def test_prompt_registry_loads_central_prompt_file() -> None:
    prompts = PromptRegistry()

    assert "QQ 群聊行动决策器" in prompts.render(
        "decision",
        "system",
        persona_name="张风雪",
        persona_decision_prompt="人格摘要",
    )
    assert "群聊三候选生成" in prompts.raw["flows"]["reply_candidates"]["flow"]
    assert "普通接话" in prompts.action_guide("reply")
    assert "关心/承接" in prompts.action_guide("care")


def test_central_prompt_file_contains_all_runtime_flows() -> None:
    prompts = PromptRegistry()
    required_flows = {
        "decision",
        "jargon_select",
        "reply",
        "reply_candidates",
        "daily_review",
        "mid_memory",
        "style_learning",
    }

    assert required_flows <= set(prompts.flows)
    for flow in required_flows:
        assert prompts.raw["flows"][flow]["flow"]
        assert prompts.render(
            flow,
            "system",
            persona_name="张风雪",
            persona_decision_prompt="人格",
            persona_prompt="人格",
            max_reply_chars=520,
        )


def test_persona_registry_loads_persona_from_central_prompt_file() -> None:
    registry = PersonaRegistry(PROJECT_ROOT / "prompts")
    persona = registry.get("zhangxuefeng")

    assert persona.name == "张风雪"
    assert "温柔美少女妹妹" in persona.prompt
    assert "QQ 群里的温柔美少女妹妹" in persona.decision_prompt
    assert "棘手的大问题" in persona.decision_prompt
