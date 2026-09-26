import asyncio
from types import SimpleNamespace

from qq_social_agent.config import AppConfig
from qq_social_agent.llm_gateway import LLMGateway


class FakeChatClient:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    async def create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(model=kwargs["model"], choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


def _gateway() -> LLMGateway:
    config = AppConfig({
        "llm": {
            "providers": {
                "mimo": {"base_url": "https://api.xiaomimimo.com/v1", "api_key_env": "MIMO_API_KEY"},
                "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
            },
            "reply_model": "mimo/mimo-v2.6-pro",
            "fallback_models": {"reply": ["deepseek/deepseek-flash", "siliconflow/backup"]},
            "usage_tracking_enabled": False,
        }
    }).llm
    gateway = LLMGateway.__new__(LLMGateway)
    gateway.config = config
    gateway.route_overrides = {}
    gateway._provider_failures = {}
    gateway._provider_circuit_until = {}
    gateway._provider_exhausted = set()
    return gateway


def test_new_provider_routes_to_existing_models_after_balance_exhaustion() -> None:
    gateway = _gateway()
    exhausted = RuntimeError("Insufficient Balance")
    exhausted.status_code = 402
    mimo = FakeChatClient(error=exhausted)
    deepseek = FakeChatClient(error=RuntimeError("provider unavailable"))
    siliconflow = FakeChatClient()
    gateway.clients = {"mimo": mimo, "deepseek": deepseek, "siliconflow": siliconflow}

    first = asyncio.run(gateway._chat_completion(task="reply", route_name="reply", request={"messages": []}))
    second = asyncio.run(gateway._chat_completion(task="reply", route_name="reply", request={"messages": []}))

    assert first.model == second.model == "backup"
    assert mimo.calls == 1
    assert deepseek.calls == siliconflow.calls == 2
    assert gateway._provider_exhausted == {"mimo"}


def test_missing_new_provider_key_uses_configured_fallback() -> None:
    gateway = _gateway()
    deepseek = FakeChatClient()
    gateway.clients = {"deepseek": deepseek}

    response = asyncio.run(gateway._chat_completion(task="reply", route_name="reply", request={"messages": []}))

    assert response.model == "deepseek-flash"
    assert deepseek.calls == 1


def test_model_probe_checks_requested_provider_without_fallback() -> None:
    gateway = _gateway()
    mimo = FakeChatClient(error=RuntimeError("service unavailable"))
    deepseek = FakeChatClient()
    gateway.clients = {"mimo": mimo, "deepseek": deepseek}

    available, reason = asyncio.run(gateway.probe_model(gateway.config.routes["reply"]))

    assert not available
    assert reason == "RuntimeError"
    assert mimo.calls == 1
    assert deepseek.calls == 0
