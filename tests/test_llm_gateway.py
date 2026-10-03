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


def test_reply_peak_combo_skips_mimo_and_selects_configured_pair() -> None:
    gateway = _gateway()
    gateway._is_reply_peak_now = lambda: False
    gateway.set_reply_peak_combo()
    assert [route.provider for route in gateway._candidate_routes("reply")] == ["deepseek", "siliconflow"]
    gateway._is_reply_peak_now = lambda: True
    assert [route.provider for route in gateway._candidate_routes("reply")] == ["siliconflow", "deepseek"]
    gateway.set_route_override("reply", gateway.config.routes["reply"])
    assert gateway.current_route("reply").provider == "mimo"


def test_selected_background_model_does_not_change_reply_route() -> None:
    gateway = _gateway()
    gateway.clients = {"deepseek": FakeChatClient(), "mimo": FakeChatClient()}
    selected = gateway.config.fallback_routes["reply"]

    result = asyncio.run(gateway.complete_on_model(task="daily_review", route=selected, request={"messages": []}))

    assert result.model == selected.model
    assert gateway.current_route("reply").provider == "mimo"
    assert gateway.clients["mimo"].calls == 0


def test_responses_gateway_preserves_json_usage_and_timeout_fallback() -> None:
    from dataclasses import replace
    from qq_social_agent.config import LLMProviderConfig, LLMModelRoute
    gateway = _gateway()
    provider = LLMProviderConfig('lingsuan', 'https://edge.lingsuan.org/v1', 'LINGSUAN_API_KEY', 'disabled', 'responses')
    gateway.config = replace(gateway.config, providers={**gateway.config.providers, 'lingsuan': provider})
    gateway.route_overrides['reply'] = LLMModelRoute('lingsuan', 'gpt-6.1-sol')
    calls = []
    class ResponsesClient(FakeChatClient):
        def __init__(self):
            super().__init__()
            self.responses = self
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"text":"好呀"}', status='completed', usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18))
    client = ResponsesClient()
    gateway.clients = {'lingsuan': client, 'deepseek': FakeChatClient()}
    request = {'messages': [{'role': 'user', 'content': 'Return JSON'}], 'max_tokens': 320, 'temperature': 0.6, 'response_format': {'type': 'json_object'}}
    response = asyncio.run(gateway._chat_completion(task='reply_direct', route_name='reply', request=request))
    assert response.choices[0].message.content == '{"text":"好呀"}'
    assert response.usage.prompt_tokens == 11
    assert calls[0]['store'] is False
    assert calls[0]['reasoning'] == {'effort': 'none'}
    assert calls[0]['text'] == {'format': {'type': 'json_object'}}
    assert 'temperature' not in calls[0] and 'messages' not in calls[0]
    async def timeout(**kwargs):
        raise asyncio.TimeoutError()
    client.create = timeout
    result = asyncio.run(gateway._chat_completion(task='reply_direct', route_name='reply', request=request))
    assert result.model == 'deepseek-flash'


def test_lingsuan_uses_medium_for_reply_low_for_review_and_only_official_fallback():
    from dataclasses import replace
    from qq_social_agent.config import LLMProviderConfig, LLMModelRoute
    gateway = _gateway()
    provider = LLMProviderConfig('lingsuan', 'https://edge.lingsuan.org/v1', 'LINGSUAN_API_KEY',
                                'enabled', 'responses', reply_timeout_seconds=30, reply_total_timeout_seconds=50)
    gateway.config = replace(gateway.config, providers={**gateway.config.providers, 'lingsuan': provider})
    for route in ['reply', 'decision']:
        gateway.route_overrides[route] = LLMModelRoute('lingsuan', 'gpt-6.1-sol')
    calls = []
    class ResponsesClient(FakeChatClient):
        def __init__(self):
            super().__init__()
            self.responses = self
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='OK', status='completed', usage=None)
    gateway.clients = {'lingsuan': ResponsesClient()}
    asyncio.run(gateway._chat_completion(task='reply_direct', route_name='reply', request={'messages': []}))
    asyncio.run(gateway._chat_completion(task='timing_review', route_name='decision', request={'messages': []}))
    assert [request['reasoning']['effort'] for request in calls] == ['medium', 'low']
    assert [route.provider for route in gateway._candidate_routes('reply')] == ['lingsuan', 'deepseek']
    assert [route.provider for route in gateway._candidate_routes('decision')] == ['lingsuan', 'deepseek']
    assert gateway._task_timeouts(task='timing_review', route_name='decision') == (30.0, 50.0)


def _production_gateway():
    from qq_social_agent.config import PROJECT_ROOT, load_config
    gateway = _gateway()
    gateway.config = load_config(PROJECT_ROOT / 'config.yaml').llm
    return gateway


def test_verysadai_chain_preserves_manual_lingsuan_and_peak_combo():
    from qq_social_agent.config import LLMModelRoute
    gateway = _production_gateway()
    gateway._is_reply_peak_now = lambda: True
    # Default reply route is Lingsuan; VerySadai is a manual switch (its group
    # injects Codex instructions that break the persona).
    assert [r.provider for r in gateway._candidate_routes('reply')] == ['lingsuan', 'deepseek']
    gateway.set_route_override('reply', LLMModelRoute('verysadai', 'gpt-6.1-sol'))
    assert [r.provider for r in gateway._candidate_routes('reply')] == ['verysadai', 'lingsuan', 'deepseek']
    assert gateway._task_timeouts(task='reply_direct', route_name='reply') == (30.0, 80.0)
    gateway.set_route_override('reply', LLMModelRoute('lingsuan', 'gpt-6.1-sol'))
    assert [r.provider for r in gateway._candidate_routes('reply')] == ['lingsuan', 'deepseek']
    gateway.set_reply_peak_combo()
    assert [r.provider for r in gateway._candidate_routes('reply')] == ['siliconflow', 'deepseek']


def test_both_sol_providers_fail_then_official_deepseek_receives_original_messages():
    from qq_social_agent.config import LLMModelRoute
    gateway = _production_gateway()
    gateway.set_route_override('reply', LLMModelRoute('verysadai', 'gpt-6.1-sol'))
    calls = []
    class BrokenResponsesClient(FakeChatClient):
        def __init__(self, label):
            super().__init__()
            self.responses = self
            self.label = label
        async def create(self, **kwargs):
            calls.append((self.label, kwargs))
            raise asyncio.TimeoutError()
    ds = FakeChatClient()
    gateway.clients = {'verysadai': BrokenResponsesClient('verysadai'), 'lingsuan': BrokenResponsesClient('lingsuan'), 'deepseek': ds}
    result = asyncio.run(gateway._chat_completion(task='reply_direct', route_name='reply', request={'messages': [{'role': 'user', 'content': '你好'}]}))
    assert result.model == 'deepseek-flash'
    assert [label for label, _ in calls] == ['verysadai', 'lingsuan']
    assert [request['reasoning']['effort'] for _, request in calls] == ['medium', 'medium']
    assert ds.calls == 1
    gateway._provider_circuit_until['verysadai'] = float('inf')
    assert [r.provider for r in gateway._candidate_routes('reply')] == ['lingsuan', 'deepseek']
