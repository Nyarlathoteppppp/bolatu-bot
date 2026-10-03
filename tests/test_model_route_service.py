import asyncio
import json
import logging

import pytest
from types import SimpleNamespace

from qq_social_agent.config import LLMModelRoute, parse_llm_model_route
from qq_social_agent.model_route_service import (
    BACKGROUND_MODEL_OVERRIDES_KEY,
    MODEL_ROUTE_OVERRIDES_KEY,
    MODEL_ROUTE_STORAGE_NAMES,
    REPLY_PEAK_COMBO_KEY,
    REPLY_PEAK_COMBO_LABEL,
    ModelRouteService,
)


def _route(label: str) -> LLMModelRoute:
    provider, model = label.split("/", 1)
    return LLMModelRoute(provider, model)


def _provider(name: str, *, fallbacks: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(name=name, api_key_env=f"{name.upper()}_API_KEY", reply_fallback_models=fallbacks)


def _llm() -> SimpleNamespace:
    providers = {
        "deepseek": _provider("deepseek", fallbacks=("deepseek/official-a", "deepseek/official-b")),
        "siliconflow": _provider("siliconflow", fallbacks=("siliconflow/sf-fallback",)),
        "mimo": _provider("mimo"),
        "lingsuan": _provider("lingsuan", fallbacks=("lingsuan/should-not-show",)),
    }
    routes = {
        "decision": _route("deepseek/decision-model"),
        "reply": _route("deepseek/reply-model"),
        "search": _route("deepseek/search-model"),
        "jargon": _route("siliconflow/jargon-model"),
        "memory": _route("siliconflow/memory-model"),
        "style": _route("siliconflow/style-model"),
        "member_profile": _route("siliconflow/profile-model"),
        "utility": _route("deepseek/utility-model"),
    }
    fallback_routes = {name: _route(f"siliconflow/{name}-fallback") for name in routes}
    additional = {
        "reply": (_route("mimo/reply-extra"),),
        "search": (_route("mimo/search-extra"),),
    }
    return SimpleNamespace(
        providers=providers,
        routes=routes,
        fallback_routes=fallback_routes,
        additional_fallback_routes=additional,
        model_catalog=(
            _route("mimo/mimo-v2.6-pro"),
            _route("deepseek/deepseek-flash"),
            _route("siliconflow/deepseek-ai/DeepSeek-V4-Flash"),
        ),
    )


class Kv:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def app_kv_get(self, key: str) -> str | None:
        return self.data.get(key)

    def app_kv_set(self, key: str, value: str) -> None:
        self.data[key] = value


class Client:
    def __init__(self, llm: SimpleNamespace) -> None:
        self.llm = llm
        self.overrides: dict[str, LLMModelRoute | None] = {}
        self.peak = False
        self.probes: list[LLMModelRoute] = []
        self.max_inflight = 0
        self._inflight = 0
        self.probe_gate = None
        self.fail_parse = False

    def parse_model_route(self, value: str, *, default_provider: str = "siliconflow") -> LLMModelRoute:
        if self.fail_parse:
            raise RuntimeError("boom")
        return parse_llm_model_route(value, self.llm.providers, default_provider=default_provider)

    def set_route_override(self, route_name: str, route: LLMModelRoute | None) -> None:
        if route is None:
            self.overrides.pop(route_name, None)
            if route_name == "reply":
                self.peak = False
            return
        self.overrides[route_name] = route
        if route_name == "reply":
            self.peak = False

    def set_reply_peak_combo(self) -> None:
        self.overrides.pop("reply", None)
        self.peak = True

    def current_route(self, route_name: str) -> LLMModelRoute:
        if route_name == "reply" and self.peak:
            return self.llm.fallback_routes["reply"]
        return self.overrides.get(route_name, self.llm.routes[route_name])

    async def probe_model(self, route: LLMModelRoute) -> tuple[bool, str]:
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        self.probes.append(route)
        try:
            if self.probe_gate is not None:
                await self.probe_gate(route)
            if route.provider == "mimo":
                return True, "可用"
            return False, "HTTP 403"
        finally:
            self._inflight -= 1


class Batch:
    def __init__(self) -> None:
        self.result = SimpleNamespace(status="completed", content="ok", error=None)
        self.counts = {"queued": 1, "completed": 2}

    async def probe(self):
        return self.result

    def status_snapshot(self):
        return {"counts": self.counts}


class Sender:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send(self, bot, user_id: int, text: str) -> None:
        self.messages.append((user_id, text))


def _service(*, client: Client | None = None, owner: bool = True) -> tuple[ModelRouteService, Kv, Client, Batch, Sender]:
    llm = _llm()
    kv = Kv()
    resolved = client if client is not None else Client(llm)
    batch = Batch()
    sender = Sender()
    background = {
        "memory": "verysadai/gpt-6.1-sol",
        "review": "lingsuan/gpt-6.1-sol",
        "catalog": [
            "verysadai/gpt-6.1-sol",
            "openrouter/z-ai/glm-5.3-flash:batch",
            "siliconflow/deepseek-ai/DeepSeek-V4-Flash",
            "硅基/模型",
        ],
    }
    service = ModelRouteService(
        llm=llm,
        background_models=background,
        kv=kv,
        client=resolved,
        batch_service=batch,
        is_owner=lambda user_id: owner and user_id == 1535071184,
        send_private_text=sender.send,
        logger=logging.getLogger("qq_social_agent"),
    )
    return service, kv, resolved, batch, sender


def test_catalog_numbering_and_select_reject_batch_and_bad_index() -> None:
    service, _, client, _, _ = _service()
    options = service.model_options()

    assert [option.label for option in options] == [
        "mimo/mimo-v2.6-pro",
        "deepseek/deepseek-flash",
        "siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        REPLY_PEAK_COMBO_LABEL,
    ]
    assert options[-1].preset == REPLY_PEAK_COMBO_KEY
    assert options[-1].routes == (
        service.llm.fallback_routes["reply"],
        service.llm.additional_fallback_routes["reply"][0],
    )
    assert service.select_option("2") == options[1]
    assert service.select_option("01").label == options[0].label
    with pytest.raises(ValueError, match=r"模型编号无效，请输入 1-4。"):
        service.select_option("999")
    with pytest.raises(ValueError, match="批处理模型只能用于后台记忆/复盘。"):
        service.select_option("openrouter/z-ai/glm-5.3-flash:batch")
    selected = service.select_option("siliconflow/MiniMaxAI/MiniMax-M2.5")
    assert selected.label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert client.probes == []


def test_route_override_roundtrip_filters_and_does_not_clear_bad_json(caplog) -> None:
    service, kv, _, _, _ = _service()
    kv.app_kv_set(MODEL_ROUTE_OVERRIDES_KEY, "{")
    with caplog.at_level(logging.WARNING, logger="qq_social_agent"):
        assert service.route_overrides() == {}
    assert kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY) == "{"
    assert "qq_social_agent invalid model route overrides json, clearing" in caplog.text

    kv.app_kv_set(MODEL_ROUTE_OVERRIDES_KEY, "[]")
    assert service.route_overrides() == {}
    kv.app_kv_set(
        MODEL_ROUTE_OVERRIDES_KEY,
        json.dumps({" reply ": " siliconflow/x ", "nope": "y", "search": "", "utility": "deepseek/u"}),
    )
    assert service.route_overrides() == {"reply": "siliconflow/x", "utility": "deepseek/u"}
    service.save_route_overrides({"search": "deepseek/s", "utility_group": "no", "reply": "硅基/模型"})
    assert kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY) == json.dumps(
        {"reply": "硅基/模型", "search": "deepseek/s"},
        ensure_ascii=False,
        sort_keys=True,
    )


def test_apply_saved_overrides_keeps_going_after_one_failure(caplog) -> None:
    service, kv, client, _, _ = _service()
    kv.app_kv_set(
        MODEL_ROUTE_OVERRIDES_KEY,
        json.dumps(
            {
                "reply": REPLY_PEAK_COMBO_KEY,
                "search": "not-a-route",
                "memory": "siliconflow/memory-override",
            },
            sort_keys=True,
        ),
    )

    def parse_model_route(value: str, *, default_provider: str = "siliconflow") -> LLMModelRoute:
        if value == "not-a-route":
            raise ValueError("bad label")
        return Client.parse_model_route(client, value, default_provider=default_provider)

    client.parse_model_route = parse_model_route
    with caplog.at_level(logging.WARNING, logger="qq_social_agent"):
        service.apply_saved_overrides()

    assert client.peak is True
    assert client.overrides["memory"].label == "siliconflow/memory-override"
    assert "search" not in client.overrides
    assert "route=search label='not-a-route' error=bad label" in caplog.text
    service.client = None
    service.apply_saved_overrides()


def test_fallback_display_and_status_text_keep_client_label() -> None:
    service, kv, client, _, _ = _service()
    kv.app_kv_set(MODEL_ROUTE_OVERRIDES_KEY, json.dumps({"reply": REPLY_PEAK_COMBO_KEY}))
    client.current_route = lambda route_name: _route("mimo/live-not-config")

    assert service.active_label("reply") == REPLY_PEAK_COMBO_LABEL
    assert service.fallback_labels("reply", REPLY_PEAK_COMBO_LABEL) == ("mimo/reply-extra",)
    assert service.fallback_labels("reply", "deepseek/reply-model") == (
        "deepseek/official-a",
        "deepseek/official-b",
    )
    assert service.fallback_labels("search", "lingsuan/gpt-6.1-sol") == ("siliconflow/search-fallback",)
    assert service.fallback_labels("search", "siliconflow/live") == (
        "siliconflow/search-fallback",
        "mimo/search-extra",
    )
    assert service.provider_key_source("deepseek") == "DeepSeek 官方 key，第一次提供"
    assert service.provider_key_source("siliconflow") == "硅基流动 key，第二次提供"
    assert service.provider_key_source("mimo") == "mimo key"

    text = service.format_route_status()
    assert text.splitlines()[:6] == [
        "模型状态：",
        "可切换部分：",
        "- 决策模型（decision）：群聊是否插嘴、action、是否需要联网搜索",
        "  当前：mimo/live-not-config （配置）",
        "  config：deepseek/decision-model",
        "  fallback：siliconflow/decision-fallback",
    ]
    assert "  当前：高峰 DeepSeek/SiliconFlow 组合 （覆盖）" in text
    assert "  fallback：mimo/reply-extra" in text
    assert "兼容命令：切工具模型 <模型> = 同时切黑话/记忆/风格/画像。" in text
    assert "1. mimo/mimo-v2.6-pro（mimo key / MIMO_API_KEY）" in text
    assert "4. 高峰 DeepSeek/SiliconFlow 组合（平峰 DeepSeek，高峰 SiliconFlow）" in text
    assert text.endswith("命令：测试模型（检测清单）；测试模型 1（单测）；切回复模型 1；清模型覆盖。编号与上方列表对应。")


def test_background_selection_batch_and_sync_defaults() -> None:
    service, kv, _, _, _ = _service()

    assert service.background_selection("memory") == "verysadai/gpt-6.1-sol"
    assert service.background_selection("review") == "lingsuan/gpt-6.1-sol"
    assert service.background_sync_model("memory") == LLMModelRoute("siliconflow", "verysadai/gpt-6.1-sol")
    assert not service.background_batch_enabled("memory")
    kv.app_kv_set(BACKGROUND_MODEL_OVERRIDES_KEY, "{")
    assert service.background_overrides() == {}
    kv.app_kv_set(
        BACKGROUND_MODEL_OVERRIDES_KEY,
        json.dumps({"memory": "missing", "review": "openrouter/z-ai/glm-5.3-flash:batch", "other": "x"}),
    )
    assert service.background_overrides() == {"review": "openrouter/z-ai/glm-5.3-flash:batch"}
    assert service.background_selection("memory") == "verysadai/gpt-6.1-sol"
    assert service.background_batch_enabled("review")
    assert service.background_sync_model("review") is None
    assert service.background_sync_model("absent") is None
    status = service.format_background_status()
    assert "- 后台记忆：verysadai/gpt-6.1-sol" in status
    assert "2. openrouter/z-ai/glm-5.3-flash:batch（异步批处理）" in status
    assert "3. siliconflow/deepseek-ai/DeepSeek-V4-Flash（实时）" in status
    assert "批任务：queued 1、completed 2" in status
    assert status.endswith("命令：测试后台模型；切后台记忆模型 1；切后台复盘模型 1；清后台模型覆盖。")


def test_private_and_admin_share_switch_reset_without_sharing_messages() -> None:
    service, kv, client, _, sender = _service()

    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切回复模型 2"))
    assert client.overrides["reply"].label == "deepseek/deepseek-flash"
    assert sender.messages[-1] == (
        1535071184,
        "已切回复模型：deepseek/deepseek-flash\n影响路由：reply",
    )
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切搜索模型 siliconflow/MiniMaxAI/MiniMax-M2.5"))
    admin_text = service.apply_admin_action("model_route", {"route": "reply", "model": "1"})
    assert admin_text == "已切换模型：reply -> mimo/mimo-v2.6-pro。"
    assert client.overrides["reply"].label == "mimo/mimo-v2.6-pro"
    assert client.overrides["search"].label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert "search" in kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY)

    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切工具模型 deepseek/deepseek-flash"))
    assert [client.overrides[name].label for name in ("jargon", "memory", "style", "member_profile")] == [
        "deepseek/deepseek-flash"
    ] * 4
    assert sender.messages[-1][1].endswith("影响路由：jargon、memory、style、member_profile")
    assert service.apply_admin_action("model_route", {"route": "utility_group", "model": "2"}) == (
        "已切换模型：jargon, memory, style, member_profile -> deepseek/deepseek-flash。"
    )

    combo = service.model_options()[-1]
    service.switch_route("reply", combo)
    assert client.peak is True
    assert json.loads(kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY))["reply"] == REPLY_PEAK_COMBO_KEY
    assert service.apply_admin_action("model_route", {"route": "search", "model": str(len(service.model_options()))}) == (
        "高峰组合只适用于回复模型。"
    )
    assert asyncio.run(
        service.handle_private_command(object(), 1535071184, f"切风格模型 {len(service.model_options())}")
    )
    assert sender.messages[-1][1] == "这个组合只适用于回复模型。"

    assert service.reset_routes() == "已清除模型覆盖，恢复 config.yaml 默认模型。"
    assert kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY) == "{}"
    assert client.overrides == {}
    assert client.peak is False
    kv.app_kv_set(MODEL_ROUTE_OVERRIDES_KEY, '{"reply":"deepseek/deepseek-flash"}')
    client.set_route_override("reply", _route("deepseek/deepseek-flash"))
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "清模型覆盖"))
    assert sender.messages[-1][1] == "已清除模型覆盖，恢复 config.yaml 默认模型。"
    assert set(MODEL_ROUTE_STORAGE_NAMES) <= set(client.overrides) or client.overrides == {}


def test_private_command_texts_probe_order_and_concurrency() -> None:
    service, _, client, _, sender = _service()
    entered = 0
    release = asyncio.Event()

    async def gate(_route):
        nonlocal entered
        entered += 1
        if entered >= 3:
            release.set()
        await asyncio.wait_for(release.wait(), 1)

    client.probe_gate = gate
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试模型"))
    assert client.max_inflight == 3
    assert sender.messages[-1][1] == "\n".join(
        [
            "模型实测：",
            "1. ✅ mimo/mimo-v2.6-pro：可用",
            "2. ❌ deepseek/deepseek-flash：HTTP 403",
            "3. ❌ siliconflow/deepseek-ai/DeepSeek-V4-Flash：HTTP 403",
            "4. 高峰 DeepSeek/SiliconFlow 组合：siliconflow ❌HTTP 403 / mimo ✅可用",
        ]
    )
    sender.messages.clear()
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试模型 1"))
    assert sender.messages[-1] == (1535071184, "模型实测：\n1. ✅ mimo/mimo-v2.6-pro：可用")
    sender.messages.clear()
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试模型 siliconflow/custom"))
    assert sender.messages[-1][1] == "模型实测：\n❌ siliconflow/custom：HTTP 403"
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "模型状态"))
    assert sender.messages[-1][1].startswith("模型状态：")
    assert not asyncio.run(service.handle_private_command(object(), 1535071184, "你好"))
    assert service.route_name_from_text("回想") == "memory"
    assert service.route_name_from_text("Utility_Model") == "utility_group"


def test_private_errors_keep_distinct_catches_and_owner_text() -> None:
    service, kv, client, _, sender = _service()
    denied, _, _, _, denied_sender = _service(owner=False)
    assert asyncio.run(denied.handle_private_command(object(), 3370998238, "切回复模型 1"))
    assert denied_sender.messages[-1][1] == "只有主人能查询、测试和切换模型。"
    assert asyncio.run(denied.handle_background_command(object(), 3370998238, "后台模型状态"))
    assert denied_sender.messages[-1][1] == "只有主人能查询、测试和切换后台模型。"
    assert MODEL_ROUTE_OVERRIDES_KEY not in kv.data

    service.client = None
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试模型"))
    assert sender.messages[-1][1] == "模型客户端还没初始化，稍后再测。"
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切回复模型 1"))
    assert sender.messages[-1][1] == "模型客户端还没初始化，稍后再切。"
    assert service.apply_admin_action("model_route", {"route": "reply", "model": "1"}) == "模型客户端还没初始化。"
    assert service.apply_admin_action("model_route", {"route": "nope", "model": "1"}) == "模型客户端还没初始化。"

    service.client = client
    assert service.apply_admin_action("model_route", {"route": "nope", "model": "1"}) == "未知模型流程。"
    client.fail_parse = True
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(service.handle_private_command(object(), 1535071184, "切回复模型 custom-model"))
    assert service.apply_admin_action("model_route", {"route": "reply", "model": "custom-model"}) == (
        "模型路由解析失败：boom"
    )
    client.fail_parse = False
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切回复模型 999"))
    assert sender.messages[-1][1] == "模型路由解析失败：模型编号无效，请输入 1-4。"
    assert service.apply_admin_action("jargon_add", {}) is None


def test_background_commands_and_admin_share_switch_but_not_text() -> None:
    service, kv, client, batch, sender = _service()
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切后台记忆模型 3"))
    assert service.background_selection("memory") == "siliconflow/deepseek-ai/DeepSeek-V4-Flash"
    assert service.background_selection("review") == "lingsuan/gpt-6.1-sol"
    assert not service.background_batch_enabled("memory")
    assert sender.messages[-1][1] == "已切后台记忆模型：siliconflow/deepseek-ai/DeepSeek-V4-Flash"
    assert service.apply_admin_action(
        "background_model_route",
        {"route": "review", "model": "硅基/模型"},
    ) == "已切换后台模型：review -> 硅基/模型。"
    saved = json.loads(kv.app_kv_get(BACKGROUND_MODEL_OVERRIDES_KEY))
    assert saved == {
        "memory": "siliconflow/deepseek-ai/DeepSeek-V4-Flash",
        "review": "硅基/模型",
    }
    assert "\\u" not in kv.app_kv_get(BACKGROUND_MODEL_OVERRIDES_KEY)
    assert service.apply_admin_action("background_model_route", {"route": "other", "model": "硅基/模型"}) == (
        "后台模型或任务组无效。"
    )
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "切后台复盘模型 99"))
    assert sender.messages[-1][1] == "后台模型编号无效，请输入 1-4。"
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试后台模型 nope"))
    assert sender.messages[-1][1] == "后台模型不在可切换清单中。"

    batch.result = SimpleNamespace(status="queued", content=None, error=None)
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试后台模型 2"))
    assert "2. openrouter/z-ai/glm-5.3-flash:batch：⏳ queued，等待批处理完成" in sender.messages[-1][1]
    batch.result = SimpleNamespace(status="failed", content=None, error=None)
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试后台模型 2"))
    assert "❌ failed: 无有效回复" in sender.messages[-1][1]
    service.client = None
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试后台模型 硅基/模型"))
    assert "❌ 模型客户端尚未初始化" in sender.messages[-1][1]
    service.client = client

    async def explode(_route):
        raise RuntimeError("a\n\n" + ("x" * 100))

    client.probe_model = explode
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "测试后台模型 3"))
    line = sender.messages[-1][1].splitlines()[-1]
    assert line.startswith("3. siliconflow/deepseek-ai/DeepSeek-V4-Flash：❌ RuntimeError: a ")
    assert line.endswith("…")
    assert len(line.split("RuntimeError: ", 1)[1]) == 80

    kv.app_kv_set(
        BACKGROUND_MODEL_OVERRIDES_KEY,
        json.dumps({"memory": "missing", "review": "硅基/模型", "other": "x"}, ensure_ascii=False),
    )
    service.switch_background("memory", "硅基/模型")
    assert json.loads(kv.app_kv_get(BACKGROUND_MODEL_OVERRIDES_KEY)) == {
        "memory": "硅基/模型",
        "review": "硅基/模型",
    }
    assert service.reset_background() == "已恢复后台记忆和复盘的默认模型。"
    assert kv.app_kv_get(BACKGROUND_MODEL_OVERRIDES_KEY) == "{}"
    assert asyncio.run(service.handle_private_command(object(), 1535071184, "清后台模型覆盖"))
    assert sender.messages[-1][1] == "已恢复后台记忆和复盘的默认模型。"


def test_status_and_admin_state_skip_missing_route_and_keep_peak_label() -> None:
    service, kv, client, batch, _ = _service()
    kv.app_kv_set(MODEL_ROUTE_OVERRIDES_KEY, json.dumps({"reply": REPLY_PEAK_COMBO_KEY}))
    client.peak = True
    del service.llm.routes["utility"]

    def current_route(route_name: str) -> LLMModelRoute:
        if route_name == "utility":
            raise ValueError("unknown route: utility")
        return Client.current_route(client, route_name)

    client.current_route = current_route
    routes = service.status_routes()
    assert "utility" not in routes
    assert routes["reply"] == REPLY_PEAK_COMBO_LABEL
    assert routes["decision"] == "deepseek/decision-model"
    state = service.admin_model_state()
    reply_row = next(row for row in state["models"] if row["route"] == "reply")
    assert reply_row["active"] == REPLY_PEAK_COMBO_LABEL
    assert reply_row["overridden"] is True
    assert reply_row["fallback"] == "mimo/reply-extra"
    assert state["model_catalog"][-1]["source"] == "平峰 DeepSeek，高峰 SiliconFlow"
    assert state["background_model_catalog"][1] == "openrouter/z-ai/glm-5.3-flash:batch"
    assert state["background_batch_status"] == batch.counts
