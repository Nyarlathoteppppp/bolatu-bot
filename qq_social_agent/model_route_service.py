"""Owner model-route commands and the shared override store.

QQ private commands and admin tool actions both call select/switch/reset.
This module does not choose a route for generation and does not import plugin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .config import LLMConfig, LLMModelRoute, parse_llm_model_route
from .speaker_context import _short_notice_text


MODEL_ROUTE_OVERRIDES_KEY = "llm_model_route_overrides"
REPLY_PEAK_COMBO_KEY = "reply_peak_deepseek_siliconflow"
REPLY_PEAK_COMBO_LABEL = "高峰 DeepSeek/SiliconFlow 组合"
MODEL_ROUTE_STATUS_COMMANDS = {"模型状态", "模型", "model status", "/模型状态"}
MODEL_ROUTE_RESET_COMMANDS = {"清模型覆盖", "清除模型覆盖", "重置模型", "恢复默认模型", "model reset", "/清模型覆盖"}
MODEL_PROBE_COMMAND_RE = re.compile(r"^(?:/)?(?:测试模型|检测模型|model test)(?:\s+(?P<model>\S+))?$", re.IGNORECASE)
MODEL_ROUTE_COMMAND_RE = re.compile(
    r"^(?:/)?(?:切|设置|更换|改)?(?P<target>回复|reply|搜索|search|决策|decision|黑话|jargon|记忆|memory|回想|风格|style|学习|style_learning|画像|群友画像|member_profile|profile|工具|utility|utility_model)模型\s+"
    r"(?P<model>\S+)$",
    re.IGNORECASE,
)
BACKGROUND_MODEL_OVERRIDES_KEY = "llm_background_model_overrides"
BACKGROUND_MODEL_STATUS_COMMANDS = {"后台模型状态", "后台模型", "后台模型列表"}
BACKGROUND_MODEL_RESET_COMMANDS = {"清后台模型覆盖", "重置后台模型"}
BACKGROUND_MODEL_PROBE_RE = re.compile(r"^(?:测试|检测)后台模型(?:\s+(?P<model>\S+))?$")
BACKGROUND_MODEL_COMMAND_RE = re.compile(r"^(?:切|设置|更换|改)后台(?P<target>记忆|复盘)模型\s+(?P<model>\S+)$")
MODEL_ROUTE_INFOS = (
    ("decision", "决策", "群聊是否插嘴、action、是否需要联网搜索"),
    ("reply", "回复", "私聊回复、群聊审批三候选生成"),
    ("search", "搜索回复", "消化联网事实并生成精简回答"),
    ("jargon", "黑话", "黑话词典注入选择"),
    ("memory", "记忆", "中期聊天回想压缩"),
    ("style", "风格", "群聊表达风格学习"),
    ("member_profile", "画像", "群友长期画像摘要"),
)
MODEL_ROUTE_NAMES = tuple(route_name for route_name, _, _ in MODEL_ROUTE_INFOS)
MODEL_ROUTE_STORAGE_NAMES = (*MODEL_ROUTE_NAMES, "utility")
UTILITY_GROUP_ROUTE_NAMES = ("jargon", "memory", "style", "member_profile")
ROUTE_RESET_MESSAGE = "已清除模型覆盖，恢复 config.yaml 默认模型。"
BACKGROUND_RESET_MESSAGE = "已恢复后台记忆和复盘的默认模型。"
_ROUTE_NAME_FROM_TEXT = {
    "回复": "reply",
    "reply": "reply",
    "搜索": "search",
    "search": "search",
    "决策": "decision",
    "decision": "decision",
    "黑话": "jargon",
    "jargon": "jargon",
    "记忆": "memory",
    "memory": "memory",
    "回想": "memory",
    "风格": "style",
    "style": "style",
    "学习": "style",
    "style_learning": "style",
    "画像": "member_profile",
    "群友画像": "member_profile",
    "member_profile": "member_profile",
    "profile": "member_profile",
    "工具": "utility_group",
    "utility": "utility_group",
    "utility_model": "utility_group",
}


@dataclass(frozen=True)
class ModelOption:
    label: str
    routes: tuple[LLMModelRoute, ...]
    preset: str = ""


class ModelRouteStore(Protocol):
    def app_kv_get(self, key: str) -> str | None: ...

    def app_kv_set(self, key: str, value: str) -> None: ...


class ModelRouteClient(Protocol):
    def parse_model_route(self, value: str, *, default_provider: str = "siliconflow") -> LLMModelRoute: ...

    def set_route_override(self, route_name: str, route: LLMModelRoute | None) -> None: ...

    def set_reply_peak_combo(self) -> None: ...

    def current_route(self, route_name: str) -> LLMModelRoute: ...

    async def probe_model(self, route: LLMModelRoute) -> tuple[bool, str]: ...


class BackgroundBatchService(Protocol):
    async def probe(self) -> Any: ...

    def status_snapshot(self) -> Mapping[str, Any]: ...


class BackgroundModelSettings:
    """Background selections depend only on persisted settings, not a live LLM client."""

    def __init__(self, *, kv: ModelRouteStore, config: Mapping[str, Any] | None) -> None:
        self.kv = kv
        self.config = config if isinstance(config, dict) else {}

    def catalog(self) -> tuple[str, ...]:
        raw = self.config.get("catalog", ())
        return tuple(str(value).strip() for value in raw if str(value).strip()) if isinstance(raw, list) else ()

    def overrides(self) -> dict[str, str]:
        raw = self.kv.app_kv_get(BACKGROUND_MODEL_OVERRIDES_KEY)
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
        catalog = self.catalog()
        if not isinstance(data, dict):
            return {}
        return {
            key: value
            for key, value in data.items()
            if key in {"memory", "review"} and value in catalog
        }

    def selection(self, group: str) -> str:
        return self.overrides().get(group, str(self.config.get(group, "")))

    def batch_enabled(self, group: str) -> bool:
        return self.selection(group).endswith(":batch")

    def switch(self, group: str, label: str) -> None:
        overrides = self.overrides()
        overrides[group] = label
        self.kv.app_kv_set(
            BACKGROUND_MODEL_OVERRIDES_KEY,
            json.dumps(overrides, ensure_ascii=False, sort_keys=True),
        )

    def reset(self) -> str:
        self.kv.app_kv_set(BACKGROUND_MODEL_OVERRIDES_KEY, "{}")
        return BACKGROUND_RESET_MESSAGE


class ModelRouteService:
    def __init__(
        self,
        *,
        llm: LLMConfig,
        background_models: Mapping[str, Any] | None,
        kv: ModelRouteStore,
        client: ModelRouteClient | None,
        batch_service: BackgroundBatchService,
        is_owner: Callable[[int], bool],
        send_private_text: Callable[[Any, int, str], Awaitable[None]],
        logger: Any | None = None,
    ) -> None:
        self.llm = llm
        self.background = BackgroundModelSettings(kv=kv, config=background_models)
        self.kv = kv
        self.client = client
        self.batch_service = batch_service
        self.is_owner = is_owner
        self.send_private_text = send_private_text
        self.logger = logger if logger is not None else logging.getLogger("qq_social_agent")

    def route_overrides(self) -> dict[str, str]:
        raw = self.kv.app_kv_get(MODEL_ROUTE_OVERRIDES_KEY)
        if raw is None:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("qq_social_agent invalid model route overrides json, clearing")
            return {}
        if not isinstance(data, dict):
            return {}
        overrides: dict[str, str] = {}
        for route_name, route_label in data.items():
            route = str(route_name).strip()
            label = str(route_label).strip()
            if route in MODEL_ROUTE_STORAGE_NAMES and label:
                overrides[route] = label
        return overrides

    def save_route_overrides(self, overrides: Mapping[str, str]) -> None:
        cleaned = {
            route_name: route_label
            for route_name, route_label in overrides.items()
            if route_name in MODEL_ROUTE_STORAGE_NAMES and route_label
        }
        self.kv.app_kv_set(
            MODEL_ROUTE_OVERRIDES_KEY,
            json.dumps(cleaned, ensure_ascii=False, sort_keys=True),
        )

    def apply_saved_overrides(self) -> None:
        if self.client is None:
            return
        for route_name, route_label in self.route_overrides().items():
            try:
                if route_name == "reply" and route_label == REPLY_PEAK_COMBO_KEY:
                    self.client.set_reply_peak_combo()
                    continue
                self.client.set_route_override(
                    route_name,
                    self.client.parse_model_route(route_label, default_provider="siliconflow"),
                )
            except Exception as exc:
                self.logger.warning(
                    "qq_social_agent failed applying model route override: "
                    f"route={route_name} label={route_label!r} error={exc}"
                )

    def model_options(self) -> tuple[ModelOption, ...]:
        models = tuple(ModelOption(route.label, (route,)) for route in self.llm.model_catalog)
        combo_routes = (
            self.llm.fallback_routes["reply"],
            *self.llm.additional_fallback_routes["reply"],
        )
        return (*models, ModelOption(REPLY_PEAK_COMBO_LABEL, combo_routes, REPLY_PEAK_COMBO_KEY))

    def select_option(self, value: str) -> ModelOption:
        if value.isdecimal():
            index = int(value) - 1
            options = self.model_options()
            if not 0 <= index < len(options):
                raise ValueError(f"模型编号无效，请输入 1-{len(options)}。")
            return options[index]
        if self.client is None:
            raise ValueError("模型客户端还没初始化。")
        if value.endswith(":batch"):
            raise ValueError("批处理模型只能用于后台记忆/复盘。")
        route = self.client.parse_model_route(value, default_provider="siliconflow")
        return ModelOption(route.label, (route,))

    def provider_key_source(self, provider_name: str) -> str:
        if provider_name == "deepseek":
            return "DeepSeek 官方 key，第一次提供"
        if provider_name == "siliconflow":
            return "硅基流动 key，第二次提供"
        return f"{provider_name} key"

    def route_name_from_text(self, target: str) -> str | None:
        return _ROUTE_NAME_FROM_TEXT.get(target.strip().casefold())

    def active_label(self, route_name: str, overrides: Mapping[str, str] | None = None) -> str:
        saved = self.route_overrides() if overrides is None else overrides
        if route_name == "reply" and saved.get(route_name) == REPLY_PEAK_COMBO_KEY:
            return REPLY_PEAK_COMBO_LABEL
        if self.client is not None:
            return self.client.current_route(route_name).label
        return saved.get(route_name, self.llm.routes[route_name].label)

    def fallback_labels(self, route_name: str, active: str) -> tuple[str, ...]:
        fallback = self.llm.fallback_routes[route_name]
        if active == REPLY_PEAK_COMBO_LABEL:
            return tuple(route.label for route in self.llm.additional_fallback_routes.get(route_name, ()))
        route = parse_llm_model_route(active, self.llm.providers, default_provider="deepseek")
        provider = self.llm.providers[route.provider]
        if route_name == "reply" and provider.reply_fallback_models:
            return provider.reply_fallback_models
        if route.provider == "lingsuan":
            return (fallback.label,)
        return (
            fallback.label,
            *(route.label for route in self.llm.additional_fallback_routes.get(route_name, ())),
        )

    def format_route_status(self) -> str:
        overrides = self.route_overrides()
        lines = ["模型状态：", "可切换部分："]
        for route_name, title, flow in MODEL_ROUTE_INFOS:
            configured = self.llm.routes[route_name].label
            active = self.active_label(route_name, overrides)
            suffix = "（覆盖）" if route_name in overrides else "（配置）"
            lines.append(f"- {title}模型（{route_name}）：{flow}")
            lines.append(f"  当前：{active} {suffix}")
            lines.append(f"  config：{configured}")
            fallback_chain = " → ".join(self.fallback_labels(route_name, active))
            lines.append(f"  fallback：{fallback_chain}")
        lines.append("兼容命令：切工具模型 <模型> = 同时切黑话/记忆/风格/画像。")
        lines.append("")
        lines.append("可切换模型：")
        for index, option in enumerate(self.model_options(), start=1):
            if option.preset:
                lines.append(f"{index}. {option.label}（平峰 DeepSeek，高峰 SiliconFlow）")
                continue
            route = option.routes[0]
            provider = self.llm.providers[route.provider]
            lines.append(
                f"{index}. {option.label}（{self.provider_key_source(provider.name)} / {provider.api_key_env}）"
            )
        lines.append("")
        lines.append("命令：测试模型（检测清单）；测试模型 1（单测）；切回复模型 1；清模型覆盖。编号与上方列表对应。")
        return "\n".join(lines)

    def background_catalog(self) -> tuple[str, ...]:
        return self.background.catalog()

    def background_overrides(self) -> dict[str, str]:
        return self.background.overrides()

    def background_selection(self, group: str) -> str:
        return self.background.selection(group)

    def background_batch_enabled(self, group: str) -> bool:
        return self.background.batch_enabled(group)

    def background_sync_model(self, group: str) -> LLMModelRoute | None:
        label = self.background_selection(group)
        if not label or label.endswith(":batch"):
            return None
        return parse_llm_model_route(label, self.llm.providers, default_provider="siliconflow")

    def format_background_status(self) -> str:
        lines = ["后台模型状态："]
        for group, name in (("memory", "记忆"), ("review", "复盘")):
            lines.append(f"- 后台{name}：{self.background_selection(group)}")
        lines.append("可切换后台模型：")
        for number, label in enumerate(self.background_catalog(), 1):
            lines.append(f"{number}. {label}{'（异步批处理）' if label.endswith(':batch') else '（实时）'}")
        counts = self.batch_service.status_snapshot().get("counts", {})
        if counts:
            lines.append("批任务：" + "、".join(f"{status} {count}" for status, count in counts.items()))
        lines.append("命令：测试后台模型；切后台记忆模型 1；切后台复盘模型 1；清后台模型覆盖。")
        return "\n".join(lines)

    def switch_route(self, route_name: str, option: ModelOption) -> tuple[str, ...]:
        target_routes = UTILITY_GROUP_ROUTE_NAMES if route_name == "utility_group" else (route_name,)
        overrides = self.route_overrides()
        if option.preset:
            self.client.set_reply_peak_combo()
            overrides["reply"] = option.preset
        else:
            for target_route in target_routes:
                self.client.set_route_override(target_route, option.routes[0])
                overrides[target_route] = option.routes[0].label
        self.save_route_overrides(overrides)
        return target_routes

    def reset_routes(self) -> str:
        self.save_route_overrides({})
        if self.client is not None:
            for route_name in MODEL_ROUTE_STORAGE_NAMES:
                self.client.set_route_override(route_name, None)
        return ROUTE_RESET_MESSAGE

    def switch_background(self, group: str, label: str) -> None:
        self.background.switch(group, label)

    def reset_background(self) -> str:
        return self.background.reset()

    def status_routes(self) -> dict[str, str]:
        overrides = self.route_overrides()
        routes: dict[str, str] = {}
        for route_name in MODEL_ROUTE_STORAGE_NAMES:
            if route_name == "utility_group":
                continue
            try:
                route = (
                    self.client.current_route(route_name)
                    if self.client is not None
                    else self.llm.routes.get(route_name)
                )
            except Exception:
                route = self.llm.routes.get(route_name)
            if route is not None:
                routes[route_name] = self.active_label(route_name, overrides)
        return routes

    def admin_model_state(self) -> dict[str, object]:
        overrides = self.route_overrides()
        model_rows: list[dict[str, object]] = []
        for route_name, title, flow in MODEL_ROUTE_INFOS:
            configured = self.llm.routes[route_name].label
            active = self.active_label(route_name, overrides)
            model_rows.append(
                {
                    "route": route_name,
                    "title": title,
                    "flow": flow,
                    "active": active,
                    "configured": configured,
                    "fallback": " → ".join(self.fallback_labels(route_name, active)),
                    "overridden": route_name in overrides,
                }
            )
        return {
            "models": model_rows,
            "model_catalog": [
                {
                    "label": option.label,
                    "source": (
                        "平峰 DeepSeek，高峰 SiliconFlow"
                        if option.preset
                        else (
                            f"{self.provider_key_source(option.routes[0].provider)} / "
                            f"{self.llm.providers[option.routes[0].provider].api_key_env}"
                        )
                    ),
                }
                for option in self.model_options()
            ],
            "background_models": [
                {"group": group, "title": title, "active": self.background_selection(group)}
                for group, title in (("memory", "后台记忆"), ("review", "后台复盘"))
            ],
            "background_model_catalog": list(self.background_catalog()),
            "background_batch_status": self.batch_service.status_snapshot().get("counts", {}),
        }

    def apply_admin_action(self, action: str, form: Mapping[str, str]) -> str | None:
        if action == "model_reset":
            return self.reset_routes()
        if action == "model_route":
            if self.client is None:
                return "模型客户端还没初始化。"
            route_name = form.get("route", "").strip()
            route_label = form.get("model", "").strip()
            if route_name not in (*MODEL_ROUTE_NAMES, "utility_group"):
                return "未知模型流程。"
            try:
                option = self.select_option(route_label)
            except Exception as exc:
                return f"模型路由解析失败：{exc}"
            if option.preset and route_name != "reply":
                return "高峰组合只适用于回复模型。"
            target_routes = self.switch_route(route_name, option)
            return f"已切换模型：{', '.join(target_routes)} -> {option.label}。"
        if action == "background_model_route":
            group = form.get("route", "").strip()
            label = form.get("model", "").strip()
            if group not in {"memory", "review"} or label not in self.background_catalog():
                return "后台模型或任务组无效。"
            self.switch_background(group, label)
            return f"已切换后台模型：{group} -> {label}。"
        if action == "background_model_reset":
            return self.reset_background()
        return None

    async def handle_background_command(self, bot: Any, user_id: int, text: str) -> bool:
        switch_match = BACKGROUND_MODEL_COMMAND_RE.match(text)
        probe_match = BACKGROUND_MODEL_PROBE_RE.match(text)
        if (
            text not in BACKGROUND_MODEL_STATUS_COMMANDS | BACKGROUND_MODEL_RESET_COMMANDS
            and switch_match is None
            and probe_match is None
        ):
            return False
        if not self.is_owner(user_id):
            await self.send_private_text(bot, user_id, "只有主人能查询、测试和切换后台模型。")
            return True
        if text in BACKGROUND_MODEL_STATUS_COMMANDS:
            await self.send_private_text(bot, user_id, self.format_background_status())
            return True
        if text in BACKGROUND_MODEL_RESET_COMMANDS:
            await self.send_private_text(bot, user_id, self.reset_background())
            return True
        catalog = self.background_catalog()
        if switch_match is not None:
            value = switch_match.group("model")
            index = int(value) - 1 if value.isdecimal() else -1
            label = catalog[index] if 0 <= index < len(catalog) else value if value in catalog else ""
            if not label:
                await self.send_private_text(bot, user_id, f"后台模型编号无效，请输入 1-{len(catalog)}。")
                return True
            group = "memory" if switch_match.group("target") == "记忆" else "review"
            self.switch_background(group, label)
            await self.send_private_text(bot, user_id, f"已切后台{switch_match.group('target')}模型：{label}")
            return True
        value = (probe_match.group("model") or "").strip()
        if value.isdecimal():
            index = int(value) - 1
            if not 0 <= index < len(catalog):
                await self.send_private_text(bot, user_id, f"后台模型编号无效，请输入 1-{len(catalog)}。")
                return True
            targets = ((index + 1, catalog[index]),)
        elif value:
            targets = ((catalog.index(value) + 1, value),) if value in catalog else ()
        else:
            targets = tuple(enumerate(catalog, 1))
        if not targets:
            await self.send_private_text(bot, user_id, "后台模型不在可切换清单中。")
            return True
        lines = ["后台模型实测："]
        for number, label in targets:
            try:
                if label.endswith(":batch"):
                    result = await self.batch_service.probe()
                    status = (
                        "✅ 可用"
                        if result.status == "completed" and result.content
                        else (
                            f"⏳ {result.status}，等待批处理完成"
                            if result.status in {"queued", "in_progress"}
                            else f"❌ {result.status}: {result.error or '无有效回复'}"
                        )
                    )
                elif self.client is None:
                    status = "❌ 模型客户端尚未初始化"
                else:
                    route = parse_llm_model_route(label, self.llm.providers, default_provider="siliconflow")
                    available, reason = await self.client.probe_model(route)
                    status = f"{'✅' if available else '❌'} {reason}"
            except Exception as exc:
                status = f"❌ {type(exc).__name__}: {_short_notice_text(str(exc), 80)}"
            lines.append(f"{number}. {label}：{status}")
        await self.send_private_text(bot, user_id, "\n".join(lines))
        return True

    async def handle_private_command(self, bot: Any, user_id: int, text: str) -> bool:
        if await self.handle_background_command(bot, user_id, text):
            return True
        route_match = MODEL_ROUTE_COMMAND_RE.match(text)
        probe_match = MODEL_PROBE_COMMAND_RE.match(text)
        if (
            text not in MODEL_ROUTE_STATUS_COMMANDS | MODEL_ROUTE_RESET_COMMANDS
            and route_match is None
            and probe_match is None
        ):
            return False
        if not self.is_owner(user_id):
            await self.send_private_text(bot, user_id, "只有主人能查询、测试和切换模型。")
            return True
        if text in MODEL_ROUTE_STATUS_COMMANDS:
            await self.send_private_text(bot, user_id, self.format_route_status())
            return True
        if probe_match is not None:
            if self.client is None:
                await self.send_private_text(bot, user_id, "模型客户端还没初始化，稍后再测。")
                return True
            model_label = (probe_match.group("model") or "").strip()
            if model_label:
                try:
                    options = (self.select_option(model_label),)
                except ValueError as exc:
                    await self.send_private_text(bot, user_id, str(exc))
                    return True
            else:
                options = self.model_options()
            semaphore = asyncio.Semaphore(3)
            catalog_numbers = {option.label: index for index, option in enumerate(self.model_options(), start=1)}

            async def _probe_one(route: LLMModelRoute) -> tuple[bool, str]:
                async with semaphore:
                    return await self.client.probe_model(route)

            probe_tasks = {
                route.label: asyncio.create_task(_probe_one(route))
                for option in options
                for route in option.routes
            }
            probe_results = dict(zip(probe_tasks, await asyncio.gather(*probe_tasks.values())))
            results = []
            for option in options:
                number = catalog_numbers.get(option.label)
                prefix = f"{number}. " if number is not None else ""
                if option.preset:
                    details = " / ".join(
                        f"{route.provider} {'✅' if probe_results[route.label][0] else '❌'}{probe_results[route.label][1]}"
                        for route in option.routes
                    )
                    results.append(f"{prefix}{option.label}：{details}")
                else:
                    available, reason = probe_results[option.routes[0].label]
                    results.append(f"{prefix}{'✅' if available else '❌'} {option.label}：{reason}")
            await self.send_private_text(bot, user_id, "模型实测：\n" + "\n".join(results))
            return True
        if text in MODEL_ROUTE_RESET_COMMANDS:
            await self.send_private_text(bot, user_id, self.reset_routes())
            return True
        match = route_match
        if self.client is None:
            await self.send_private_text(bot, user_id, "模型客户端还没初始化，稍后再切。")
            return True
        route_name = self.route_name_from_text(match.group("target"))
        if route_name is None:
            await self.send_private_text(bot, user_id, "未知模型类型，只能切 决策/回复/搜索/黑话/记忆/风格/画像/工具 模型。")
            return True
        route_label = match.group("model").strip()
        try:
            option = self.select_option(route_label)
        except ValueError as exc:
            await self.send_private_text(bot, user_id, f"模型路由解析失败：{exc}")
            return True
        if option.preset and route_name != "reply":
            await self.send_private_text(bot, user_id, "这个组合只适用于回复模型。")
            return True
        target_routes = self.switch_route(route_name, option)
        target_label = "、".join(target_routes)
        await self.send_private_text(
            bot,
            user_id,
            f"已切{match.group('target')}模型：{option.label}\n影响路由：{target_label}",
        )
        return True
