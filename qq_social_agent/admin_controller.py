"""HTTP controllers and route assembly for local admin endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import FastAPI
from nonebot.adapters.onebot.v11 import Message
from nonebot.adapters.onebot.v11.exception import ActionFailed
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from .admin_ui import render_admin_dashboard, render_message_detail_page, render_plugins_page
from .admin_edit_controller import AdminEditController
from .admin_memory_controller import AdminMemoryController
from .admin_summaries_controller import AdminSummariesController
from .admin_tools_controller import AdminToolsController
from .memory import MemoryStore


@dataclass(frozen=True)
class AdminOperationsServices:
    is_local_admin_request: Callable[[Request], bool]
    first_connected_bot: Callable[[], object | None]
    target_groups: Callable[[], tuple[int, ...]]
    group_allowed: Callable[[int], bool]
    send_manual_daily_reviews: Callable[[object, str], Awaitable[tuple[int, int]]]
    send_proactive_chat_for_group: Callable[[object, int, int, float], Awaitable[bool]]
    send_group_message: Callable[[object, int, Message], Awaitable[object]]
    record_group_sent_message: Callable[..., None]
    send_private_message: Callable[..., Awaitable[object]]
    record_private_sent_message: Callable[[int, int, str], None]
    summarize_action_failed: Callable[[Exception], str]


class AdminOperationsController:
    def __init__(self, services: AdminOperationsServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/daily-review/{mode}", self.daily_review, methods=["POST"])
        app.add_api_route("/admin/proactive-chat", self.proactive_chat, methods=["POST"])
        app.add_api_route("/admin/send-group", self.send_group, methods=["POST"])
        app.add_api_route("/admin/send-private", self.send_private, methods=["POST"])

    async def daily_review_payload(self, *, mode: str) -> tuple[dict[str, object], int]:
        bot = self.services.first_connected_bot()
        if bot is None:
            return {"ok": False, "reason": "onebot_disconnected"}, 503
        normalized_mode = (mode or "today").strip().lower()
        sent_count, total_count = await self.services.send_manual_daily_reviews(bot, normalized_mode)
        ok = sent_count > 0
        return {
            "ok": ok,
            "mode": normalized_mode,
            "sent_count": sent_count,
            "target_count": total_count,
        }, 200 if ok else 503

    async def proactive_chat_payload(self, *, group_id: int | None) -> tuple[dict[str, object], int]:
        bot = self.services.first_connected_bot()
        if bot is None:
            return {"ok": False, "reason": "onebot_disconnected"}, 503
        target_groups = (group_id,) if group_id is not None else self.services.target_groups()
        sent = 0
        results: list[dict[str, object]] = []
        for target_group_id in target_groups:
            if not self.services.group_allowed(int(target_group_id)):
                results.append({"group_id": int(target_group_id), "ok": False, "reason": "group_not_allowed"})
                continue
            ok = await self.services.send_proactive_chat_for_group(bot, int(target_group_id), 100, 0.0)
            sent += 1 if ok else 0
            results.append({"group_id": int(target_group_id), "ok": bool(ok)})
        return {
            "ok": sent > 0,
            "sent_count": sent,
            "target_count": len(tuple(target_groups)),
            "results": results,
        }, 200 if sent > 0 else 503

    async def daily_review(self, request: Request, mode: str) -> JSONResponse:
        if not self.services.is_local_admin_request(request):
            return JSONResponse({"ok": False, "reason": "local_admin_only"}, status_code=403)
        payload, status_code = await self.daily_review_payload(mode=mode)
        return JSONResponse(payload, status_code=status_code)

    async def proactive_chat(self, request: Request) -> JSONResponse:
        if not self.services.is_local_admin_request(request):
            return JSONResponse({"ok": False, "reason": "local_admin_only"}, status_code=403)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            group_id = int(str(payload.get("group_id") or "").strip())
        except (TypeError, ValueError):
            group_id = None
        response, status_code = await self.proactive_chat_payload(group_id=group_id)
        return JSONResponse(response, status_code=status_code)

    async def send_group(self, request: Request) -> JSONResponse:
        if not self.services.is_local_admin_request(request):
            return JSONResponse({"ok": False, "reason": "local_admin_only"}, status_code=403)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            group_id = int(str(payload.get("group_id") or "").strip())
        except (TypeError, ValueError):
            group_id = None
        message_text = str(payload.get("message") or "").strip()
        if group_id is None or not message_text:
            return JSONResponse({"ok": False, "reason": "missing_group_id_or_message"}, status_code=400)
        bot = self.services.first_connected_bot()
        if bot is None:
            return JSONResponse({"ok": False, "reason": "onebot_disconnected"}, status_code=503)
        message_id = await self.services.send_group_message(bot, group_id, Message(message_text))
        self.services.record_group_sent_message(
            group_id=group_id,
            message_id=message_id,
            bot_reply=message_text,
            trigger_user_id=0,
            trigger_nickname="Codex手动发起",
            trigger_text=str(payload.get("reason") or "manual proactive topic")[:500],
            action="manual_proactive",
        )
        return JSONResponse({"ok": True, "group_id": group_id, "message_id": message_id})

    async def send_private(self, request: Request) -> JSONResponse:
        if not self.services.is_local_admin_request(request):
            return JSONResponse({"ok": False, "reason": "local_admin_only"}, status_code=403)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            user_id = int(str(payload.get("user_id") or "").strip())
        except (TypeError, ValueError):
            user_id = None
        message_text = str(payload.get("message") or "").strip()
        if user_id is None or not message_text:
            return JSONResponse({"ok": False, "reason": "missing_user_id_or_message"}, status_code=400)
        bot = self.services.first_connected_bot()
        if bot is None:
            return JSONResponse({"ok": False, "reason": "onebot_disconnected"}, status_code=503)
        try:
            result = await self.services.send_private_message(bot, user_id=user_id, message=Message(message_text))
        except ActionFailed as exc:
            return JSONResponse(
                {"ok": False, "reason": self.services.summarize_action_failed(exc)},
                status_code=502,
            )
        self.services.record_private_sent_message(user_id, int(bot.self_id), message_text)
        return JSONResponse({"ok": True, "user_id": user_id, "result": str(result)[:160]})


@dataclass(frozen=True)
class AdminDashboardServices:
    is_local_admin_request: Callable[[Request], bool]
    get_memory: Callable[[], MemoryStore]
    target_groups: Callable[[], tuple[int, ...]]
    ready_payload: Callable[[], dict[str, object]]
    health_payload: Callable[[], dict[str, object]]
    status_payload: Callable[[], dict[str, object]]
    model_routes: Callable[[], dict[str, str]]
    pending_approvals: Callable[[], list[object]]
    plugins_summary: Callable[[], list[dict[str, object]]]


class AdminDashboardController:
    def __init__(self, services: AdminDashboardServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin", self.page, methods=["GET"])

    async def page(self, request: Request, group_id: int | None = None) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(
            render_admin_dashboard(
                memory=self.services.get_memory(),
                groups=self.services.target_groups(),
                selected_group_id=group_id,
                ready=self.services.ready_payload(),
                health=self.services.health_payload(),
                status=self.services.status_payload(),
                model_routes=self.services.model_routes(),
                pending_approvals=self.services.pending_approvals(),
                plugins=self.services.plugins_summary(),
            )
        )


@dataclass(frozen=True)
class AdminPluginsServices:
    is_local_admin_request: Callable[[Request], bool]
    reload_plugins: Callable[[], None]
    plugins_summary: Callable[[], list[dict[str, object]]]
    plugin_errors: Callable[[], list[dict[str, object]]]


class AdminPluginsController:
    def __init__(self, services: AdminPluginsServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/plugins", self.page, methods=["GET"])

    async def page(self, request: Request) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        self.services.reload_plugins()
        return HTMLResponse(
            render_plugins_page(
                plugins=self.services.plugins_summary(),
                errors=self.services.plugin_errors(),
            )
        )


@dataclass(frozen=True)
class AdminMessageServices:
    is_local_admin_request: Callable[[Request], bool]
    get_memory: Callable[[], MemoryStore]


class AdminMessageController:
    def __init__(self, services: AdminMessageServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/messages/{message_id}", self.detail, methods=["GET"])

    async def detail(self, request: Request, message_id: int) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(render_message_detail_page(memory=self.services.get_memory(), message_id=message_id))


@dataclass(frozen=True)
class AdminController:
    operations: AdminOperationsController
    dashboard: AdminDashboardController
    tools: AdminToolsController
    edit: AdminEditController
    summaries: AdminSummariesController
    memory: AdminMemoryController
    plugins: AdminPluginsController
    messages: AdminMessageController

    def register(self, app: FastAPI) -> None:
        self.operations.register(app)
        self.dashboard.register(app)
        self.tools.register(app)
        self.edit.register(app)
        self.summaries.register(app)
        self.memory.register_private_memory(app)
        self.plugins.register(app)
        self.messages.register(app)
        self.memory.register_memory_audit(app)
