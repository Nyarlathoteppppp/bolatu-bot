"""HTTP controller for the admin tools page and actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlencode

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from .admin_http import admin_form_data, admin_form_int
from .admin_ui import render_admin_tools_page


@dataclass(frozen=True)
class AdminToolsServices:
    is_local_admin_request: Callable[[Request], bool]
    target_groups: Callable[[], tuple[int, ...]]
    state_for_group: Callable[[int | None], dict[str, object]]
    apply_action: Callable[[dict[str, str], int | None], Awaitable[str]]
    report: Callable[..., Awaitable[tuple[str, str]]]


class AdminToolsController:
    def __init__(self, services: AdminToolsServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/tools", self.page, methods=["GET"])
        app.add_api_route("/admin/tools/report", self.report_page, methods=["GET"])
        app.add_api_route("/admin/tools/action", self.apply_action, methods=["POST"])

    def selected_group_id(self, group_id: int | None = None) -> int | None:
        if group_id is not None:
            return int(group_id)
        groups = self.services.target_groups()
        return groups[0] if groups else None

    @staticmethod
    def tools_url(*, group_id: int | None = None, notice: str = "") -> str:
        params: dict[str, object] = {}
        if group_id is not None:
            params["group_id"] = group_id
        if notice.strip():
            params["notice"] = notice.strip()
        return "/admin/tools" + ("?" + urlencode(params) if params else "")

    async def page(
        self,
        request: Request,
        group_id: int | None = None,
        notice: str = "",
    ) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        selected_group_id = self.selected_group_id(group_id)
        return HTMLResponse(
            render_admin_tools_page(
                state=self.services.state_for_group(selected_group_id),
                selected_group_id=selected_group_id,
                notice=notice,
            )
        )

    async def report_page(
        self,
        request: Request,
        kind: str = "metrics",
        group_id: int | None = None,
        limit: int = 20,
        window: str = "today",
        query: str = "",
    ) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        selected_group_id = self.selected_group_id(group_id)
        title, report_text = await self.services.report(
            kind=kind,
            group_id=selected_group_id,
            limit=limit,
            window=window,
            query=query,
        )
        return HTMLResponse(
            render_admin_tools_page(
                state=self.services.state_for_group(selected_group_id),
                selected_group_id=selected_group_id,
                report_title=title,
                report_text=report_text,
            )
        )

    async def apply_action(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        group_id = self.selected_group_id(admin_form_int(form, "group_id"))
        notice = await self.services.apply_action(form, group_id)
        return RedirectResponse(url=self.tools_url(group_id=group_id, notice=notice), status_code=303)
