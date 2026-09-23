"""HTTP controller for manually maintained group memory summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from .admin_http import admin_form_data, admin_form_int
from .admin_ui import render_memory_summaries_page, render_memory_summary_detail_page
from .memory import MemoryStore


@dataclass(frozen=True)
class AdminSummariesServices:
    is_local_admin_request: Callable[[Request], bool]
    get_memory: Callable[[], MemoryStore]
    target_groups: Callable[[], tuple[int, ...]]


class AdminSummariesController:
    def __init__(self, services: AdminSummariesServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/summaries", self.list_page, methods=["GET"])
        app.add_api_route("/admin/summaries/action", self.action_get, methods=["GET"])
        app.add_api_route("/admin/summaries/action", self.action_post, methods=["POST"])
        app.add_api_route("/admin/summaries/add", self.add, methods=["POST"])
        app.add_api_route("/admin/summaries/save", self.save, methods=["POST"])
        app.add_api_route("/admin/summaries/{summary_id}", self.detail, methods=["GET"])

    @staticmethod
    def summaries_url(
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        q: str = "",
        notice: str = "",
    ) -> str:
        params: dict[str, object] = {"status": status or "active", "limit": limit}
        if group_id is not None:
            params["group_id"] = group_id
        if q.strip():
            params["q"] = q.strip()
        if notice.strip():
            params["notice"] = notice.strip()
        return "/admin/summaries?" + urlencode(params)

    @staticmethod
    def summary_detail_url(summary_id: int, *, notice: str = "") -> str:
        params = {"notice": notice.strip()} if notice.strip() else {}
        return f"/admin/summaries/{int(summary_id)}" + ("?" + urlencode(params) if params else "")

    @staticmethod
    def split_cues(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,，、;；\n]+", text or "") if part.strip()][:12]

    async def list_page(
        self,
        request: Request,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        q: str = "",
        notice: str = "",
    ) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(
            render_memory_summaries_page(
                memory=self.services.get_memory(),
                groups=self.services.target_groups(),
                selected_group_id=group_id,
                status=status,
                limit=limit,
                q=q,
                notice=notice,
            )
        )

    async def action_get(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse("summary actions require POST", status_code=405)

    async def action_post(
        self,
        request: Request,
        summary_id: int,
        action: str,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        q: str = "",
    ):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        ok = self.services.get_memory().admin_set_memory_summary_state(summary_id, action=action)
        notice = "已更新回想状态" if ok else "没有找到回想或动作无效"
        return RedirectResponse(
            url=self.summaries_url(group_id=group_id, status=status, limit=limit, q=q, notice=notice),
            status_code=303,
        )

    async def add(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        group_id = admin_form_int(form, "group_id")
        if group_id is None:
            return RedirectResponse(url=self.summaries_url(notice="新增失败：需要填写群号"), status_code=303)
        summary_id = self.services.get_memory().admin_add_memory_summary(
            group_id=group_id,
            summary=form.get("summary", ""),
            recall_cues=self.split_cues(form.get("recall_cues", "")),
            locked=form.get("locked") == "1",
        )
        if not summary_id:
            return RedirectResponse(
                url=self.summaries_url(group_id=group_id, notice="新增失败：回想内容为空"),
                status_code=303,
            )
        return RedirectResponse(
            url=self.summary_detail_url(summary_id, notice="已新增人工回想"),
            status_code=303,
        )

    async def save(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        summary_id = admin_form_int(form, "summary_id") or 0
        ok = self.services.get_memory().admin_update_memory_summary(
            summary_id,
            summary=form.get("summary", ""),
            recall_cues=self.split_cues(form.get("recall_cues", "")),
            status=form.get("status", "active"),
            locked=form.get("locked") == "1",
        )
        notice = "已保存回想" if ok else "保存失败：回想不存在或内容为空"
        return RedirectResponse(url=self.summary_detail_url(summary_id, notice=notice), status_code=303)

    async def detail(self, request: Request, summary_id: int, notice: str = "") -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(
            render_memory_summary_detail_page(
                memory=self.services.get_memory(),
                summary_id=summary_id,
                groups=self.services.target_groups(),
                notice=notice,
            )
        )
