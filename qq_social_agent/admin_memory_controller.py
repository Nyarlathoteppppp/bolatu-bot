"""HTTP controllers for memory audit and private conversation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from .admin_http import admin_form_data, admin_form_int
from .admin_ui import render_memory_atom_detail_page, render_memory_audit_page, render_private_memory_page
from .memory import MemoryStore


@dataclass(frozen=True)
class AdminMemoryServices:
    is_local_admin_request: Callable[[Request], bool]
    get_memory: Callable[[], MemoryStore]
    target_groups: Callable[[], tuple[int, ...]]


class AdminMemoryController:
    def __init__(self, services: AdminMemoryServices) -> None:
        self.services = services

    def register_private_memory(self, app: FastAPI) -> None:
        app.add_api_route("/admin/private-memory", self.private_memory_page, methods=["GET"])
        app.add_api_route("/admin/private-memory/save", self.save_private_memory, methods=["POST"])

    def register_memory_audit(self, app: FastAPI) -> None:
        app.add_api_route("/admin/memory", self.memory_page, methods=["GET"])
        app.add_api_route("/admin/memory/action", self.action_get, methods=["GET"])
        app.add_api_route("/admin/memory/action", self.action_post, methods=["POST"])
        app.add_api_route("/admin/memory/correct", self.correct, methods=["POST"])
        app.add_api_route("/admin/memory/merge", self.merge, methods=["POST"])
        app.add_api_route("/admin/memory/{atom_id}", self.detail, methods=["GET"])

    @staticmethod
    def memory_url(
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        user_id: int | None = None,
        atom_type: str = "",
        q: str = "",
        notice: str = "",
    ) -> str:
        params: dict[str, object] = {"status": status or "active", "limit": limit}
        if group_id is not None:
            params["group_id"] = group_id
        if user_id is not None:
            params["user_id"] = user_id
        if atom_type.strip():
            params["atom_type"] = atom_type.strip()
        if q.strip():
            params["q"] = q.strip()
        if notice.strip():
            params["notice"] = notice.strip()
        return "/admin/memory?" + urlencode(params)

    @staticmethod
    def memory_detail_url(atom_id: int, *, notice: str = "") -> str:
        params = {"notice": notice.strip()} if notice.strip() else {}
        return f"/admin/memory/{int(atom_id)}" + ("?" + urlencode(params) if params else "")

    async def private_memory_page(self, request: Request, notice: str = "") -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(render_private_memory_page(memory=self.services.get_memory(), notice=notice))

    async def save_private_memory(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        chat_id = admin_form_int(form, "chat_id")
        user_id = admin_form_int(form, "user_id")
        if chat_id is None or user_id is None:
            return RedirectResponse(url="/admin/private-memory?notice=保存失败：缺少会话编号", status_code=303)
        self.services.get_memory().update_private_conversation_state(
            chat_id=chat_id,
            user_id=user_id,
            display_name=form.get("display_name", ""),
            relationship_note=form.get("relationship_note", ""),
            interaction_tone=form.get("interaction_tone", ""),
            current_topic=form.get("current_topic", ""),
            open_threads=[item.strip() for item in form.get("open_threads", "").splitlines() if item.strip()],
            frozen_fields=[item.strip() for item in form.get("frozen_fields", "").split(",") if item.strip()],
        )
        return RedirectResponse(url="/admin/private-memory?notice=私聊状态已保存", status_code=303)

    async def memory_page(
        self,
        request: Request,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        notice: str = "",
        user_id: int | None = None,
        atom_type: str = "",
        q: str = "",
    ) -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(
            render_memory_audit_page(
                memory=self.services.get_memory(),
                groups=self.services.target_groups(),
                selected_group_id=group_id,
                status=status,
                limit=limit,
                notice=notice,
                user_id=user_id,
                atom_type=atom_type,
                q=q,
            )
        )

    async def action_get(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse("memory actions require POST", status_code=405)

    async def action_post(
        self,
        request: Request,
        atom_id: int,
        action: str,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        user_id: int | None = None,
        atom_type: str = "",
        q: str = "",
    ):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        ok = self.services.get_memory().admin_review_memory_atom(atom_id, action=action, actor_user_id=0)
        notice = "已更新" if ok else "没有找到记忆或动作无效"
        return RedirectResponse(
            url=self.memory_url(
                group_id=group_id,
                status=status,
                limit=limit,
                user_id=user_id,
                atom_type=atom_type,
                q=q,
                notice=notice,
            ),
            status_code=303,
        )

    async def correct(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        atom_id = admin_form_int(form, "atom_id") or 0
        content = form.get("content", "").strip()
        atom_type = form.get("atom_type", "").strip() or None
        subject_user_id = admin_form_int(form, "subject_user_id")
        object_user_id = admin_form_int(form, "object_user_id")
        reason = form.get("reason", "").strip() or "WebUI 手动纠正"
        new_atom_id = self.services.get_memory().correct_memory_atom(
            atom_id,
            content=content,
            source="admin_ui",
            actor_user_id=0,
            reason=reason,
            confidence=1.0,
            atom_type=atom_type,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
        )
        target_id = new_atom_id or atom_id
        notice = f"已创建纠正记忆 #{new_atom_id}" if new_atom_id else "纠正失败：没有找到有效记忆或内容为空"
        return RedirectResponse(url=self.memory_detail_url(target_id, notice=notice), status_code=303)

    async def merge(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        source_atom_id = admin_form_int(form, "source_atom_id") or 0
        target_atom_id = admin_form_int(form, "target_atom_id") or 0
        reason = form.get("reason", "").strip() or "WebUI 合并重复记忆"
        ok = self.services.get_memory().admin_merge_memory_atoms(
            source_atom_id,
            target_atom_id,
            actor_user_id=0,
            note=reason,
        )
        notice = f"已将 #{source_atom_id} 合并到 #{target_atom_id}" if ok else "合并失败：检查 ID、群号、状态是否有效"
        return RedirectResponse(
            url=self.memory_detail_url(target_atom_id or source_atom_id, notice=notice),
            status_code=303,
        )

    async def detail(self, request: Request, atom_id: int, notice: str = "") -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        return HTMLResponse(
            render_memory_atom_detail_page(
                memory=self.services.get_memory(),
                atom_id=atom_id,
                groups=self.services.target_groups(),
                notice=notice,
            )
        )
