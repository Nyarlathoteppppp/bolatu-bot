"""HTTP controller and file service for editable admin configuration."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import yaml
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from .admin_http import admin_form_data
from .config import PROJECT_ROOT, load_config
from .admin_ui import render_admin_edit_page
from .prompts import PromptRegistry


ADMIN_EDITABLE_FILES: tuple[dict[str, object], ...] = (
    {
        "key": "prompt",
        "label": "人格 / Prompt",
        "path": PROJECT_ROOT / "prompts" / "zhangfengxue.yaml",
        "description": "集中人格、action_guides 和所有 LLM flow。保存后会立即热重载到当前 bot。",
        "reload": "prompt",
    },
    {
        "key": "config",
        "label": "后端配置 config.yaml",
        "path": PROJECT_ROOT / "config.yaml",
        "description": "工作强度、模型、白名单、搜索、频率等配置。保存会校验 YAML；多数配置需要重启后端才完整生效。",
        "reload": "restart_required",
    },
)
ADMIN_BACKUP_DIR = PROJECT_ROOT / "data" / "admin_backups"


class AdminEditableFileService:
    def __init__(
        self,
        *,
        reload_prompt_runtime: Callable[[], None],
        editable_files: tuple[dict[str, object], ...] = ADMIN_EDITABLE_FILES,
        project_root: Path = PROJECT_ROOT,
        backup_dir: Path = ADMIN_BACKUP_DIR,
    ) -> None:
        self.reload_prompt_runtime = reload_prompt_runtime
        self.editable_files = editable_files
        self.project_root = project_root
        self.backup_dir = backup_dir

    def files_summary(self) -> list[dict[str, str]]:
        return [
            {
                "key": str(item.get("key", "")),
                "label": str(item.get("label", item.get("key", ""))),
                "description": str(item.get("description", "")),
            }
            for item in self.editable_files
        ]

    def editable_file(self, key: str) -> dict[str, object]:
        clean_key = (key or "").strip()
        for item in self.editable_files:
            if str(item.get("key")) == clean_key:
                return item
        return self.editable_files[0]

    @staticmethod
    def edit_url(key: str, *, notice: str = "") -> str:
        params: dict[str, object] = {"file": key}
        if notice.strip():
            params["notice"] = notice.strip()
        return "/admin/edit?" + urlencode(params)

    def read_file(self, key: str) -> tuple[str, str]:
        item = self.editable_file(key)
        path = Path(item["path"])
        try:
            return path.read_text(encoding="utf-8"), ""
        except OSError as exc:
            return "", f"读取失败：{exc}"

    def save_file(self, key: str, content: str) -> str:
        item = self.editable_file(key)
        path = Path(item["path"])
        self._ensure_path_allowed(path)
        if not path.exists():
            raise ValueError(f"文件不存在：{path}")
        self._validate_content(item, content)
        backup_path = self._backup_file(path)
        tmp_path = path.with_name(f".{path.name}.admin_tmp")
        tmp_path.write_text(content, encoding="utf-8")
        try:
            tmp_path.replace(path)
        except OSError as exc:
            # Docker single-file bind mounts cannot be atomically replaced. Keep the
            # backup and fall back to an in-place overwrite so WebUI config edits work.
            if getattr(exc, "errno", None) != 16:
                raise
            path.write_text(content, encoding="utf-8")
            try:
                tmp_path.unlink()
            except OSError:
                pass
        reload_mode = str(item.get("reload", ""))
        if reload_mode == "prompt":
            self.reload_prompt_runtime()
            return f"已保存并热重载：{item.get('label')}；备份 {backup_path.name}"
        return f"已保存：{item.get('label')}；备份 {backup_path.name}。这个配置多数需要重启后端生效。"

    def _ensure_path_allowed(self, path: Path) -> None:
        root = self.project_root.resolve()
        resolved = path.resolve()
        try:
            allowed = resolved.is_relative_to(root)
        except AttributeError:
            allowed = str(resolved).startswith(str(root) + "/") or resolved == root
        if not allowed:
            raise ValueError("拒绝编辑项目目录外的文件")

    def _backup_file(self, path: Path) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int((now % 1) * 1000):03d}"
        try:
            rel = path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            rel = path.name
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", rel)
        backup_path = self.backup_dir / f"{timestamp}_{safe_name}"
        suffix = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{timestamp}_{suffix}_{safe_name}"
            suffix += 1
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    def _validate_content(self, item: dict[str, object], content: str) -> None:
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML 解析失败：{exc}") from exc
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("YAML 顶层必须是对象")
        key = str(item.get("key", ""))
        if key == "prompt":
            self._validate_prompt_content(content, raw or {})
        elif key == "config":
            self._validate_config_content(content)

    def _validate_prompt_content(self, content: str, raw: dict[str, object]) -> None:
        persona_raw = raw.get("persona")
        if not isinstance(persona_raw, dict) or not str(persona_raw.get("id", "")).strip():
            raise ValueError("Prompt 文件需要 persona.id")
        if not str(persona_raw.get("prompt", "")).strip():
            raise ValueError("Prompt 文件需要 persona.prompt")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.backup_dir / f".validate_prompt_{int(time.time() * 1000)}.yaml"
        tmp_path.write_text(content, encoding="utf-8")
        try:
            registry = PromptRegistry(tmp_path)
            required_flows = (
                "timing_gate",
                "decision",
                "reply",
                "reply_candidates",
                "reply_direct",
                "mid_memory",
                "style_learning",
                "member_profile",
                "daily_review",
            )
            for flow in required_flows:
                section = registry.flows.get(flow)
                if not isinstance(section, dict):
                    raise ValueError(f"Prompt 文件缺少 flows.{flow}")
                if not str(section.get("system", "")).strip() or not str(section.get("user", "")).strip():
                    raise ValueError(f"flows.{flow} 需要 system 和 user")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _validate_config_content(self, content: str) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.backup_dir / f".validate_config_{int(time.time() * 1000)}.yaml"
        tmp_path.write_text(content, encoding="utf-8")
        try:
            load_config(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


@dataclass(frozen=True)
class AdminEditServices:
    is_local_admin_request: Callable[[Request], bool]
    file_service: AdminEditableFileService


class AdminEditController:
    def __init__(self, services: AdminEditServices) -> None:
        self.services = services

    def register(self, app: FastAPI) -> None:
        app.add_api_route("/admin/edit", self.page, methods=["GET"])
        app.add_api_route("/admin/edit/save", self.save, methods=["POST"])

    async def page(self, request: Request, file: str = "prompt", notice: str = "") -> HTMLResponse:
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        item = self.services.file_service.editable_file(file)
        content, read_notice = self.services.file_service.read_file(str(item["key"]))
        return HTMLResponse(
            render_admin_edit_page(
                editable_files=self.services.file_service.files_summary(),
                selected_key=str(item["key"]),
                content=content,
                notice=notice or read_notice,
            )
        )

    async def save(self, request: Request):
        if not self.services.is_local_admin_request(request):
            return HTMLResponse("local admin only", status_code=403)
        form = await admin_form_data(request)
        key = form.get("file", "prompt").strip() or "prompt"
        content = form.get("content", "")
        item = self.services.file_service.editable_file(key)
        try:
            notice = self.services.file_service.save_file(str(item["key"]), content)
        except Exception as exc:
            return HTMLResponse(
                render_admin_edit_page(
                    editable_files=self.services.file_service.files_summary(),
                    selected_key=str(item["key"]),
                    content=content,
                    notice=f"保存失败：{exc}",
                ),
                status_code=400,
            )
        return RedirectResponse(
            url=self.services.file_service.edit_url(str(item["key"]), notice=notice),
            status_code=303,
        )
