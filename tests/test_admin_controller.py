import asyncio
from urllib.parse import parse_qs, urlsplit

import httpx
import nonebot
from fastapi import FastAPI

nonebot.init()

from qq_social_agent import plugin
from qq_social_agent.admin_edit_controller import AdminEditableFileService
from qq_social_agent.memory import MemoryStore


EXPECTED_ADMIN_ROUTES = [
    ("/admin/daily-review/{mode}", "POST"),
    ("/admin/proactive-chat", "POST"),
    ("/admin/send-group", "POST"),
    ("/admin/send-private", "POST"),
    ("/admin", "GET"),
    ("/admin/tools", "GET"),
    ("/admin/tools/report", "GET"),
    ("/admin/tools/action", "POST"),
    ("/admin/edit", "GET"),
    ("/admin/edit/save", "POST"),
    ("/admin/summaries", "GET"),
    ("/admin/summaries/action", "GET"),
    ("/admin/summaries/action", "POST"),
    ("/admin/summaries/add", "POST"),
    ("/admin/summaries/save", "POST"),
    ("/admin/summaries/{summary_id}", "GET"),
    ("/admin/private-memory", "GET"),
    ("/admin/private-memory/save", "POST"),
    ("/admin/plugins", "GET"),
    ("/admin/messages/{message_id}", "GET"),
    ("/admin/memory", "GET"),
    ("/admin/memory/action", "GET"),
    ("/admin/memory/action", "POST"),
    ("/admin/memory/correct", "POST"),
    ("/admin/memory/merge", "POST"),
    ("/admin/memory/{atom_id}", "GET"),
]


def make_admin_app() -> FastAPI:
    app = FastAPI()
    plugin.admin_controller.register(app)
    return app


def test_admin_controller_registers_all_routes_in_compatible_order() -> None:
    app = make_admin_app()
    routes = [
        (route.path, method)
        for route in app.routes
        if route.path.startswith("/admin")
        for method in sorted(route.methods or ())
    ]

    assert routes == EXPECTED_ADMIN_ROUTES
    assert routes.index(("/admin/summaries/action", "GET")) < routes.index(("/admin/summaries/{summary_id}", "GET"))
    assert routes.index(("/admin/memory/action", "GET")) < routes.index(("/admin/memory/{atom_id}", "GET"))


def test_admin_http_auth_get_post_and_static_action_routes(monkeypatch, tmp_path) -> None:
    memory = MemoryStore(tmp_path / "admin.sqlite3")
    monkeypatch.setattr(plugin, "memory", memory)
    monkeypatch.setattr(plugin, "_http_ready_payload", lambda: {"ok": True})
    monkeypatch.setattr(plugin, "_http_health_payload", lambda: {"ok": True})
    monkeypatch.setattr(plugin, "_http_status_payload", lambda: {})
    monkeypatch.setattr(plugin, "_status_model_routes", lambda: {})
    summary_id = memory.admin_add_memory_summary(
        group_id=1026813421,
        summary="一条用于管理页面的人工回想",
        recall_cues=["管理测试"],
        locked=False,
    )
    app = make_admin_app()
    async def exercise_http_routes() -> None:
        # ASGITransport runs the handlers on this event-loop thread. MemoryStore's
        # SQLite connection is intentionally thread-bound, so TestClient's portal
        # thread would not represent production access here.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("8.8.8.8", 12345)),
            base_url="http://test",
            follow_redirects=False,
        ) as remote:
            denied = await remote.post(
                "/admin/send-group",
                content="not-json",
                headers={"content-type": "application/json"},
            )
            assert denied.status_code == 403
            assert denied.json() == {"ok": False, "reason": "local_admin_only"}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 12345)),
            base_url="http://test",
            follow_redirects=False,
        ) as local:
            dashboard = await local.get("/admin")
            assert dashboard.status_code == 200
            assert "张风雪 Admin" in dashboard.text

            summaries_action = await local.get("/admin/summaries/action")
            memory_action = await local.get("/admin/memory/action")
            assert summaries_action.status_code == 405
            assert summaries_action.text == "summary actions require POST"
            assert memory_action.status_code == 405
            assert memory_action.text == "memory actions require POST"

            edit_page = await local.get("/admin/edit", params={"file": "prompt"})
            assert edit_page.status_code == 200
            assert "人格 / Prompt" in edit_page.text

            prompt_path = plugin.ADMIN_EDITABLE_FILES[0]["path"]
            prompt_before = prompt_path.read_text(encoding="utf-8")
            invalid_save = await local.post(
                "/admin/edit/save",
                data={"file": "prompt", "content": "- invalid top-level"},
            )
            assert invalid_save.status_code == 400
            assert prompt_path.read_text(encoding="utf-8") == prompt_before

            archived = await local.post(
                "/admin/summaries/action",
                params={"summary_id": summary_id, "action": "archive", "group_id": 1026813421},
            )
            assert archived.status_code == 303
            assert archived.headers["location"].startswith("/admin/summaries?")
            redirect_query = parse_qs(urlsplit(archived.headers["location"]).query)
            assert redirect_query == {
                "status": ["active"],
                "limit": ["80"],
                "group_id": ["1026813421"],
                "notice": ["已更新回想状态"],
            }
            assert memory.memory_summary(summary_id).status == "archived"

            summary_page = await local.get(
                "/admin/summaries", params={"group_id": 1026813421, "status": "archived"}
            )
            assert summary_page.status_code == 200
            assert "一条用于管理页面的人工回想" in summary_page.text

    asyncio.run(exercise_http_routes())


def test_admin_edit_service_keeps_project_path_guard_and_backup(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    editable_path = project_root / "config.yaml"
    editable_path.write_text("setting: before\n", encoding="utf-8")
    backup_dir = project_root / "data" / "admin_backups"
    service = AdminEditableFileService(
        reload_prompt_runtime=lambda: None,
        editable_files=(
            {
                "key": "config",
                "label": "测试配置",
                "path": editable_path,
                "description": "测试用文件",
                "reload": "restart_required",
            },
        ),
        project_root=project_root,
        backup_dir=backup_dir,
    )

    notice = service.save_file("config", "setting: after\n")

    assert "已保存：测试配置" in notice
    assert editable_path.read_text(encoding="utf-8") == "setting: after\n"
    backups = list(backup_dir.glob("*_config.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "setting: before\n"

    outside_path = tmp_path / "outside.yaml"
    outside_path.write_text("setting: outside\n", encoding="utf-8")
    unsafe_service = AdminEditableFileService(
        reload_prompt_runtime=lambda: None,
        editable_files=({"key": "outside", "path": outside_path},),
        project_root=project_root,
        backup_dir=backup_dir,
    )
    try:
        unsafe_service.save_file("outside", "setting: changed\n")
    except ValueError as exc:
        assert str(exc) == "拒绝编辑项目目录外的文件"
    else:
        raise AssertionError("editing a path outside the project root must be rejected")
    assert outside_path.read_text(encoding="utf-8") == "setting: outside\n"
