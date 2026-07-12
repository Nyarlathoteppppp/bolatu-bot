from types import SimpleNamespace

import nonebot

nonebot.init()

import qq_social_agent.plugin as plugin


def test_health_payload_only_depends_on_process_and_database(monkeypatch) -> None:
    monkeypatch.setattr(plugin, "_status_db_health", lambda: (True, ""))

    payload = plugin._http_health_payload()

    assert payload["ok"] is True
    assert payload["database"] == {"ok": True, "error": ""}
    assert payload["process"]["uptime_seconds"] >= 0


def test_ready_payload_reports_each_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr(plugin, "_status_db_health", lambda: (False, "db down"))
    monkeypatch.setattr(plugin, "onebot_status_snapshot", lambda: {"connected_bots": [], "bots": []})
    monkeypatch.setattr(plugin, "deepseek_client", None)

    payload = plugin._http_ready_payload()

    assert payload["ok"] is False
    assert payload["reasons"] == [
        "database_unavailable",
        "llm_client_unavailable",
        "onebot_disconnected",
    ]


def test_ready_payload_accepts_initialized_provider_and_onebot(monkeypatch) -> None:
    monkeypatch.setattr(plugin, "_status_db_health", lambda: (True, ""))
    monkeypatch.setattr(
        plugin,
        "onebot_status_snapshot",
        lambda: {"connected_bots": ["123"], "bots": [{"bot_id": "123", "connected": True}]},
    )
    monkeypatch.setattr(plugin, "deepseek_client", SimpleNamespace(clients={"provider": object()}))

    payload = plugin._http_ready_payload()

    assert payload["ok"] is True
    assert payload["reasons"] == []
    assert payload["connected_bot_count"] == 1
