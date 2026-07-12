from types import SimpleNamespace

import nonebot

nonebot.init()

import qq_social_agent.plugin as plugin
from qq_social_agent.memory import MemoryStore


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


def test_trace_payload_supports_correlation_and_message_lookup(monkeypatch, tmp_path) -> None:
    store = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr(plugin, "memory", store)
    store.add_metric_event(
        event_type="message_received",
        group_id=100,
        user_id=200,
        stage="group",
        action="received",
        metadata={"correlation_id": "group:100:42", "source_message_id": "42"},
        created_at=1000.0,
    )
    store.add_metric_event(
        event_type="decision_result",
        group_id=100,
        user_id=200,
        stage="llm",
        action="reply",
        metadata={"correlation_id": "group:100:42", "should_reply": True, "elapsed_ms": 50},
        created_at=1001.0,
    )

    by_correlation = plugin._http_trace_payload(trace_id="group:100:42")
    by_message = plugin._http_trace_payload(trace_id="42")

    assert by_correlation["trace_count"] == 1
    assert by_message["trace_count"] == 1
    assert by_message["traces"][0]["trace_id"] == "group:100:42"
