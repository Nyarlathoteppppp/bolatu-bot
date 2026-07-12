import pytest

from qq_social_agent import observability


def _bot_state(bot_id: str) -> dict[str, object]:
    snapshot = observability.onebot_status_snapshot()
    return next(item for item in snapshot["bots"] if item["bot_id"] == bot_id)


def test_onebot_lifecycle_tracks_seen_api_success_timeout_and_disconnect(monkeypatch) -> None:
    bot_id = "observability-lifecycle-test"
    timestamps = iter((100.0, 110.0, 120.0, 130.0, 140.0))
    monkeypatch.setattr(observability.time, "time", lambda: next(timestamps))

    observability.mark_bot_connected(bot_id)
    connected = _bot_state(bot_id)
    assert connected["connected"] is True
    assert connected["last_connected_at"] == 100.0
    assert connected["last_seen_at"] == 100.0
    assert connected["connection_count"] == 1

    observability.mark_bot_seen(bot_id)
    assert _bot_state(bot_id)["last_seen_at"] == 110.0

    observability.mark_onebot_api_success(bot_id, "get_group_info", elapsed_ms=12.9)
    succeeded = _bot_state(bot_id)
    assert succeeded["last_seen_at"] == 120.0
    assert succeeded["last_api_success_at"] == 120.0
    assert succeeded["last_api_name"] == "get_group_info"
    assert succeeded["last_api_outcome"] == "success"
    assert succeeded["last_api_elapsed_ms"] == 12
    assert succeeded["api_call_count"] == 1
    assert succeeded["api_success_count"] == 1
    assert succeeded["consecutive_api_errors"] == 0

    error = TimeoutError("OneBot call exceeded 8 seconds")
    observability.mark_onebot_api_error(
        bot_id,
        "get_group_msg_history",
        elapsed_ms=8_001,
        error=error,
        timeout=True,
    )
    timed_out = _bot_state(bot_id)
    assert timed_out["last_seen_at"] == 120.0
    assert timed_out["last_api_error_at"] == 130.0
    assert timed_out["last_api_outcome"] == "timeout"
    assert timed_out["last_api_error_type"] == "TimeoutError"
    assert timed_out["last_api_error"] == "OneBot call exceeded 8 seconds"
    assert timed_out["api_call_count"] == 2
    assert timed_out["api_error_count"] == 1
    assert timed_out["api_timeout_count"] == 1
    assert timed_out["consecutive_api_errors"] == 1

    observability.mark_bot_disconnected(bot_id)
    disconnected = _bot_state(bot_id)
    assert disconnected["connected"] is False
    assert disconnected["last_disconnected_at"] == 140.0
    assert disconnected["disconnect_count"] == 1
    assert bot_id not in observability.onebot_status_snapshot()["connected_bots"]


def test_correlation_scope_restores_nested_and_exception_state() -> None:
    original_token = observability.set_correlation_id("before")
    try:
        with observability.correlation_scope("outer") as correlation_id:
            assert correlation_id == "outer"
            assert observability.current_correlation_id() == "outer"
            with observability.correlation_scope("inner"):
                assert observability.current_correlation_id() == "inner"
            assert observability.current_correlation_id() == "outer"

        assert observability.current_correlation_id() == "before"

        with pytest.raises(RuntimeError, match="boom"):
            with observability.correlation_scope("failing"):
                raise RuntimeError("boom")
        assert observability.current_correlation_id() == "before"
    finally:
        observability.reset_correlation_id(original_token)
