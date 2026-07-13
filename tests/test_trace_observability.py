import json
from types import SimpleNamespace

from qq_social_agent.observability import (
    TRACE_STAGE_ORDER,
    build_trace_snapshot,
    render_trace_html,
    sanitize_trace_metadata,
    trace_json_snapshot,
)


def _event(
    event_type: str,
    stage: str,
    action: str,
    created_at: float,
    *,
    correlation_id: str = "group:100:9001",
    **metadata: object,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "group_id": 100,
        "user_id": 200,
        "stage": stage,
        "action": action,
        "created_at": created_at,
        "metadata": {"correlation_id": correlation_id, **metadata},
    }


def test_trace_snapshot_uses_fixed_stage_order_and_calculates_durations() -> None:
    events = [
        _event("message_sent", "send", "reply", 106.0, elapsed_ms=25),
        _event("candidate_generated", "reply_candidates", "reply", 102.0, elapsed_ms=500),
        _event("message_received", "group", "received", 100.0, source_message_id="9001"),
        _event("image_ocr", "media_context", "recognized", 100.1, elapsed_ms=100),
        {
            "event_type": "onebot_api",
            "group_id": 100,
            "user_id": 200,
            "stage": "onebot",
            "action": "success",
            "created_at": 100.2,
            "metadata": {"message_id": "9001", "api": "get_msg", "elapsed_ms": 30},
        },
        _event("message_buffered", "locked", "recorded", 100.3),
        _event("rag_retrieval", "decision_context", "injected", 100.35, elapsed_ms=40, hit_document_ids=[7]),
        _event("decision_start", "group", "start", 100.4),
        _event("decision_result", "llm", "reply", 101.4, elapsed_ms=1_000, should_reply=True),
        _event("tool_call", "fresh_context", "news", 101.5, latency_ms=200, status="ok"),
        _event("approval_requested", "approval", "pending", 103.0),
        _event("approval_accepted", "approval", "reply", 105.0, approval_wait_ms=2_000),
    ]

    snapshot = build_trace_snapshot(events, generated_at=200.0)

    assert snapshot["schema_version"] == 2
    assert snapshot["stage_order"] == list(TRACE_STAGE_ORDER)
    assert snapshot["trace_count"] == 1
    trace = snapshot["traces"][0]
    assert trace["trace_id"] == "group:100:9001"
    assert trace["message_id"] == "9001"
    assert trace["status"] == "complete"
    assert trace["total_duration_ms"] == 6_000
    assert trace["event_count"] == len(events)
    assert trace["error_count"] == 0

    phases = {phase["stage"]: phase for phase in trace["phases"]}
    assert list(phases) == list(TRACE_STAGE_ORDER)
    assert all(phase["status"] == "ok" for phase in phases.values())
    assert phases["ocr"]["duration_ms"] == 100
    assert phases["history"]["duration_ms"] == 30
    assert phases["rag"]["duration_ms"] == 40
    assert phases["decision"]["duration_ms"] == 1_000
    assert phases["search/tool"]["duration_ms"] == 200
    assert phases["generation"]["duration_ms"] == 500
    assert phases["approval"]["duration_ms"] == 2_000
    assert phases["send"]["duration_ms"] == 25
    json.dumps(snapshot, ensure_ascii=False)


def test_trace_falls_back_to_message_id_and_redacts_nested_metadata() -> None:
    metadata = {
        "source_message_id": "42",
        "text": "用户 123456789 发了 sk-abcdefghijklmnop",
        "api_key": "should-never-appear",
        "headers": {"Authorization": "Bearer top-secret-value"},
        "source_url": "https://example.com/a?token=secret&safe=1#fragment",
        "prompt_tokens": 123,
    }
    events = [
        {
            "event_type": "message_received",
            "group_id": 7,
            "user_id": 8,
            "stage": "group",
            "action": "received",
            "created_at": 10,
            "metadata_json": json.dumps(metadata),
        },
        SimpleNamespace(
            event_type="tool_call",
            group_id=7,
            user_id=8,
            stage="fresh_context",
            action="news",
            created_at=11,
            metadata={
                "message_id": "42",
                "status": "error",
                "error": "provider rejected sk-zzzzzzzzzzzzzzzz",
            },
        ),
    ]

    snapshot = trace_json_snapshot(events, generated_at=20)
    trace = snapshot["traces"][0]
    assert trace["trace_id"] == "message:7:42"
    assert trace["status"] == "error"
    assert trace["error_count"] == 1
    assert "sk-" not in json.dumps(trace, ensure_ascii=False).casefold()

    receive = trace["phases"][0]["events"][0]["metadata"]
    assert receive["api_key"] == "[REDACTED]"
    assert receive["headers"]["Authorization"] == "[REDACTED]"
    assert receive["prompt_tokens"] == 123
    assert "123456789" not in receive["text"]
    assert "fragment" not in receive["source_url"]
    assert "secret" not in receive["source_url"]


def test_trace_snapshot_bounds_traces_and_events_per_stage() -> None:
    events = [
        _event(
            "message_received",
            "group",
            "received",
            300 + index,
            correlation_id="trace-newest",
            sequence=index,
        )
        for index in range(25)
    ]
    events.extend(
        [
            _event("message_received", "group", "received", 200, correlation_id="trace-middle"),
            _event("message_received", "group", "received", 100, correlation_id="trace-oldest"),
        ]
    )

    snapshot = build_trace_snapshot(events, limit=2, events_per_stage=3, generated_at=500)

    assert snapshot["available_trace_count"] == 3
    assert snapshot["trace_count"] == 2
    assert snapshot["omitted_trace_count"] == 1
    assert snapshot["truncated"] is True
    assert [trace["trace_id"] for trace in snapshot["traces"]] == ["trace-newest", "trace-middle"]
    newest = snapshot["traces"][0]
    assert newest["event_count"] == 25
    assert newest["displayed_event_count"] == 3
    assert newest["omitted_event_count"] == 22
    assert len(newest["phases"][0]["events"]) == 3


def test_trace_html_escapes_all_dynamic_content_and_has_no_script() -> None:
    events = [
        _event(
            "message_received",
            "group",
            "received",
            1,
            text='<img src=x onerror="alert(1)">',
        ),
        _event(
            "decision_result",
            "llm",
            "</td><script>alert(2)</script>",
            2,
            should_reply=False,
            reason="<b>do not trust this</b>",
        ),
    ]
    snapshot = build_trace_snapshot(events, generated_at=3)

    rendered = render_trace_html(snapshot, title="<script>alert(0)</script>")

    assert "<script>alert" not in rendered
    assert "<img src=x" not in rendered
    assert "<b>do not trust" not in rendered
    assert "&lt;script&gt;alert(0)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=" in rendered
    assert "alert(1)" in rendered
    assert "&lt;b&gt;do not trust this&lt;/b&gt;" in rendered
    assert "Content-Security-Policy" in rendered


def test_sanitize_trace_metadata_is_bounded_and_preserves_token_counts() -> None:
    sanitized = sanitize_trace_metadata(
        {
            "password": "nope",
            "completion_tokens": 88,
            "items": list(range(30)),
            "deep": {"a": {"b": {"c": {"d": "too deep"}}}},
        }
    )

    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["completion_tokens"] == 88
    assert len(sanitized["items"]) == 13
    assert sanitized["items"][-1] == "[18 more items omitted]"
    assert "omitted" in json.dumps(sanitized["deep"])
