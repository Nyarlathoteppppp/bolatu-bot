"""HTML trace presentation with escaped dynamic content."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Mapping

from .trace_snapshot import (
    MAX_TRACE_ERRORS,
    MAX_TRACE_EVENTS_PER_STAGE,
    MAX_TRACE_LIMIT,
    TRACE_STAGE_ORDER,
    _coerce_float,
    sanitize_trace_metadata,
)


def render_trace_html(snapshot: Mapping[str, object], *, title: str = "QQ Social Agent Traces") -> str:
    """Render a compact, script-free HTML trace view.

    Every dynamic value is HTML-escaped, including data from an externally
    supplied snapshot. Rendering applies its own bounds in addition to the
    snapshot builder's limits.
    """

    safe_title = _html_text(title, limit=120)
    raw_traces = snapshot.get("traces", []) if isinstance(snapshot, Mapping) else []
    traces = list(raw_traces)[:MAX_TRACE_LIMIT] if isinstance(raw_traces, (list, tuple)) else []
    generated_at = _format_trace_time(snapshot.get("generated_at"))
    summary = (
        f"traces={_html_text(snapshot.get('trace_count', len(traces)), limit=20)} "
        f"source_events={_html_text(snapshot.get('source_event_count', ''), limit=20)} "
        f"generated={_html_text(generated_at, limit=40)}"
    )
    articles = "".join(_render_trace_article(trace) for trace in traces if isinstance(trace, Mapping))
    if not articles:
        articles = '<p class="empty">No traceable events.</p>'
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" '
        'content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        f"<title>{safe_title}</title>"
        "<style>"
        ":root{color-scheme:light dark;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}"
        "body{max-width:1180px;margin:0 auto;padding:18px;line-height:1.45}"
        "h1{font:600 1.35rem system-ui;margin:0 0 4px}.summary,.meta{opacity:.72;font-size:.86rem}"
        "article{border:1px solid #8886;border-radius:10px;padding:12px;margin:14px 0}"
        ".trace-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}"
        ".trace-id{font-weight:700;overflow-wrap:anywhere}.badge{border-radius:999px;padding:2px 8px;font-size:.78rem}"
        ".ok{background:#2e7d3230}.warn{background:#ef6c0030}.bad{background:#c6282830}.idle{background:#7773}"
        "ol{list-style:none;margin:10px 0 0;padding:0;display:grid;gap:6px}"
        "li{border-left:4px solid #8886;padding:5px 8px;background:#8881}"
        "li.ok{border-color:#2e7d32}li.warn{border-color:#ef6c00}li.bad{border-color:#c62828}"
        ".stage-line{display:flex;gap:9px;flex-wrap:wrap}.stage{font-weight:700;min-width:92px}"
        "details{margin-top:4px}table{border-collapse:collapse;width:100%;font-size:.8rem}"
        "th,td{text-align:left;vertical-align:top;border-top:1px solid #8884;padding:4px;overflow-wrap:anywhere}"
        "code{white-space:pre-wrap;word-break:break-word}.errors{color:#c62828}.empty{opacity:.7}"
        "</style></head><body>"
        f"<h1>{safe_title}</h1><div class=\"summary\">{summary}</div>{articles}</body></html>"
    )


def _render_trace_article(trace: Mapping[str, object]) -> str:
    status = str(trace.get("status") or "in_progress")
    css_class = _trace_status_css(status)
    trace_id = _html_text(trace.get("trace_id", "unknown"), limit=180)
    total_ms = _html_text(trace.get("total_duration_ms", 0), limit=24)
    event_count = _html_text(trace.get("event_count", 0), limit=16)
    group_id = _html_text(trace.get("group_id", ""), limit=40)
    user_id = _html_text(trace.get("user_id", ""), limit=40)
    errors = trace.get("errors", [])
    error_items = list(errors)[:MAX_TRACE_ERRORS] if isinstance(errors, (list, tuple)) else []
    error_html = ""
    if error_items:
        rows = []
        for item in error_items:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                "<li>"
                f"{_html_text(item.get('stage', ''), limit=40)}: "
                f"{_html_text(item.get('message', ''), limit=240)}"
                "</li>"
            )
        if rows:
            error_html = f'<ul class="errors">{"".join(rows)}</ul>'

    raw_phases = trace.get("phases", [])
    phase_map = {
        str(item.get("stage")): item
        for item in list(raw_phases)[: len(TRACE_STAGE_ORDER)]
        if isinstance(item, Mapping)
    } if isinstance(raw_phases, (list, tuple)) else {}
    phases_html = "".join(_render_trace_phase(stage, phase_map.get(stage, {})) for stage in TRACE_STAGE_ORDER)
    return (
        "<article>"
        '<div class="trace-head">'
        f'<span class="trace-id">{trace_id}</span>'
        f'<span class="badge {css_class}">{_html_text(status, limit=40)}</span>'
        f'<span class="meta">total={total_ms}ms events={event_count} group={group_id} user={user_id}</span>'
        f"</div>{error_html}<ol>{phases_html}</ol></article>"
    )


def _render_trace_phase(stage: str, phase: Mapping[str, object]) -> str:
    status = str(phase.get("status") or "not_reached")
    css_class = _trace_status_css(status)
    duration = phase.get("duration_ms")
    duration_text = "-" if duration is None else f"{_html_text(duration, limit=24)}ms"
    event_count = _html_text(phase.get("event_count", 0), limit=16)
    raw_events = phase.get("events", [])
    events = list(raw_events)[-MAX_TRACE_EVENTS_PER_STAGE:] if isinstance(raw_events, (list, tuple)) else []
    details = ""
    if events:
        rows: list[str] = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            metadata = sanitize_trace_metadata(event.get("metadata", {}))
            metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            rows.append(
                "<tr>"
                f"<td>{_html_text(_format_trace_time(event.get('created_at')), limit=40)}</td>"
                f"<td>{_html_text(event.get('event_type', ''), limit=80)}</td>"
                f"<td>{_html_text(event.get('action', ''), limit=80)}</td>"
                f"<td><code>{_html_text(metadata_json, limit=2000)}</code></td>"
                "</tr>"
            )
        if rows:
            details = (
                f"<details><summary>{len(rows)} event(s)</summary><table>"
                "<thead><tr><th>time</th><th>event</th><th>action</th><th>metadata</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></details>"
            )
    return (
        f'<li class="{css_class}"><div class="stage-line">'
        f'<span class="stage">{_html_text(stage, limit=30)}</span>'
        f'<span>{_html_text(status, limit=30)}</span><span>{duration_text}</span>'
        f'<span class="meta">events={event_count}</span></div>{details}</li>'
    )


def _trace_status_css(status: str) -> str:
    if status in {"complete", "ok"}:
        return "ok"
    if status in {"error", "timeout"}:
        return "bad"
    if status in {"pending", "pending_approval", "skipped"}:
        return "warn"
    return "idle"


def _format_trace_time(value: object) -> str:
    timestamp = _coerce_float(value, 0.0)
    if timestamp <= 0:
        return "-"
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return "-"


def _html_text(value: object, *, limit: int) -> str:
    text = str(value if value is not None else "")
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return html.escape(text, quote=True)
