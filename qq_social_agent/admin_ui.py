from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from .memory import MemoryAtom, MemoryStore


def render_admin_dashboard(
    *,
    memory: MemoryStore,
    groups: tuple[int, ...],
    selected_group_id: int | None,
    ready: dict[str, object],
    health: dict[str, object],
    status: dict[str, object],
    model_routes: dict[str, str],
    pending_approvals: list[object],
    plugins: list[dict[str, Any]] | None = None,
) -> str:
    group_id = selected_group_id or (groups[0] if groups else None)
    recent_messages = memory.admin_recent_messages(group_id=group_id, limit=40)
    relation_events = memory.admin_recent_metric_events(event_types=("message_relation",), group_id=group_id, limit=25)
    decision_events = memory.admin_recent_metric_events(event_types=("decision_start", "llm_decision", "timing_gate"), group_id=group_id, limit=30)
    rag_events = memory.admin_recent_metric_events(event_types=("rag_retrieval", "tool_call", "tool_route_plan"), group_id=group_id, limit=25)
    atoms = memory.admin_recent_memory_atoms(group_id=group_id, status="active", limit=16)
    styles = memory.recent_style_rules(group_id, 12) if group_id is not None else []
    usage = memory.llm_usage_summary(start_at=time.time() - 24 * 3600, end_at=None)
    profiles = memory.member_impressions_for_context(group_id, [], limit=16) if group_id is not None else []

    body = f"""
    {_header('张风雪 Admin')}
    <main>
      {_group_nav(groups, group_id)}
      <section class="grid cards">
        {_status_card('Ready', ready)}
        {_status_card('Health', health)}
        {_model_card(model_routes)}
        {_approval_card(pending_approvals)}
        {_plugin_card(plugins or [])}
      </section>
      <section class="grid two">
        {_panel('最近关系判断', _metric_table(relation_events, show_meta=True))}
        {_panel('最近决策', _metric_table(decision_events, show_meta=True))}
      </section>
      <section class="grid two">
        {_panel('RAG / 工具', _metric_table(rag_events, show_meta=True))}
        {_panel('24h Token / 模型调用', _usage_table(usage))}
      </section>
      <section class="grid two">
        {_panel('近期消息链', _message_table(recent_messages))}
        {_panel('群友画像速览', _profile_list(profiles))}
      </section>
      <section class="grid two">
        {_panel('记忆审计入口', _memory_atom_list(atoms))}
        {_panel('近期风格规则', _style_rule_list(styles))}
      </section>
    </main>
    {_footer()}
    """
    return _page('张风雪 Admin', body)


def render_memory_audit_page(
    *,
    memory: MemoryStore,
    groups: tuple[int, ...],
    selected_group_id: int | None,
    status: str,
    limit: int,
    notice: str = "",
) -> str:
    group_id = selected_group_id or (groups[0] if groups else None)
    atoms = memory.admin_recent_memory_atoms(group_id=group_id, status=status, limit=limit)
    body = f"""
    {_header('记忆审计')}
    <main>
      {_group_nav(groups, group_id, path='/admin/memory')}
      {_notice(notice)}
      <nav class="tabs">
        {_tab('/admin/memory', 'active', group_id, status == 'active')}
        {_tab('/admin/memory', 'disputed', group_id, status == 'disputed')}
        {_tab('/admin/memory', 'expired', group_id, status == 'expired')}
        {_tab('/admin/memory', 'all', group_id, status == 'all')}
      </nav>
      <section class="panel wide"><h2>记忆单元</h2>{_memory_atom_table(atoms, group_id=group_id, status=status)}</section>
    </main>
    {_footer()}
    """
    return _page('张风雪记忆审计', body)



def render_plugins_page(
    *,
    plugins: list[dict[str, Any]],
    errors: list[dict[str, str]] | None = None,
) -> str:
    body = f"""
    {_header('本地插件')}
    <main>
      <section class="panel wide"><h2>插件清单</h2>{_plugin_table(plugins, full=True)}</section>
      <section class="panel wide"><h2>加载错误</h2>{_plugin_error_table(errors or [])}</section>
    </main>
    {_footer()}
    """
    return _page('张风雪本地插件', body)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_e(title)}</title><style>
:root {{ color-scheme: light dark; --bg:#f6f7f9; --fg:#20242a; --muted:#667085; --line:#d8dde6; --panel:#ffffff; --accent:#2f6fed; --bad:#b42318; --ok:#027a48; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#101318; --fg:#eef1f6; --muted:#9aa4b2; --line:#303744; --panel:#171b22; --accent:#7aa2ff; --bad:#ff8a80; --ok:#7bdba7; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
a {{ color:var(--accent); text-decoration:none; }} header {{ position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line); padding:14px 22px; display:flex; align-items:center; gap:18px; }}
header h1 {{ margin:0; font-size:18px; }} header nav {{ display:flex; gap:12px; flex-wrap:wrap; }} main {{ padding:18px 22px 40px; max-width:1480px; margin:0 auto; }}
.grid {{ display:grid; gap:14px; margin-bottom:14px; }} .cards {{ grid-template-columns:repeat(4,minmax(180px,1fr)); }} .two {{ grid-template-columns:1fr 1fr; }} @media(max-width:1000px) {{ .cards,.two {{ grid-template-columns:1fr; }} }}
.panel,.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; overflow:auto; }} .wide {{ width:100%; }}
h2 {{ margin:0 0 10px; font-size:15px; }} .muted {{ color:var(--muted); }} .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-weight:600; white-space:nowrap; }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }} pre {{ white-space:pre-wrap; margin:6px 0 0; color:var(--muted); font-size:12px; }}
.badge {{ display:inline-block; padding:2px 6px; border:1px solid var(--line); border-radius:999px; color:var(--muted); margin:0 4px 4px 0; font-size:12px; }}
.tabs,.groupnav {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 14px; }} .tabs a,.groupnav a,.btn {{ display:inline-block; padding:6px 9px; border:1px solid var(--line); border-radius:6px; background:var(--panel); }} .tabs a.active,.groupnav a.active {{ border-color:var(--accent); color:var(--accent); }}
.actions {{ display:flex; gap:5px; flex-wrap:wrap; }} .danger {{ color:var(--bad); }} .notice {{ padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:14px; }} .small {{ font-size:12px; }}
</style></head><body>{body}</body></html>"""


def _header(title: str) -> str:
    return f'<header><h1>{_e(title)}</h1><nav><a href="/admin">概览</a><a href="/admin/memory">记忆审计</a><a href="/admin/plugins">插件</a><a href="/trace">Trace</a><a href="/readyz">Ready JSON</a></nav></header>'


def _footer() -> str:
    return '<footer class="muted small" style="padding:0 22px 24px;max-width:1480px;margin:0 auto;">只建议通过 SSH 隧道访问，不对公网开放。</footer>'


def _group_nav(groups: tuple[int, ...], selected: int | None, *, path: str = '/admin') -> str:
    if not groups:
        return '<p class="muted">没有配置群。</p>'
    links = []
    for group_id in groups:
        active = 'active' if selected == group_id else ''
        links.append(f'<a class="{active}" href="{path}?group_id={group_id}">群 {group_id}</a>')
    return '<nav class="groupnav">' + ''.join(links) + '</nav>'


def _tab(path: str, status: str, group_id: int | None, active: bool) -> str:
    group_q = f'&group_id={group_id}' if group_id is not None else ''
    return f'<a class="{"active" if active else ""}" href="{path}?status={status}{group_q}">{_e(status)}</a>'


def _status_card(title: str, payload: dict[str, object]) -> str:
    ok = bool(payload.get('ok', True))
    cls = 'ok' if ok else 'bad'
    return f'<section class="card"><h2>{_e(title)} <span class="{cls}">{"OK" if ok else "BAD"}</span></h2><pre>{_e(_short_json(payload, 600))}</pre></section>'


def _model_card(routes: dict[str, str]) -> str:
    rows = ''.join(f'<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>' for k, v in sorted(routes.items()))
    return f'<section class="card"><h2>模型路由</h2><table>{rows}</table></section>'


def _approval_card(pending: list[object]) -> str:
    if not pending:
        return '<section class="card"><h2>审批队列</h2><p class="muted">暂无待审批。</p></section>'
    items = []
    for approval in pending[:6]:
        items.append(f'<li><b>{_e(str(getattr(approval, "approval_id", "")))}</b> 群 {getattr(approval, "group_id", "")} 触发：{_e(str(getattr(approval, "trigger_nickname", "")))}<br><span class="muted">{_e(_trim(str(getattr(approval, "trigger_text", "")), 120))}</span></li>')
    return f'<section class="card"><h2>审批队列：{len(pending)}</h2><ul>{"".join(items)}</ul></section>'



def _plugin_card(plugins: list[dict[str, Any]]) -> str:
    enabled = [plugin for plugin in plugins if plugin.get('enabled')]
    if not plugins:
        return '<section class="card"><h2>本地插件</h2><p class="muted">暂无插件 manifest。</p></section>'
    items = []
    for plugin in enabled[:6]:
        items.append(
            f'<li><b>{_e(plugin.get("name") or plugin.get("id"))}</b> '
            f'<span class="muted small">{_capability_count_badges(plugin)}</span></li>'
        )
    more = f'<p class="muted small">启用 {len(enabled)} / 总计 {len(plugins)}</p>'
    return f'<section class="card"><h2>本地插件</h2><ul>{"".join(items)}</ul>{more}<p><a class="btn" href="/admin/plugins">查看插件</a></p></section>'


def _plugin_table(plugins: list[dict[str, Any]], *, full: bool) -> str:
    if not plugins:
        return '<p class="muted">暂无插件 manifest。</p>'
    out = ['<table><tr><th>状态</th><th>插件</th><th>能力</th><th>权限</th><th>入口/说明</th></tr>']
    for plugin in sorted(plugins, key=lambda item: str(item.get('id') or '')):
        enabled = bool(plugin.get('enabled'))
        status = '<span class="ok">启用</span>' if enabled else '<span class="muted">关闭</span>'
        permissions = plugin.get('permissions') if isinstance(plugin.get('permissions'), list) else []
        permission_badges = ''.join(f'<span class="badge">{_e(item)}</span>' for item in permissions) or '<span class="muted small">无</span>'
        detail = _e(plugin.get('description') or '')
        if full:
            detail += f'<pre>entrypoint={_e(plugin.get("entrypoint") or "")}\npath={_e(plugin.get("path") or "")}</pre>'
        out.append(
            '<tr>'
            f'<td>{status}</td>'
            f'<td><b>{_e(plugin.get("name") or plugin.get("id"))}</b><br><span class="muted small">{_e(plugin.get("id"))} v{_e(plugin.get("version"))}</span></td>'
            f'<td>{_capability_count_badges(plugin)}</td>'
            f'<td>{permission_badges}</td>'
            f'<td>{detail}</td>'
            '</tr>'
        )
    out.append('</table>')
    return ''.join(out)


def _plugin_error_table(errors: list[dict[str, str]]) -> str:
    if not errors:
        return '<p class="muted">暂无加载错误。</p>'
    out = ['<table><tr><th>文件</th><th>错误</th></tr>']
    for error in errors:
        out.append(f'<tr><td>{_e(error.get("path"))}</td><td class="bad">{_e(error.get("error"))}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _capability_count_badges(plugin: dict[str, Any]) -> str:
    counts = plugin.get('capability_counts')
    if not isinstance(counts, dict):
        capabilities = plugin.get('capabilities') if isinstance(plugin.get('capabilities'), dict) else {}
        counts = {str(key): len(value) for key, value in capabilities.items() if isinstance(value, list)}
    labels = {
        'commands': '命令',
        'tools': '工具',
        'scheduled_tasks': '定时',
        'web_routes': '页面',
        'event_handlers': '事件',
    }
    badges = []
    for key, label in labels.items():
        count = int(counts.get(key, 0) or 0)
        if count:
            badges.append(f'<span class="badge">{label}×{count}</span>')
    return ''.join(badges) or '<span class="muted small">无能力声明</span>'


def _panel(title: str, inner: str) -> str:
    return f'<section class="panel"><h2>{_e(title)}</h2>{inner}</section>'


def _metric_table(rows: list[Any], *, show_meta: bool) -> str:
    if not rows:
        return '<p class="muted">暂无记录。</p>'
    out = ['<table><tr><th>时间</th><th>类型</th><th>阶段</th><th>动作</th><th>摘要</th></tr>']
    for row in rows:
        meta = _loads_json(_row(row, 'metadata_json'))
        summary = _metric_summary(meta) if show_meta else ''
        out.append(f'<tr><td class="small">{_fmt_time(_row(row, "created_at"))}</td><td>{_e(_row(row, "event_type"))}</td><td>{_e(_row(row, "stage"))}</td><td>{_e(_row(row, "action"))}</td><td>{summary}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _metric_summary(meta: dict[str, Any]) -> str:
    keys = ('target_note', 'reply_target', 'reason', 'text', 'decision_action', 'pipeline_mode', 'route', 'error', 'query')
    badges = []
    for key in keys:
        if key in meta and meta[key] not in (None, '', []):
            badges.append(f'<span class="badge">{_e(key)}={_e(_trim(str(meta[key]), 80))}</span>')
    if not badges:
        return f'<pre>{_e(_short_json(meta, 220))}</pre>'
    return ''.join(badges)


def _message_table(rows: list[Any]) -> str:
    if not rows:
        return '<p class="muted">暂无消息。</p>'
    out = ['<table><tr><th>ID</th><th>时间</th><th>说话人</th><th>内容</th><th>Segments</th></tr>']
    for row in rows:
        out.append(f'<tr><td>{_row(row,"id")}</td><td class="small">{_fmt_time(_row(row,"created_at"))}</td><td>{_e(_row(row,"nickname"))}<br><span class="muted small">#{_row(row,"user_id")}</span></td><td>{_e(_trim(_row(row,"text"), 160))}</td><td>{_segment_badges(_row(row,"message_segments_json"))}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _memory_atom_list(atoms: list[MemoryAtom]) -> str:
    if not atoms:
        return '<p class="muted">暂无记忆。</p><p><a class="btn" href="/admin/memory">打开记忆审计</a></p>'
    items = []
    for atom in atoms[:12]:
        items.append(f'<li><b>#{atom.id}</b> <span class="badge">{_e(atom.atom_type)}</span> {_e(_trim(atom.content, 110))}</li>')
    return '<ul>' + ''.join(items) + '</ul><p><a class="btn" href="/admin/memory">打开记忆审计</a></p>'


def _memory_atom_table(atoms: list[MemoryAtom], *, group_id: int | None, status: str) -> str:
    if not atoms:
        return '<p class="muted">没有符合条件的记忆。</p>'
    out = ['<table><tr><th>ID</th><th>类型/主体</th><th>内容</th><th>分数</th><th>状态</th><th>操作</th></tr>']
    for atom in atoms:
        subject = []
        if atom.subject_user_id is not None:
            subject.append(f'subject={atom.subject_user_id}')
        if atom.object_user_id is not None:
            subject.append(f'object={atom.object_user_id}')
        out.append(f'<tr><td>#{atom.id}</td><td><span class="badge">{_e(atom.atom_type)}</span><br><span class="muted small">{_e(" / ".join(subject))}</span></td><td>{_e(atom.content)}<br><span class="muted small">source={_e(atom.source)} updated={_fmt_time(atom.updated_at)}</span></td><td>conf {atom.confidence:.2f}<br>imp {atom.importance:.2f}</td><td>{_e(atom.status)}</td><td><div class="actions">{_memory_actions(atom.id, group_id=group_id, status=status)}</div></td></tr>')
    out.append('</table>')
    return ''.join(out)


def _memory_actions(atom_id: int, *, group_id: int | None, status: str) -> str:
    base = f'/admin/memory/action?atom_id={atom_id}&status={status}'
    if group_id is not None:
        base += f'&group_id={group_id}'
    specs = [('keep','保留'), ('boost','提权'), ('freeze','冻结'), ('wrong_person','错人'), ('expire','过期')]
    links = []
    for action, label in specs:
        cls = 'btn danger' if action in {'wrong_person', 'expire'} else 'btn'
        links.append(f'<a class="{cls}" href="{base}&action={action}">{label}</a>')
    return ''.join(links)


def _style_rule_list(rules: list[Any]) -> str:
    if not rules:
        return '<p class="muted">暂无风格规则。</p>'
    items = [f'<li><b>当{_e(rule.situation)}</b>：{_e(rule.style)}<br><span class="muted small">scope={_e(rule.scope)} conf={rule.confidence:.2f}</span></li>' for rule in rules]
    return '<ul>' + ''.join(items) + '</ul>'


def _profile_list(profiles: list[Any]) -> str:
    if not profiles:
        return '<p class="muted">暂无画像。</p>'
    items = []
    for profile in profiles:
        summary = getattr(profile, 'ai_summary', '') or '暂无 AI 摘要'
        items.append(f'<li><b>{_e(getattr(profile,"display_name", ""))}</b> <span class="muted small">#{getattr(profile,"user_id", "")}</span><br>{_e(_trim(summary, 120))}</li>')
    return '<ul>' + ''.join(items) + '</ul>'


def _usage_table(rows: list[Any]) -> str:
    if not rows:
        return '<p class="muted">暂无用量记录。</p>'
    out = ['<table><tr><th>任务</th><th>模型</th><th>次数</th><th>输入</th><th>输出</th><th>总计</th></tr>']
    for row in rows:
        out.append(f'<tr><td>{_e(row.task)}</td><td>{_e(_trim(row.model, 36))}</td><td>{row.call_count}</td><td>{row.prompt_tokens}</td><td>{row.completion_tokens}</td><td>{row.total_tokens}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _segment_badges(raw: str) -> str:
    segments = _loads_json_list(raw)
    if not segments:
        return '<span class="muted small">无</span>'
    counts: dict[str, int] = {}
    for segment in segments:
        typ = str(segment.get('type') or 'unknown') if isinstance(segment, dict) else 'unknown'
        counts[typ] = counts.get(typ, 0) + 1
    return ''.join(f'<span class="badge">{_e(k)}×{v}</span>' for k, v in sorted(counts.items()))


def _notice(text: str) -> str:
    return f'<div class="notice">{_e(text)}</div>' if text else ''


def _row(row: Any, key: str) -> str:
    try:
        value = row[key]
    except Exception:
        value = ''
    return '' if value is None else str(value)


def _loads_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or '{}')
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _loads_json_list(raw: str) -> list[Any]:
    try:
        value = json.loads(raw or '[]')
    except Exception:
        return []
    return value if isinstance(value, list) else []


def _short_json(payload: object, limit: int) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        text = str(payload)
    return _trim(text, limit)


def _fmt_time(value: object) -> str:
    try:
        ts = float(value)
    except Exception:
        return ''
    return time.strftime('%m-%d %H:%M:%S', time.localtime(ts))


def _trim(text: str, limit: int) -> str:
    compact = ' '.join(str(text or '').split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + '…'


def _e(value: object) -> str:
    return html.escape(str(value or ''), quote=True)
