from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urlencode

from .memory import MemoryAtom, MemoryStore, MemorySummary


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


def render_admin_tools_page(
    *,
    state: dict[str, Any],
    selected_group_id: int | None,
    notice: str = "",
    report_title: str = "",
    report_text: str = "",
) -> str:
    body = f"""
    {_header('工具控制台')}
    <main>
      {_notice(notice)}
      <section class="grid cards">
        {_tools_status_card('群聊决策', state.get('groups', []))}
        {_tools_kv_card('审查/审批', state.get('approval', {}))}
        {_tools_kv_card('工作强度', state.get('work_intensity', {}))}
        {_tools_private_card(state.get('private_chat', {}))}
      </section>
      <section class="grid two">
        {_panel('运行开关', _tools_switch_forms(state))}
        {_panel('审批人 / 私聊 / 强服从', _tools_user_forms(state))}
      </section>
      <section class="grid two">
        {_panel('模型路由', _tools_model_forms(state))}
        {_panel('黑话词典', _tools_jargon_forms(state, selected_group_id))}
      </section>
      <section class="panel wide"><h2>报告与调试</h2>{_tools_report_links(selected_group_id)}{_report_block(report_title, report_text)}</section>
      <section class="panel wide"><h2>QQ 工具单说明</h2>{_tool_docs(state.get('tool_docs', {}))}</section>
    </main>
    {_footer()}
    """
    return _page('张风雪工具控制台', body)



def render_memory_audit_page(
    *,
    memory: MemoryStore,
    groups: tuple[int, ...],
    selected_group_id: int | None,
    status: str,
    limit: int,
    notice: str = "",
    user_id: int | None = None,
    atom_type: str = "",
    q: str = "",
) -> str:
    group_id = selected_group_id or (groups[0] if groups else None)
    atoms = memory.admin_recent_memory_atoms(
        group_id=group_id,
        status=status,
        limit=limit,
        user_id=user_id,
        atom_type=atom_type,
        query=q,
    )
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
      <section class="panel wide"><h2>筛选</h2>{_memory_filter_form(group_id=group_id, status=status, limit=limit, user_id=user_id, atom_type=atom_type, q=q)}</section>
      <section class="panel wide"><h2>记忆单元 <span class="muted small">{len(atoms)} 条</span></h2>{_memory_atom_table(atoms, group_id=group_id, status=status, limit=limit, user_id=user_id, atom_type=atom_type, q=q)}</section>
    </main>
    {_footer()}
    """
    return _page('张风雪记忆审计', body)



def render_message_detail_page(
    *,
    memory: MemoryStore,
    message_id: int,
) -> str:
    row = memory.admin_message(message_id)
    if row is None:
        body = f"""
        {_header('消息链详情')}
        <main><section class="panel wide"><h2>消息不存在</h2><p class="muted">没有找到消息 #{_e(message_id)}。</p></section></main>
        {_footer()}
        """
        return _page('消息链详情', body)
    body = f"""
    {_header('消息链详情')}
    <main>
      <section class="panel wide"><h2>消息 #{_row(row, 'id')}</h2>{_message_detail_summary(row)}</section>
      <section class="grid two">
        {_panel('Message Segments', _segment_table(_row(row, 'message_segments_json')))}
        {_panel('Sender', _json_block(_row(row, 'sender_json')))}
      </section>
      <section class="panel wide"><h2>Raw Message</h2>{_json_block(_row(row, 'raw_message_json'))}</section>
    </main>
    {_footer()}
    """
    return _page('消息链详情', body)



def render_memory_atom_detail_page(
    *,
    memory: MemoryStore,
    atom_id: int,
    groups: tuple[int, ...],
    notice: str = "",
) -> str:
    atom = memory.memory_atom(atom_id)
    if atom is None:
        body = f"""
        {_header('记忆详情')}
        <main><section class="panel wide"><h2>记忆不存在</h2><p class="muted">没有找到记忆 #{_e(atom_id)}。</p></section></main>
        {_footer()}
        """
        return _page('记忆详情', body)
    events = memory.memory_atom_audit_trail(atom.id, limit=120)
    source_message = memory.admin_message_by_source(atom.group_id, atom.source_message_id) if atom.source_message_id else None
    body = f"""
    {_header('记忆详情')}
    <main>
      {_group_nav(groups, atom.group_id, path='/admin/memory')}
      {_notice(notice)}
      <section class="panel wide"><h2>记忆 #{atom.id}</h2>{_memory_atom_detail(atom, source_message)}</section>
      <section class="grid two">
        {_panel('纠正这条记忆', _memory_correction_form(atom))}
        {_panel('合并到另一条记忆', _memory_merge_form(atom))}
      </section>
      <section class="panel wide"><h2>审计轨迹</h2>{_memory_audit_table(events)}</section>
    </main>
    {_footer()}
    """
    return _page('记忆详情', body)



def render_admin_edit_page(
    *,
    editable_files: list[dict[str, str]],
    selected_key: str,
    content: str,
    notice: str = "",
) -> str:
    selected = next((item for item in editable_files if item.get('key') == selected_key), editable_files[0] if editable_files else {})
    body = f"""
    {_header('编辑 Prompt / 规则')}
    <main>
      {_notice(notice)}
      <section class="panel wide"><h2>选择文件</h2>{_editable_file_tabs(editable_files, selected_key)}</section>
      <section class="panel wide"><h2>{_e(selected.get('label', selected_key))}</h2>{_editable_file_form(selected_key=selected_key, content=content, description=selected.get('description', ''))}</section>
    </main>
    {_footer()}
    """
    return _page('编辑 Prompt / 规则', body)



def render_memory_summaries_page(
    *,
    memory: MemoryStore,
    groups: tuple[int, ...],
    selected_group_id: int | None,
    status: str,
    limit: int,
    q: str = "",
    notice: str = "",
) -> str:
    group_id = selected_group_id or (groups[0] if groups else None)
    summaries = memory.admin_recent_memory_summaries(group_id=group_id, status=status, limit=limit, query=q)
    body = f"""
    {_header('聊天回想')}
    <main>
      {_group_nav(groups, group_id, path='/admin/summaries')}
      {_notice(notice)}
      <section class="panel wide"><h2>筛选</h2>{_summary_filter_form(group_id=group_id, status=status, limit=limit, q=q)}</section>
      <section class="grid two">
        {_panel('新增人工回想', _summary_add_form(group_id=group_id))}
        {_panel('说明', '<p class="muted">锁定后的回想仍会进入上下文和 RAG，但你手动改过的内容会保留为人工版本。归档/过期的回想不会进入正常召回。</p>')}
      </section>
      <section class="panel wide"><h2>回想列表 <span class="muted small">{len(summaries)} 条</span></h2>{_summary_table(summaries, group_id=group_id, status=status, limit=limit, q=q)}</section>
    </main>
    {_footer()}
    """
    return _page('聊天回想', body)



def render_memory_summary_detail_page(
    *,
    memory: MemoryStore,
    summary_id: int,
    groups: tuple[int, ...],
    notice: str = "",
) -> str:
    summary = memory.memory_summary(summary_id)
    if summary is None:
        body = f"""
        {_header('回想详情')}
        <main><section class="panel wide"><h2>回想不存在</h2><p class="muted">没有找到回想 #{_e(summary_id)}。</p></section></main>
        {_footer()}
        """
        return _page('回想详情', body)
    body = f"""
    {_header('回想详情')}
    <main>
      {_group_nav(groups, summary.group_id, path='/admin/summaries')}
      {_notice(notice)}
      <section class="panel wide"><h2>回想 #{summary.id}</h2>{_summary_detail(summary)}</section>
      <section class="panel wide"><h2>编辑并保留</h2>{_summary_edit_form(summary)}</section>
    </main>
    {_footer()}
    """
    return _page('回想详情', body)



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
.actions {{ display:flex; gap:5px; flex-wrap:wrap; }} .danger {{ color:var(--bad); }} .notice {{ padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:14px; }} .small {{ font-size:12px; }} .mini {{ max-width:160px; }} .inline {{ display:inline-flex; gap:6px; align-items:end; flex-wrap:wrap; margin:4px 8px 8px 0; }} details {{ margin:8px 0; }} summary {{ cursor:pointer; color:var(--accent); }}
.filters {{ display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:10px; align-items:end; }} .field label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }} input,select,textarea {{ width:100%; padding:7px 8px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--fg); }} textarea {{ min-height:96px; }} button {{ padding:7px 10px; border:1px solid var(--accent); border-radius:6px; background:var(--accent); color:white; cursor:pointer; }} @media(max-width:1000px) {{ .filters {{ grid-template-columns:1fr 1fr; }} }}
</style></head><body>{body}</body></html>"""


def _header(title: str) -> str:
    return f'<header><h1>{_e(title)}</h1><nav><a href="/admin">概览</a><a href="/admin/tools">工具</a><a href="/admin/edit">编辑</a><a href="/admin/summaries">回想</a><a href="/admin/memory">记忆审计</a><a href="/admin/plugins">插件</a><a href="/trace">Trace</a><a href="/readyz">Ready JSON</a></nav></header>'


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





def _tools_status_card(title: str, groups: object) -> str:
    rows = groups if isinstance(groups, list) else []
    if not rows:
        return f'<section class="card"><h2>{_e(title)}</h2><p class="muted">没有目标群。</p></section>'
    enabled = sum(1 for row in rows if isinstance(row, dict) and row.get('enabled'))
    lines = ''.join(
        f'<li>群 {_e(row.get("group_id"))}: {"开启" if row.get("enabled") else "关闭"}<br><span class="muted small">persona={_e(row.get("persona"))} muted_left={_e(_join_display(row.get("muted_left_seconds", 0)))}s</span></li>'
        for row in rows if isinstance(row, dict)
    )
    return f'<section class="card"><h2>{_e(title)}</h2><p><b>{enabled}/{len(rows)}</b> 开启</p><ul>{lines}</ul></section>'


def _tools_kv_card(title: str, payload: object) -> str:
    data = payload if isinstance(payload, dict) else {}
    rows = ''.join(f'<tr><th>{_e(key)}</th><td>{_e(_join_display(value))}</td></tr>' for key, value in data.items())
    return f'<section class="card"><h2>{_e(title)}</h2><table>{rows}</table></section>'


def _tools_private_card(payload: object) -> str:
    data = payload if isinstance(payload, dict) else {}
    rows = [
        ('固定白名单', _join_display(data.get('config_ids'))),
        ('运行时白名单', _join_display(data.get('runtime_ids'))),
        ('命令专用', _join_display(data.get('command_only_ids'))),
        ('强服从', '开启' if data.get('force_obey_enabled') else '关闭'),
    ]
    return '<section class="card"><h2>私聊</h2><table>' + ''.join(f'<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>' for k, v in rows) + '</table></section>'


def _tools_switch_forms(state: dict[str, Any]) -> str:
    groups = state.get('groups') if isinstance(state.get('groups'), list) else []
    group_options = ''.join(f'<option value="{_e(row.get("group_id"))}">群 {_e(row.get("group_id"))}</option>' for row in groups if isinstance(row, dict))
    group_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="group_decision">'
        f'<div class="field"><label>群</label><select name="group_id">{group_options}</select></div>'
        '<div class="field"><label>状态</label><select name="enabled"><option value="1">开启群聊决策</option><option value="0">关闭群聊决策</option></select></div>'
        '<button type="submit">应用</button></form>'
    )
    review = state.get('approval') if isinstance(state.get('approval'), dict) else {}
    review_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="review_enabled">'
        '<div class="field"><label>人工审查</label><select name="enabled"><option value="1">开启审查</option><option value="0">关闭审查</option></select></div>'
        '<button type="submit">应用</button></form>'
    )
    auto_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="approval_auto_send">'
        f'<div class="field"><label>免审概率%</label><input class="mini" name="percent" value="{_e(review.get("auto_send_percent", 0))}"></div>'
        '<button type="submit">设置</button></form>'
    )
    work = state.get('work_intensity') if isinstance(state.get('work_intensity'), dict) else {}
    work_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="work_intensity">'
        f'<div class="field"><label>工作强度%</label><input class="mini" name="percent" value="{_e(work.get("current_percent", 0))}"></div>'
        '<button type="submit">设置</button></form>'
    )
    quiet_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="quiet_group">'
        f'<div class="field"><label>群</label><select name="group_id">{group_options}</select></div>'
        '<div class="field"><label>闭嘴分钟，0 为解除</label><input class="mini" name="minutes" value="10"></div>'
        '<button type="submit">设置</button></form>'
    )
    review_now = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="daily_review">'
        '<div class="field"><label>复盘</label><select name="mode"><option value="today">今天到现在</option><option value="due">补发到期</option></select></div>'
        '<button type="submit">发送复盘</button></form>'
    )
    proactive_now = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="proactive_chat">'
        f'<div class="field"><label>主动发言群</label><select name="group_id">{group_options}</select></div>'
        '<button type="submit">主动发言一次</button></form>'
    )
    return group_form + review_form + auto_form + work_form + quiet_form + review_now + proactive_now


def _tools_user_forms(state: dict[str, Any]) -> str:
    approval = state.get('approval') if isinstance(state.get('approval'), dict) else {}
    private = state.get('private_chat') if isinstance(state.get('private_chat'), dict) else {}
    approval_report = (
        f'<p><b>主人：</b>{_e(_join_display(approval.get("owners")))}<br>'
        f'<b>基础审批：</b>{_e(_join_display(approval.get("basic_users")))}<br>'
        f'<b>接收审批单：</b>{_e(_join_display(approval.get("all_users")))}</p>'
    )
    approver_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="approval_user">'
        '<div class="field"><label>基础审批人 QQ</label><input name="user_id" class="mini"></div>'
        '<div class="field"><label>动作</label><select name="op"><option value="add">添加</option><option value="delete">删除</option></select></div>'
        '<button type="submit">应用</button></form>'
    )
    private_report = f'<p><b>运行时私聊：</b>{_e(_join_display(private.get("runtime_ids")))}</p>'
    private_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="private_whitelist">'
        '<div class="field"><label>私聊 QQ</label><input name="user_id" class="mini"></div>'
        '<div class="field"><label>动作</label><select name="op"><option value="add">添加</option><option value="delete">删除</option></select></div>'
        '<button type="submit">应用</button></form>'
    )
    force_form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="force_obey">'
        '<div class="field"><label>测试号强服从</label><select name="enabled"><option value="1">开启</option><option value="0">关闭</option></select></div>'
        '<button type="submit">应用</button></form>'
    )
    return approval_report + approver_form + private_report + private_form + force_form


def _tools_model_forms(state: dict[str, Any]) -> str:
    routes = state.get('models') if isinstance(state.get('models'), list) else []
    catalog = state.get('model_catalog') if isinstance(state.get('model_catalog'), list) else []
    route_options = ''.join(f'<option value="{_e(row.get("route"))}">{_e(row.get("title"))} ({_e(row.get("route"))})</option>' for row in routes if isinstance(row, dict))
    model_options = ''.join(f'<option value="{_e(row.get("label"))}">{_e(row.get("label"))} - {_e(row.get("source"))}</option>' for row in catalog if isinstance(row, dict))
    form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="model_route">'
        f'<div class="field"><label>流程</label><select name="route">{route_options}<option value="utility_group">工具组</option></select></div>'
        f'<div class="field"><label>模型</label><select name="model">{model_options}</select></div>'
        '<button type="submit">切换</button></form>'
        '<form class="inline" method="post" action="/admin/tools/action"><input type="hidden" name="action" value="model_reset"><button type="submit">清模型覆盖</button></form>'
    )
    table = ['<table><tr><th>流程</th><th>当前</th><th>config</th><th>fallback</th></tr>']
    for row in routes:
        if not isinstance(row, dict):
            continue
        suffix = ' <span class="badge">覆盖</span>' if row.get('overridden') else ''
        table.append(f'<tr><td>{_e(row.get("title"))}<br><span class="muted small">{_e(row.get("flow"))}</span></td><td>{_e(row.get("active"))}{suffix}</td><td>{_e(row.get("configured"))}</td><td>{_e(row.get("fallback"))}</td></tr>')
    table.append('</table>')
    return form + ''.join(table)


def _tools_jargon_forms(state: dict[str, Any], selected_group_id: int | None) -> str:
    group_id = selected_group_id or state.get('selected_group_id')
    entries = state.get('jargon_entries') if isinstance(state.get('jargon_entries'), list) else []
    form = (
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="jargon_add">'
        f'<input type="hidden" name="group_id" value="{_e(group_id or "")}">'
        '<div class="field"><label>词</label><input name="term" class="mini"></div>'
        '<div class="field"><label>解释</label><input name="meaning" placeholder="指代：中国"></div>'
        '<button type="submit">添加/更新</button></form>'
        '<form class="inline" method="post" action="/admin/tools/action">'
        '<input type="hidden" name="action" value="jargon_delete">'
        f'<input type="hidden" name="group_id" value="{_e(group_id or "")}">'
        '<div class="field"><label>删除词</label><input name="term" class="mini"></div>'
        '<button type="submit">删除</button></form>'
    )
    if not entries:
        return form + '<p class="muted">暂无自定义黑话。</p>'
    rows = ''.join(f'<tr><td>{_e(row.get("term"))}</td><td>{_e(row.get("explanation"))}</td><td>{_e(row.get("created_by"))}</td><td>{_fmt_time(row.get("created_at"))}</td></tr>' for row in entries if isinstance(row, dict))
    return form + '<table><tr><th>词</th><th>解释</th><th>创建者</th><th>时间</th></tr>' + rows + '</table>'


def _tools_report_links(group_id: int | None) -> str:
    params = {'group_id': group_id} if group_id is not None else {}
    specs = [
        ('blocked', '拦截20'), ('metrics', '统计今日'), ('token', 'Token'), ('memory', '回想20'),
        ('style', '风格20'), ('members', '群友画像20'), ('atoms', '记忆单元20'),
        ('rag_status', 'RAG状态'), ('rag_knowledge', 'RAG知识库'), ('rag_feedback', 'RAG反馈'), ('rag_eval', 'RAG评测'),
    ]
    links = []
    for kind, label in specs:
        query = dict(params)
        query.update({'kind': kind, 'limit': 20, 'window': 'today'})
        links.append(f'<a class="btn" href="/admin/tools/report{_query_suffix(query)}">{_e(label)}</a>')
    rag_test = (
        '<form class="inline" method="get" action="/admin/tools/report">'
        '<input type="hidden" name="kind" value="rag_test">'
        f'<input type="hidden" name="group_id" value="{_e(group_id or "")}">'
        '<div class="field"><label>RAG测试 query</label><input name="query" placeholder="以前谁聊过..."></div>'
        '<button type="submit">测试</button></form>'
    )
    return '<div class="actions">' + ''.join(links) + '</div>' + rag_test


def _report_block(title: str, text: str) -> str:
    if not text:
        return ''
    return f'<section class="panel wide" style="margin-top:12px"><h2>{_e(title or "报告")}</h2><pre>{_e(text)}</pre></section>'


def _tool_docs(docs: object) -> str:
    data = docs if isinstance(docs, dict) else {}
    if not data:
        return '<p class="muted">暂无工具说明。</p>'
    out = []
    for key, text in data.items():
        out.append(f'<details><summary>{_e(key)}</summary><pre>{_e(text)}</pre></details>')
    return ''.join(out)


def _join_display(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return '、'.join(str(item) for item in value) or '无'
    if value in (None, ''):
        return '无'
    return str(value)


def _editable_file_tabs(editable_files: list[dict[str, str]], selected_key: str) -> str:
    if not editable_files:
        return '<p class="muted">没有可编辑文件。</p>'
    links = []
    for item in editable_files:
        key = item.get('key', '')
        cls = 'active' if key == selected_key else ''
        links.append(f'<a class="{cls}" href="/admin/edit?file={_e(key)}">{_e(item.get("label", key))}</a>')
    return '<nav class="tabs">' + ''.join(links) + '</nav>'


def _editable_file_form(*, selected_key: str, content: str, description: str) -> str:
    return (
        f'<p class="muted">{_e(description)}</p>'
        '<form method="post" action="/admin/edit/save">'
        f'<input type="hidden" name="file" value="{_e(selected_key)}">'
        '<div class="field"><label>内容</label>'
        f'<textarea name="content" style="min-height:620px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;">{_e(content)}</textarea></div>'
        '<p class="muted small">保存时会先备份原文件并校验 YAML。Prompt 文件保存后立即热重载；config.yaml 多数配置仍需重启后端。</p>'
        '<button type="submit">保存并校验</button>'
        '</form>'
    )


def _summary_filter_form(*, group_id: int | None, status: str, limit: int, q: str) -> str:
    group_value = '' if group_id is None else str(group_id)
    status_options = ''.join(
        f'<option value="{_e(item)}" {"selected" if item == status else ""}>{_e(item)}</option>'
        for item in ('active', 'archived', 'expired', 'all')
    )
    return (
        '<form class="filters" method="get" action="/admin/summaries">'
        f'<div class="field"><label>群号</label><input name="group_id" value="{_e(group_value)}"></div>'
        f'<div class="field"><label>状态</label><select name="status">{status_options}</select></div>'
        f'<div class="field"><label>关键词</label><input name="q" value="{_e(q)}" placeholder="summary/cue/id"></div>'
        f'<div class="field"><label>数量</label><input name="limit" value="{_e(limit)}"></div>'
        '<div class="field"><button type="submit">筛选</button></div>'
        '</form>'
    )


def _summary_add_form(*, group_id: int | None) -> str:
    group_value = '' if group_id is None else str(group_id)
    return (
        '<form method="post" action="/admin/summaries/add">'
        f'<div class="field"><label>群号</label><input name="group_id" value="{_e(group_value)}"></div>'
        '<div class="field"><label>回想内容</label><textarea name="summary" placeholder="手动加入一条长期聊天回想"></textarea></div>'
        '<div class="field"><label>召回关键词，逗号分隔</label><input name="recall_cues" placeholder="小鸟, Claude, 旧梗"></div>'
        '<label class="small"><input type="checkbox" name="locked" value="1" checked> 锁定保留</label><br>'
        '<button type="submit">新增回想</button>'
        '</form>'
    )


def _summary_table(
    summaries: list[MemorySummary],
    *,
    group_id: int | None,
    status: str,
    limit: int,
    q: str,
) -> str:
    if not summaries:
        return '<p class="muted">暂无回想。</p>'
    params = _summary_filter_params(group_id=group_id, status=status, limit=limit, q=q)
    out = ['<table><tr><th>ID</th><th>时间</th><th>状态</th><th>关键词</th><th>摘要</th><th>操作</th></tr>']
    for summary in summaries:
        action_links = [f'<a class="btn" href="/admin/summaries/{summary.id}{_query_suffix(params)}">详情/编辑</a>']
        for action, label in (("lock", "锁定"), ("unlock", "解锁"), ("active", "恢复"), ("archive", "归档"), ("expire", "过期")):
            cls = 'btn danger' if action in {'archive', 'expire'} else 'btn'
            action_links.append(f'<a class="{cls}" href="/admin/summaries/action{_query_suffix({**params, "summary_id": summary.id, "action": action})}">{label}</a>')
        out.append(
            f'<tr><td><a href="/admin/summaries/{summary.id}">#{summary.id}</a></td>'
            f'<td class="small">{_fmt_time(summary.start_at)} - {_fmt_time(summary.end_at)}<br>更新 {_fmt_time(summary.updated_at)}</td>'
            '<td>' + _e(summary.status) + (' <span class="badge">locked</span>' if summary.locked else '') + '</td>'
            f'<td>{_e(", ".join(summary.recall_cues))}</td>'
            f'<td>{_e(_trim(summary.summary, 220))}</td>'
            f'<td><div class="actions">{"".join(action_links)}</div></td></tr>'
        )
    out.append('</table>')
    return ''.join(out)


def _summary_filter_params(*, group_id: int | None, status: str, limit: int, q: str) -> dict[str, object]:
    params: dict[str, object] = {'status': status or 'active', 'limit': limit}
    if group_id is not None:
        params['group_id'] = group_id
    if q.strip():
        params['q'] = q.strip()
    return params


def _summary_detail(summary: MemorySummary) -> str:
    rows = [
        ('群号', summary.group_id),
        ('状态', f'{summary.status} locked={summary.locked}'),
        ('时间', f'{_fmt_time(summary.start_at)} - {_fmt_time(summary.end_at)}'),
        ('创建/更新', f'{_fmt_time(summary.created_at)} / {_fmt_time(summary.updated_at)}'),
        ('关键词', ', '.join(summary.recall_cues)),
        ('内容', summary.summary),
    ]
    return '<table>' + ''.join(f'<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>' for key, value in rows) + '</table>'


def _summary_edit_form(summary: MemorySummary) -> str:
    cues = ', '.join(summary.recall_cues)
    status_options = ''.join(
        f'<option value="{_e(item)}" {"selected" if item == summary.status else ""}>{_e(item)}</option>'
        for item in ('active', 'archived', 'expired')
    )
    locked_checked = 'checked' if summary.locked else ''
    return (
        '<form method="post" action="/admin/summaries/save">'
        f'<input type="hidden" name="summary_id" value="{summary.id}">'
        f'<div class="field"><label>内容</label><textarea name="summary">{_e(summary.summary)}</textarea></div>'
        f'<div class="field"><label>召回关键词，逗号分隔</label><input name="recall_cues" value="{_e(cues)}"></div>'
        f'<div class="field"><label>状态</label><select name="status">{status_options}</select></div>'
        f'<label class="small"><input type="checkbox" name="locked" value="1" {locked_checked}> 锁定保留这条人工回想</label><br>'
        '<button type="submit">保存回想</button>'
        '</form>'
    )


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
            detail += _capability_detail_list(plugin)
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


def _capability_detail_list(plugin: dict[str, Any]) -> str:
    capabilities = plugin.get('capabilities') if isinstance(plugin.get('capabilities'), dict) else {}
    labels = {
        'commands': '命令',
        'tools': '工具',
        'scheduled_tasks': '定时',
        'web_routes': '页面',
        'event_handlers': '事件',
    }
    rows: list[str] = []
    for key, label in labels.items():
        items = capabilities.get(key)
        if not isinstance(items, list) or not items:
            continue
        rows.append(f'<b>{label}</b>')
        rows.append('<ul class=small>')
        for item in items:
            if not isinstance(item, dict):
                rows.append(f'<li>{_e(str(item))}</li>')
                continue
            name = item.get('name') or item.get('command') or item.get('task') or item.get('route') or item.get('event') or item.get('handler') or ''
            target = item.get('target') or item.get('handler') or item.get('tool') or item.get('command') or item.get('task') or item.get('route') or item.get('event') or ''
            permission = item.get('permission') or ''
            desc = item.get('description') or ''
            rows.append(
                '<li>'
                f'<code>{_e(str(name))}</code> '
                f'<span class=muted>target={_e(str(target))}</span> '
                f'<span class=badge>{_e(str(permission) or "no-permission")}</span><br>'
                f'<span class=muted>{_e(str(desc))}</span>'
                '</li>'
            )
        rows.append('</ul>')
    return ''.join(rows)


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
    out = ['<table><tr><th>ID</th><th>时间</th><th>说话人</th><th>内容</th><th>链路</th></tr>']
    for row in rows:
        message_id = _row(row, "id")
        out.append(
            f'<tr><td><a href="/admin/messages/{_e(message_id)}">#{_e(message_id)}</a></td>'
            f'<td class="small">{_fmt_time(_row(row,"created_at"))}<br><span class="muted">{_e(_row(row,"session_id"))}</span></td>'
            f'<td>{_e(_row(row,"nickname"))}<br><span class="muted small">#{_row(row,"user_id")}</span></td>'
            f'<td>{_e(_trim(_row(row,"text"), 160))}</td>'
            f'<td>{_segment_badges(_row(row,"message_segments_json"))}<br><span class="muted small">source={_e(_row(row,"source_message_id"))}</span></td></tr>'
        )
    out.append('</table>')
    return ''.join(out)


def _memory_atom_list(atoms: list[MemoryAtom]) -> str:
    if not atoms:
        return '<p class="muted">暂无记忆。</p><p><a class="btn" href="/admin/memory">打开记忆审计</a></p>'
    items = []
    for atom in atoms[:12]:
        items.append(f'<li><b>#{atom.id}</b> <span class="badge">{_e(atom.atom_type)}</span> {_e(_trim(atom.content, 110))}</li>')
    return '<ul>' + ''.join(items) + '</ul><p><a class="btn" href="/admin/memory">打开记忆审计</a></p>'


def _memory_atom_table(
    atoms: list[MemoryAtom],
    *,
    group_id: int | None,
    status: str,
    limit: int = 80,
    user_id: int | None = None,
    atom_type: str = "",
    q: str = "",
) -> str:
    if not atoms:
        return '<p class="muted">没有符合条件的记忆。</p>'
    out = ['<p class="muted small">No. 是当前筛选结果里的显示序号；真实 ID 是数据库主键，会因纠正、合并、过期和新建而跳号，这是正常的审计设计。</p><table><tr><th>No.</th><th>真实 ID</th><th>类型/主体</th><th>内容</th><th>分数</th><th>状态</th><th>操作</th></tr>']
    for number, atom in enumerate(atoms, start=1):
        subject = []
        if atom.subject_user_id is not None:
            subject.append(f'subject={atom.subject_user_id}')
        if atom.object_user_id is not None:
            subject.append(f'object={atom.object_user_id}')
        detail_href = f'/admin/memory/{atom.id}' + _query_suffix(_memory_filter_params(group_id=group_id, status=status, limit=limit, user_id=user_id, atom_type=atom_type, q=q))
        out.append(
            f'<tr><td>{number}</td><td><a href="{detail_href}">#{atom.id}</a></td>'
            f'<td><span class="badge">{_e(atom.atom_type)}</span><br><span class="muted small">{_e(" / ".join(subject))}</span></td>'
            f'<td>{_e(atom.content)}<br><span class="muted small">source={_e(atom.source)} updated={_fmt_time(atom.updated_at)}</span></td>'
            f'<td>conf {atom.confidence:.2f}<br>imp {atom.importance:.2f}</td><td>{_e(atom.status)}</td>'
            f'<td><div class="actions">{_memory_actions(atom.id, group_id=group_id, status=status, limit=limit, user_id=user_id, atom_type=atom_type, q=q)}</div></td></tr>'
        )
    out.append('</table>')
    return ''.join(out)


def _memory_actions(
    atom_id: int,
    *,
    group_id: int | None,
    status: str,
    limit: int = 80,
    user_id: int | None = None,
    atom_type: str = "",
    q: str = "",
) -> str:
    params = _memory_filter_params(group_id=group_id, status=status, limit=limit, user_id=user_id, atom_type=atom_type, q=q)
    specs = [('keep','保留'), ('boost','提权'), ('freeze','冻结'), ('wrong_person','错人'), ('expire','过期')]
    links = [f'<a class="btn" href="/admin/memory/{atom_id}{_query_suffix(params)}">详情</a>']
    for action, label in specs:
        cls = 'btn danger' if action in {'wrong_person', 'expire'} else 'btn'
        action_params = {**params, 'atom_id': atom_id, 'action': action}
        links.append(f'<a class="{cls}" href="/admin/memory/action{_query_suffix(action_params)}">{label}</a>')
    return ''.join(links)


def _memory_filter_form(
    *,
    group_id: int | None,
    status: str,
    limit: int,
    user_id: int | None,
    atom_type: str,
    q: str,
) -> str:
    group_value = '' if group_id is None else str(group_id)
    user_value = '' if user_id is None else str(user_id)
    status_options = ''.join(
        f'<option value="{_e(item)}" {"selected" if item == status else ""}>{_e(item)}</option>'
        for item in ('active', 'disputed', 'expired', 'superseded', 'all')
    )
    return (
        '<form class="filters" method="get" action="/admin/memory">'
        f'<div class="field"><label>群号</label><input name="group_id" value="{_e(group_value)}" placeholder="1026813421"></div>'
        f'<div class="field"><label>状态</label><select name="status">{status_options}</select></div>'
        f'<div class="field"><label>用户 QQ</label><input name="user_id" value="{_e(user_value)}" placeholder="subject/object"></div>'
        f'<div class="field"><label>类型</label><input name="atom_type" value="{_e(atom_type)}" placeholder="fact/relation/preference"></div>'
        f'<div class="field"><label>关键词</label><input name="q" value="{_e(q)}" placeholder="内容/来源/消息ID"></div>'
        f'<div class="field"><label>数量</label><input name="limit" value="{_e(limit)}"></div>'
        '<div class="field"><button type="submit">筛选</button></div>'
        '</form>'
    )


def _memory_filter_params(
    *,
    group_id: int | None,
    status: str,
    limit: int,
    user_id: int | None,
    atom_type: str,
    q: str,
) -> dict[str, object]:
    params: dict[str, object] = {'status': status or 'active', 'limit': limit}
    if group_id is not None:
        params['group_id'] = group_id
    if user_id is not None:
        params['user_id'] = user_id
    if atom_type.strip():
        params['atom_type'] = atom_type.strip()
    if q.strip():
        params['q'] = q.strip()
    return params


def _query_suffix(params: dict[str, object]) -> str:
    clean = {str(key): value for key, value in params.items() if value not in (None, '')}
    encoded = urlencode(clean)
    return f'?{encoded}' if encoded else ''


def _message_detail_summary(row: Any) -> str:
    rows = [
        ('群/会话', f'{_row(row, "group_id")} / {_row(row, "session_id")}'),
        ('说话人', f'{_row(row, "nickname")} #{_row(row, "user_id")}'),
        ('时间', _fmt_time(_row(row, 'created_at'))),
        ('source', f'{_row(row, "source_kind")} / {_row(row, "source_message_id")}'),
        ('correlation', _row(row, 'correlation_id')),
        ('文本', _row(row, 'text')),
    ]
    return '<table>' + ''.join(f'<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>' for k, v in rows) + '</table>'


def _segment_table(raw: str) -> str:
    segments = _loads_json_list(raw)
    if not segments:
        return '<p class="muted">没有保存 message segments。</p>'
    out = ['<table><tr><th>#</th><th>type</th><th>data</th></tr>']
    for index, segment in enumerate(segments):
        if isinstance(segment, dict):
            typ = segment.get('type') or 'unknown'
            data = segment.get('data')
            seg_index = segment.get('index', index)
        else:
            typ = 'unknown'
            data = segment
            seg_index = index
        out.append(f'<tr><td>{_e(seg_index)}</td><td><span class="badge">{_e(typ)}</span></td><td><pre>{_e(_short_json(data, 1600))}</pre></td></tr>')
    out.append('</table>')
    return ''.join(out)


def _json_block(raw: str) -> str:
    if not raw:
        return '<p class="muted">无。</p>'
    try:
        parsed = json.loads(raw)
        text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        text = raw
    return f'<pre>{_e(_trim(text, 12000))}</pre>'


def _memory_atom_detail(atom: MemoryAtom, source_message: Any | None) -> str:
    source = _e(atom.source_message_id or '')
    if source_message is not None:
        source = f'<a href="/admin/messages/{_row(source_message, "id")}">{_e(atom.source_message_id)}</a>'
    rows = [
        ('类型', atom.atom_type),
        ('主体/对象', f'subject={atom.subject_user_id or ""} object={atom.object_user_id or ""}'),
        ('内容', atom.content),
        ('状态', atom.status),
        ('分数', f'confidence={atom.confidence:.2f} importance={atom.importance:.2f}'),
        ('证据', f'{atom.evidence_type} / {atom.source} / {source}'),
        ('时间', f'observed={_fmt_time(atom.observed_at)} valid_from={_fmt_time(atom.valid_from)} valid_to={_fmt_time(atom.valid_to)} updated={_fmt_time(atom.updated_at)}'),
        ('替代关系', f'supersedes={atom.supersedes_id or ""}'),
    ]
    out = ['<table>']
    for key, value in rows:
        if key == '证据':
            out.append(f'<tr><th>{_e(key)}</th><td>{value}</td></tr>')
        else:
            out.append(f'<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>')
    out.append('</table>')
    return ''.join(out)


def _memory_correction_form(atom: MemoryAtom) -> str:
    return (
        '<form method="post" action="/admin/memory/correct">'
        f'<input type="hidden" name="atom_id" value="{atom.id}">'
        f'<div class="field"><label>新内容</label><textarea name="content">{_e(atom.content)}</textarea></div>'
        '<div class="filters" style="margin-top:10px;">'
        f'<div class="field"><label>类型</label><input name="atom_type" value="{_e(atom.atom_type)}"></div>'
        f'<div class="field"><label>subject_user_id</label><input name="subject_user_id" value="{_e(atom.subject_user_id or "")}"></div>'
        f'<div class="field"><label>object_user_id</label><input name="object_user_id" value="{_e(atom.object_user_id or "")}"></div>'
        '<div class="field"><label>原因</label><input name="reason" value="WebUI 手动纠正"></div>'
        '<div class="field"><button type="submit">封存旧记忆并创建新记忆</button></div>'
        '</div></form>'
    )


def _memory_merge_form(atom: MemoryAtom) -> str:
    return (
        '<form method="post" action="/admin/memory/merge">'
        f'<input type="hidden" name="source_atom_id" value="{atom.id}">'
        '<div class="field"><label>合并到目标记忆 ID</label><input name="target_atom_id" placeholder="例如 123"></div>'
        '<div class="field"><label>原因</label><input name="reason" value="WebUI 合并重复记忆"></div>'
        f'<p class="muted small">当前 #{atom.id} 会被标为 superseded，目标记忆保留并提取较高置信/重要度。</p>'
        '<button type="submit">合并</button>'
        '</form>'
    )


def _memory_audit_table(events: list[Any]) -> str:
    if not events:
        return '<p class="muted">暂无审计事件。</p>'
    out = ['<table><tr><th>时间</th><th>动作</th><th>证据</th><th>操作者</th><th>详情</th></tr>']
    for event in events:
        evidence = f'{getattr(event, "evidence_type", "")} / {getattr(event, "source", "")}'
        if getattr(event, 'source_message_id', None):
            evidence += f' / {getattr(event, "source_message_id")}'
        meta = getattr(event, 'metadata', {}) or {}
        detail = getattr(event, 'detail', '')
        if meta:
            detail += '\n' + json.dumps(meta, ensure_ascii=False)
        out.append(
            f'<tr><td class="small">{_fmt_time(getattr(event, "created_at", 0))}</td>'
            f'<td>{_e(getattr(event, "action", ""))}</td><td>{_e(evidence)}</td>'
            f'<td>{_e(getattr(event, "actor_user_id", "") or "")}</td><td><pre>{_e(detail)}</pre></td></tr>'
        )
    out.append('</table>')
    return ''.join(out)


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
        group_id = getattr(profile, "group_id", "")
        user_id = getattr(profile, "user_id", "")
        link = f'/admin/memory?group_id={_e(group_id)}&status=all&user_id={_e(user_id)}' if group_id and user_id else '/admin/memory'
        items.append(f'<li><b><a href="{link}">{_e(getattr(profile,"display_name", ""))}</a></b> <span class="muted small">#{user_id}</span><br>{_e(_trim(summary, 120))}</li>')
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
    return "" if value is None else html.escape(str(value), quote=True)
