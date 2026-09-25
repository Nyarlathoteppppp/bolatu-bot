from qq_social_agent.admin_ui import (
    _group_flow_table,
    render_admin_dashboard,
    render_admin_edit_page,
    render_admin_tools_page,
    render_memory_atom_detail_page,
    render_memory_audit_page,
    render_memory_summaries_page,
    render_memory_summary_detail_page,
    render_message_detail_page,
)
from qq_social_agent.memory import MemoryStore


def test_group_flow_table_links_gates_and_end_to_end_timing(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    correlation_id = 'group:100:<unsafe>'
    for event_type, stage, action, metadata, created_at in (
        ('pipeline_receive', 'receive', 'start', {}, 100.0),
        ('group_gate', 'work_intensity', 'passed', {'percent': 100}, 101.0),
        ('group_flow_timing', 'lock_wait', 'completed', {'elapsed_ms': 7}, 102.0),
        ('decision_result', 'llm', 'reply', {'elapsed_ms': 800}, 103.0),
        ('message_sent', 'send', 'reply', {'receive_elapsed_ms': 1400}, 104.0),
    ):
        memory.add_metric_event(
            event_type=event_type, group_id=100, stage=stage, action=action,
            metadata={'correlation_id': correlation_id, **metadata}, created_at=created_at,
        )
    rows = memory.admin_recent_metric_events(group_id=100, limit=30)
    rendered = _group_flow_table(rows)
    assert 'work_intensity:passed(100%)' in rendered
    assert '等锁 7' in rendered
    assert '收到至此 1400' in rendered
    assert '已发送' in rendered
    assert '&lt;unsafe&gt;' in rendered
    assert 'trace_id=group%3A100%3A%3Cunsafe%3E' in rendered


def test_admin_message_detail_renders_saved_message_chain(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    assert memory.add_message(
        1026813421,
        184589072,
        "小鸟",
        "风雪你看这个",
        source_message_id="m1",
        session_id="group:1026813421",
        message_segments_json='[{"index":0,"type":"at","data":{"qq":"1801507496"}},{"index":1,"type":"text","data":{"text":"风雪你看这个"}}]',
        raw_message_json='{"message_id":"m1","message_type":"group"}',
        sender_json='{"card":"小鸟","user_id":184589072}',
    )
    row = memory.admin_recent_messages(group_id=1026813421, limit=1)[0]

    html = render_message_detail_page(memory=memory, message_id=int(row["id"]))

    assert "Message Segments" in html
    assert "Raw Message" in html
    assert "at" in html
    assert "1801507496" in html


def test_admin_memory_filters_and_detail_page(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    bird_id = memory.upsert_memory_atom(
        atom_type="preference",
        group_id=1026813421,
        subject_user_id=184589072,
        content="小鸟需要超级温柔地回复",
        source="test",
        confidence=0.9,
        importance=0.9,
    )
    memory.upsert_memory_atom(
        atom_type="fact",
        group_id=1026813421,
        subject_user_id=3370998238,
        content="瑰奇负责审批",
        source="test",
    )

    filtered = memory.admin_recent_memory_atoms(
        group_id=1026813421,
        status="all",
        user_id=184589072,
        atom_type="preference",
        query="温柔",
        limit=20,
    )

    assert [atom.id for atom in filtered] == [bird_id]
    list_html = render_memory_audit_page(
        memory=memory,
        groups=(1026813421,),
        selected_group_id=1026813421,
        status="all",
        limit=20,
        user_id=184589072,
        atom_type="preference",
        q="温柔",
    )
    detail_html = render_memory_atom_detail_page(
        memory=memory,
        atom_id=bird_id,
        groups=(1026813421,),
    )

    assert "用户 QQ" in list_html
    assert f"/admin/memory/{bird_id}" in list_html
    assert "纠正这条记忆" in detail_html
    assert "审计轨迹" in detail_html


def test_admin_can_merge_memory_atoms(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    target = memory.upsert_memory_atom(
        atom_type="fact",
        group_id=1026813421,
        content="科代喜欢反串政治梗",
        source="test",
        confidence=0.7,
        importance=0.5,
    )
    source = memory.upsert_memory_atom(
        atom_type="fact",
        group_id=1026813421,
        content="邪恶代代和可爱代代是同一个人",
        source="test",
        confidence=0.9,
        importance=0.8,
    )

    assert memory.admin_merge_memory_atoms(source, target, actor_user_id=0, note="重复合并")

    source_atom = memory.memory_atom(source)
    target_atom = memory.memory_atom(target)
    assert source_atom is not None and source_atom.status == "superseded"
    assert source_atom.supersedes_id == target
    assert target_atom is not None and target_atom.importance >= 0.8
    assert [event.action for event in memory.memory_atom_audit_trail(source)][-1] == "merged_into"
    assert [event.action for event in memory.memory_atom_audit_trail(target)][-1] == "merged_from"

def test_admin_edit_and_memory_summary_pages_render(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(6):
        memory.add_message(1026813421, 100 + index, f"u{index}", f"m{index}", created_at=100 + index)
    batch = memory.messages_for_mid_summary(1026813421, keep_recent=0, batch_size=6)
    memory.add_memory_summary(
        1026813421,
        batch,
        summary="群里聊过小鸟和风雪的相处。",
        recall_cues=["小鸟", "风雪"],
    )
    summary_id = memory.recent_memory_summaries(1026813421, 3)[0].id

    edit_html = render_admin_edit_page(
        editable_files=[
            {
                "key": "prompt",
                "label": "人格 / Prompt",
                "description": "保存后热重载",
            }
        ],
        selected_key="prompt",
        content="persona:\n  id: zhangfengxue\n",
    )
    list_html = render_memory_summaries_page(
        memory=memory,
        groups=(1026813421,),
        selected_group_id=1026813421,
        status="active",
        limit=20,
    )
    detail_html = render_memory_summary_detail_page(
        memory=memory,
        summary_id=summary_id,
        groups=(1026813421,),
    )

    assert "保存并校验" in edit_html
    assert "新增人工回想" in list_html
    assert "编辑并保留" in detail_html
    assert "群里聊过小鸟" in detail_html

def test_admin_tools_page_renders_controls_and_docs() -> None:
    state = {
        "groups": [{"group_id": 1026813421, "enabled": True, "persona": "zhangxuefeng", "muted_left_seconds": 0}],
        "approval": {
            "review_enabled": True,
            "mode": "人工审查",
            "auto_send_percent": 30,
            "pending_count": 0,
            "owners": [1535071184],
            "basic_users": [3370998238],
            "all_users": [1535071184, 3370998238],
        },
        "work_intensity": {"current_percent": 8, "base_percent": 8, "band": "day"},
        "private_chat": {
            "config_ids": [2776760548],
            "runtime_ids": [3115344487],
            "implicit_chat_ids": [1535071184, 2776760548],
            "command_only_ids": [],
            "force_obey_enabled": False,
        },
        "models": [
            {
                "route": "reply",
                "title": "回复",
                "flow": "生成回复",
                "active": "siliconflow/MiniMaxAI/MiniMax-M2.5",
                "configured": "siliconflow/MiniMaxAI/MiniMax-M2.5",
                "fallback": "deepseek/deepseek-v4-flash",
                "overridden": False,
            }
        ],
        "model_catalog": [
            {"label": "siliconflow/MiniMaxAI/MiniMax-M2.5", "source": "硅基流动"}
        ],
        "jargon_entries": [
            {"term": "咱妈", "explanation": "中国", "created_by": 1535071184, "created_at": 100}
        ],
        "tool_docs": {"目录": "工具目录"},
    }

    html = render_admin_tools_page(state=state, selected_group_id=1026813421, report_title="报告", report_text="内容")

    assert "工具控制台" in html
    assert "免审概率" in html
    assert "模型路由" in html
    assert "黑话词典" in html
    assert "RAG状态" in html
    assert "pending_count" in html and ">0<" in html
    assert "工具目录" in html
    assert "可普通私聊的管理员" in html
    assert "主人/测试号强服从" in html
    assert ">测试号强服从<" not in html
    assert "发送到群" in html
    assert "发送私聊" in html
    assert 'name="action" value="send_group"' in html
    assert 'name="action" value="send_private"' in html


def test_admin_dashboard_renders_runtime_status(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    html = render_admin_dashboard(
        memory=memory,
        groups=(1026813421,),
        selected_group_id=1026813421,
        ready={"ok": True, "onebot_ready": True},
        health={"ok": True},
        status={
            "onebot": {
                "connected_bots": ["1801507496"],
                "bots": [
                    {
                        "bot_id": "1801507496",
                        "connected": True,
                        "last_api_name": "send_group_msg",
                        "last_api_outcome": "success",
                        "last_seen_at": 1783872000,
                    }
                ],
            },
            "search": {
                "enabled": True,
                "provider": "searxng",
                "rate_remaining": 8,
                "counters": {"successes": 3, "no_results": 1, "failures": 0},
                "last_request": {
                    "status": "ok",
                    "provider": "searxng",
                    "query_preview": "今天天气",
                    "at": 1783872000,
                },
            },
            "rag": {
                "enabled": True,
                "mode": "hybrid",
                "last_sync_at": 1783872000,
                "last_error": "",
                "store": {
                    "documents": 12,
                    "retrievals_1h": 4,
                    "active_knowledge_sources": 2,
                    "last_retrieval": {
                        "route": "hybrid",
                        "injected_count": 3,
                        "query_preview": "小鸟",
                    },
                },
            },
            "last_message": {
                "id": 9,
                "group_id": 1026813421,
                "user_id": 1535071184,
                "nickname": "主人",
                "text": "风雪在吗",
                "is_bot": False,
                "created_at": 1783872000,
                "age_seconds": 12,
            },
            "buffers": {
                "group_buffers": {"1026813421": 2},
                "generation_inflight_groups": [1026813421],
            },
            "recent_errors": [
                {
                    "created_at": 1783872000,
                    "event_type": "llm_decision",
                    "stage": "decision",
                    "action": "timeout",
                    "metadata": {"error": "deadline exceeded"},
                }
            ],
            "recent_rejections": [
                {
                    "created_at": 1783872000,
                    "event_type": "approval_canceled",
                    "stage": "approval",
                    "action": "reject",
                    "metadata": {"reason": "owner rejected"},
                }
            ],
            "groups": [
                {
                    "group_id": 1026813421,
                    "group_name": "柏拉图学院",
                    "enabled": True,
                    "persona": "zhangfengxue",
                    "muted_left_seconds": 0,
                    "member_count": 91,
                }
            ],
        },
        model_routes={"reply": "deepseek/deepseek-v4-flash"},
        pending_approvals=[],
        plugins=[],
    )

    assert "OneBot" in html
    assert "1801507496" in html
    assert "搜索" in html
    assert "今天天气" in html
    assert "RAG" in html
    assert "小鸟" in html
    assert "最后消息" in html
    assert "风雪在吗" in html
    assert "运行缓冲" in html
    assert "最近错误" in html
    assert "deadline exceeded" in html
    assert "最近拦截/拒绝" in html
    assert "owner rejected" in html
    assert "柏拉图学院" in html



def test_admin_tools_does_not_offer_review_toggle() -> None:
    html = render_admin_tools_page(
        state={
            "groups": [{"group_id": 1026813421, "enabled": True, "persona": "zhangxuefeng", "muted_left_seconds": 0}],
            "approval": {"review_enabled": False, "mode": "免审直发", "auto_send_percent": 100, "pending_count": 0, "owners": [], "basic_users": [], "all_users": []},
            "work_intensity": {"current_percent": 8, "base_percent": 8, "band": "day"},
            "private_chat": {"config_ids": [], "runtime_ids": [], "implicit_chat_ids": [], "command_only_ids": [], "force_obey_enabled": False},
            "models": [],
            "model_catalog": [],
            "jargon_entries": [],
            "tool_docs": {},
        },
        selected_group_id=1026813421,
        report_title="",
        report_text="",
    )
    assert "永久关闭" in html
    assert 'value="review_enabled"' not in html


def test_summary_actions_are_post_forms(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.admin_add_memory_summary(group_id=1, summary="测试回想", recall_cues=["测试"], locked=False)
    html = render_memory_summaries_page(memory=memory, groups=(1,), selected_group_id=1, status="active", limit=20)
    assert 'method="post" action="/admin/summaries/action' in html
    assert 'href="/admin/summaries/action' not in html
