from qq_social_agent.admin_ui import (
    render_admin_edit_page,
    render_admin_tools_page,
    render_memory_atom_detail_page,
    render_memory_audit_page,
    render_memory_summaries_page,
    render_memory_summary_detail_page,
    render_message_detail_page,
)
from qq_social_agent.memory import MemoryStore


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
            "command_only_ids": [1535071184],
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

