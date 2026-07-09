import asyncio
import time

import nonebot

nonebot.init()

import qq_social_agent.plugin as plugin
from qq_social_agent.plugin import (
    APPROVAL_CHOICE_RE,
    APPROVAL_DETAIL_COMMANDS,
    APPROVAL_HELP_COMMANDS,
    APPROVAL_REJECT_REASON_RE,
    GROUP_BUFFER_SECONDS,
    GROUP_PASSIVE_DECISION_EVERY_MESSAGES,
    GROUP_PASSIVE_DECISION_GAP_SECONDS,
    JARGON_ADD_RE,
    JARGON_DELETE_RE,
    JARGON_LIST_RE,
    group_passive_decision_state,
    last_user_reply_times,
    _extract_message_id,
    _format_member_context,
    _format_recall_feedback_context,
    _is_low_value_group_text,
    _member_label,
    _passive_decision_allowed,
    _record_user_reply,
    _user_reply_cooling_down,
)
from qq_social_agent.memory import MemoryStore, MemberProfile, RecalledReplyFeedback


class FakeApprovalBot:
    def __init__(self) -> None:
        self.private_messages: list[tuple[int, str]] = []
        self.group_messages: list[tuple[int, str]] = []

    async def send_private_msg(self, *, user_id: int, message: object) -> dict[str, int]:
        self.private_messages.append((user_id, str(message)))
        return {"message_id": 1000 + len(self.private_messages)}

    async def send_group_msg(self, *, group_id: int, message: object) -> dict[str, int]:
        self.group_messages.append((group_id, str(message)))
        return {"message_id": 2000 + len(self.group_messages)}


def _use_temp_plugin_memory(monkeypatch, tmp_path) -> MemoryStore:
    store = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr(plugin, "memory", store)
    plugin.pending_group_approvals.clear()
    plugin.last_group_mention_targets.clear()
    return store


def _pending_approval() -> plugin.PendingGroupApproval:
    return plugin.PendingGroupApproval(
        group_id=1026813421,
        trigger_user_id=184589072,
        trigger_nickname="小鸟",
        trigger_text="没人理我",
        persona_name="张风雪",
        self_id=1801507496,
        candidates=(
            plugin.PendingApprovalCandidate(1, "第一条回复", "tease", "短句吐槽"),
            plugin.PendingApprovalCandidate(2, "第二条回复", "agree", "温和认可"),
            plugin.PendingApprovalCandidate(3, "第三条回复", "answer", "正常回答"),
        ),
        mention_targets={},
        created_at=1000.0,
    )


def test_group_buffer_seconds_is_six() -> None:
    assert GROUP_BUFFER_SECONDS == 6.0


def test_low_value_group_text_ignored() -> None:
    for text in ["好的", "一般", "绷", "嗯", "6", "哈哈", "可以", "哈哈哈哈！！！"]:
        assert _is_low_value_group_text(text)


def test_approval_reject_reason_regex() -> None:
    match = APPROVAL_REJECT_REASON_RE.match("不准奏原因：回答太僵硬")
    assert match is not None
    assert match.group("index") is None
    assert match.group("reason") == "回答太僵硬"

    indexed_match = APPROVAL_REJECT_REASON_RE.match("不准奏2原因：没接住第二条")
    assert indexed_match is not None
    assert indexed_match.group("index") == "2"
    assert indexed_match.group("reason") == "没接住第二条"


def test_approval_choice_regex() -> None:
    assert APPROVAL_CHOICE_RE.match("1")
    assert APPROVAL_CHOICE_RE.match("2!")
    assert APPROVAL_CHOICE_RE.match("3！")
    assert not APPROVAL_CHOICE_RE.match("4")


def test_approval_help_commands() -> None:
    assert "审批规则" in APPROVAL_HELP_COMMANDS
    assert "审批规则详情" in APPROVAL_DETAIL_COMMANDS
    assert "详细规则" in APPROVAL_DETAIL_COMMANDS
    assert "token用量" in plugin.APPROVAL_RULES_MESSAGE
    assert "token用量 1h/7d/all" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "token用量 2026-07-10" in plugin.APPROVAL_RULES_DETAIL_MESSAGE


def test_parse_token_report_date_window() -> None:
    window = plugin._parse_token_report_window("2026-07-10")

    assert window.label == "2026-07-10"
    assert window.start_at == time.mktime(time.strptime("2026-07-10", "%Y-%m-%d"))
    assert window.end_at == window.start_at + 24 * 60 * 60


def test_parse_token_report_slash_date_window() -> None:
    window = plugin._parse_token_report_window("2026/7/10")

    assert window.label == "2026-07-10"


def test_jargon_command_regexes() -> None:
    add = JARGON_ADD_RE.match("/黑话：咱妈 指代：中国")
    assert add is not None
    assert add.group("term") == "咱妈"
    assert add.group("meaning") == "中国"
    assert JARGON_LIST_RE.match("/黑话列表")
    delete = JARGON_DELETE_RE.match("/删黑话：咱妈")
    assert delete is not None
    assert delete.group("term") == "咱妈"


def test_passive_decision_gate_idle_then_every_three_messages() -> None:
    group_passive_decision_state.clear()

    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1000.0,
        last_message_at=1000.0,
    )
    assert allowed
    assert reason == "gap_first_message"

    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1005.0,
        last_message_at=1005.0,
    )
    assert not allowed
    assert reason == "waiting_1/3"

    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1010.0,
        last_message_at=1010.0,
    )
    assert not allowed
    assert reason == "waiting_2/3"

    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1015.0,
        last_message_at=1015.0,
    )
    assert allowed
    assert reason == "every_three_messages"


def test_passive_decision_gate_resets_after_idle_window() -> None:
    group_passive_decision_state.clear()

    _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1000.0,
        last_message_at=1000.0,
    )
    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1000.0 + GROUP_PASSIVE_DECISION_GAP_SECONDS,
        last_message_at=1000.0 + GROUP_PASSIVE_DECISION_GAP_SECONDS,
    )

    assert allowed
    assert reason == "gap_first_message"
    assert GROUP_PASSIVE_DECISION_EVERY_MESSAGES == 3


def test_passive_decision_gate_allows_after_thirty_second_gap() -> None:
    group_passive_decision_state.clear()

    _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1000.0,
        last_message_at=1000.0,
    )
    allowed, reason = _passive_decision_allowed(
        1026813421,
        message_count=1,
        first_message_at=1031.0,
        last_message_at=1031.0,
    )

    assert allowed
    assert reason == "gap_first_message"
    assert GROUP_PASSIVE_DECISION_GAP_SECONDS == 30


def test_meaningful_group_text_not_low_value() -> None:
    for text in ["股票又亏了", "你用什么刮胡子", "可以去投算法岗", "哈哈这项目真离谱"]:
        assert not _is_low_value_group_text(text)


def test_specific_user_reply_cooldown() -> None:
    last_user_reply_times.clear()

    _record_user_reply(1026813421, 3370998238, now=1000.0)

    assert not _user_reply_cooling_down(1026813421, 3370998238, now=1119.0)
    assert not _user_reply_cooling_down(1026813421, 3370998238, now=1120.0)
    assert not _user_reply_cooling_down(1026813421, 1535071184, now=1119.0)


def test_member_label_uses_qq_tail() -> None:
    assert _member_label(3370998238, "乌木") == "乌木[#98238]"


def test_format_member_context_includes_aliases() -> None:
    context = _format_member_context(
        [
            MemberProfile(
                group_id=1026813421,
                user_id=3370998238,
                display_name="🦕",
                aliases=("🦕", "乌木", "旧名"),
                last_seen_at=1000.0,
            )
        ]
    )

    assert context == "- 🦕[#98238]，曾用名/历史名：乌木、旧名"


def test_extract_message_id_from_action_result() -> None:
    assert _extract_message_id({"message_id": "123"}) == 123
    assert _extract_message_id({}) is None


def test_format_recall_feedback_context() -> None:
    context = _format_recall_feedback_context(
        [
            RecalledReplyFeedback(
                group_id=1026813421,
                message_id=42,
                bot_reply="僵硬回复",
                trigger_user_id=100,
                trigger_nickname="群友",
                trigger_text="你怎么看",
                action="answer",
                owner_reason="回答僵硬",
                scene_summary="群友在问普通问题",
                bad_reply_problem="回答像模板",
                avoid_rule="不要模板化回答",
                better_direction="先短句判断",
                tags=("僵硬", "模板"),
                operator_id=1535071184,
                reason_user_id=1535071184,
                recalled_at=1000.0,
                reason_at=1010.0,
            )
        ]
    )

    assert "主人撤回反馈" not in context
    assert "场景：群友在问普通问题" in context
    assert "避免：不要模板化回答" in context


def test_format_recall_feedback_context_owner_feedback_raw() -> None:
    context = _format_recall_feedback_context(
        [
            RecalledReplyFeedback(
                group_id=1026813421,
                message_id=42,
                bot_reply="僵硬回复",
                trigger_user_id=100,
                trigger_nickname="群友",
                trigger_text="你怎么看",
                action="answer",
                owner_reason="只保存我给的评价",
                scene_summary="主人原始评价",
                bad_reply_problem="只保存我给的评价",
                avoid_rule="只保存我给的评价",
                better_direction="只保存我给的评价",
                tags=("owner_feedback",),
                operator_id=1535071184,
                reason_user_id=1535071184,
                recalled_at=1000.0,
                reason_at=1010.0,
            )
        ]
    )

    assert context == "- 主人原始评价：只保存我给的评价"


def test_approval_detail_command_does_not_consume_pending(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "审批规则详情"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "张风雪群发审批规则详情" in bot.private_messages[-1][1]


def test_approval_token_report_command_does_not_consume_pending(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        created_at=1000.0,
    )
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "token用量 all"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "Token 用量报告（全部）" in bot.private_messages[-1][1]
    assert "decision / deepseek-v4-flash" in bot.private_messages[-1][1]


def test_approval_token_report_date_command(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    target_start = time.mktime(time.strptime("2026-07-10", "%Y-%m-%d"))
    store.add_llm_usage(
        task="old",
        model="deepseek-v4-flash",
        prompt_tokens=9000,
        completion_tokens=900,
        total_tokens=9900,
        created_at=target_start - 1,
    )
    store.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        created_at=target_start + 60,
    )
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "token用量 2026-07-10"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "Token 用量报告（2026-07-10）" in bot.private_messages[-1][1]
    assert "decision / deepseek-v4-flash" in bot.private_messages[-1][1]
    assert "old / deepseek-v4-flash" not in bot.private_messages[-1][1]


def test_approval_close_clears_pending_and_resends_rules(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "关闭"))

    assert handled
    assert plugin.pending_group_approvals == {}
    assert store.group_state(1026813421)["enabled"] is False
    rule_messages = [item for item in bot.private_messages if item[1] == plugin.APPROVAL_RULES_MESSAGE]
    assert {user_id for user_id, _ in rule_messages} == set(plugin.GROUP_APPROVAL_USER_IDS)


def test_approval_open_restores_decision_and_resends_rules(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.set_group_enabled(1026813421, False)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "开启"))

    assert handled
    assert store.group_state(1026813421)["enabled"] is True
    rule_messages = [item for item in bot.private_messages if item[1] == plugin.APPROVAL_RULES_MESSAGE]
    assert {user_id for user_id, _ in rule_messages} == set(plugin.GROUP_APPROVAL_USER_IDS)


def test_approval_reject_second_candidate_records_that_candidate(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(bot, 3370998238, "不准奏2原因：应该写对群宠小鸟不够温柔")
    )

    assert handled
    assert not bot.group_messages
    feedback = store.recent_recalled_reply_feedback(1026813421, 3)
    assert len(feedback) == 1
    assert feedback[0].bot_reply == "第二条回复"
    assert feedback[0].scene_summary == "审批不准奏原始评价，针对第 2 条候选"
    assert feedback[0].owner_reason == "应该写对群宠小鸟不够温柔"


def test_approval_high_quality_choice_sends_and_records_positive(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "1!"))

    assert handled
    assert bot.group_messages == [(1026813421, "第一条回复")]
    approved = store.recent_approved_reply_feedback(1026813421, 3)
    assert len(approved) == 1
    assert approved[0].candidate_text == "第一条回复"
    assert approved[0].operator_id == 3370998238


def test_matched_custom_jargon_entries_only_returns_current_hits(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.upsert_custom_jargon(group_id=1026813421, term="达斯", explanation="指代：打死", created_by=3370998238)
    store.upsert_custom_jargon(group_id=1026813421, term="火宅", explanation="指代：活摘", created_by=3370998238)

    entries = plugin._matched_custom_group_jargon_entries(1026813421, ["这波达斯了"])

    assert len(entries) == 1
    assert entries[0].terms == ("达斯",)


def test_format_token_usage_report(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        created_at=1000.0,
    )
    store.add_llm_usage(
        task="reply_candidates",
        model="deepseek-v4-flash",
        prompt_tokens=2000,
        completion_tokens=300,
        total_tokens=2300,
        created_at=1010.0,
    )

    report = plugin._format_token_usage_report(
        summaries=store.llm_usage_summary(),
        recent_events=store.recent_llm_usage_events(limit=2),
        label="全部",
    )

    assert "Token 用量报告（全部）" in report
    assert "总调用：2 次" in report
    assert "decision / deepseek-v4-flash" in report
    assert "reply_candidates / deepseek-v4-flash" in report
    assert "估算成本" in report
