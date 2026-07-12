import asyncio
import time
from types import SimpleNamespace

import nonebot

nonebot.init()

import qq_social_agent.plugin as plugin
from qq_social_agent.plugin import (
    APPROVAL_CHOICE_RE,
    APPROVAL_DETAIL_COMMANDS,
    APPROVAL_HELP_COMMANDS,
    APPROVAL_REJECT_REASON_RE,
    GROUP_BUFFER_SECONDS,
    GROUP_INFLIGHT_BUFFER_RETRY_SECONDS,
    GROUP_PASSIVE_DECISION_EVERY_MESSAGES,
    GROUP_PASSIVE_DECISION_GAP_SECONDS,
    JARGON_ADD_RE,
    JARGON_DELETE_RE,
    JARGON_LIST_RE,
    group_passive_decision_state,
    group_generation_inflight,
    last_user_reply_times,
    _extract_message_id,
    _format_memory_context,
    _format_member_context,
    _format_member_impression_report,
    _format_raw_corpus_context,
    _format_recall_feedback_context,
    _daily_review_window,
    _apply_backend_tool_decision,
    _compact_long_message_fallback,
    _is_explicit_market_lookup,
    _is_low_value_group_text,
    _is_useful_style_rule,
    _member_label,
    _passive_decision_allowed,
    _pre_decision_gate,
    _record_user_reply,
    _sanitize_generated_text,
    _style_learning_messages_with_focus,
    _user_reply_cooling_down,
)
from qq_social_agent.cue_patterns import CueRepeatState
from qq_social_agent.config import parse_llm_model_route
from qq_social_agent.deepseek_client import ReplyDecision
from qq_social_agent.memory import ChatMessage, MemoryStore, MemorySummary, MemberImpression, MemberProfile, RawCorpusExample, RecalledReplyFeedback
from qq_social_agent.tools.fresh_context import FreshIntent
from qq_social_agent.tools.market_intent import MarketIntent


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


class FakeModelClient:
    def __init__(self) -> None:
        self.config = plugin.app_config.deepseek
        self.route_overrides = {}

    def parse_model_route(self, value: str, *, default_provider: str = "siliconflow"):
        return parse_llm_model_route(value, self.config.providers, default_provider=default_provider)

    def set_route_override(self, route_name: str, route) -> None:
        if route is None:
            self.route_overrides.pop(route_name, None)
        else:
            self.route_overrides[route_name] = route

    def current_route(self, route_name: str):
        return self.route_overrides.get(route_name, self.config.routes[route_name])


def _use_temp_plugin_memory(monkeypatch, tmp_path) -> MemoryStore:
    store = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr(plugin, "memory", store)
    monkeypatch.setattr(plugin, "deepseek_client", None)
    plugin.pending_group_approvals.clear()
    plugin.last_group_mention_targets.clear()
    plugin.recent_suppression_events.clear()
    plugin.approval_choice_cooldowns.clear()
    plugin.group_message_buffers.clear()
    plugin.group_buffer_tasks.clear()
    plugin.group_generation_inflight.clear()
    return store


def _pending_approval() -> plugin.PendingGroupApproval:
    return plugin.PendingGroupApproval(
        approval_id="test-approval",
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


def test_pad_approval_candidates_fills_to_three() -> None:
    candidates = [
        plugin.PendingApprovalCandidate(
            index=1,
            text="走错了都能吃上也挺离谱的",
            action="tease",
            style="模型原始候选",
        )
    ]

    plugin._pad_approval_candidates(candidates, action="tease", limit=3)

    assert len(candidates) == 3
    assert [candidate.index for candidate in candidates] == [1, 2, 3]
    assert candidates[1].style.startswith("后端补齐")


def test_group_buffer_seconds_is_six() -> None:
    assert GROUP_BUFFER_SECONDS == 6.0


def test_compact_long_message_fallback_keeps_short_marker() -> None:
    text = "这是一条很长的消息" * 12

    compacted = _compact_long_message_fallback(text)

    assert len(compacted) < len(text)
    assert "长消息" in compacted
    assert "已省略" in compacted


def test_short_reply_context_does_not_compact_only_because_of_labels() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "你今天是不是刚到自贡荣县，耍起了吗")
    event = SimpleNamespace(
        user_id=2849687751,
        sender=SimpleNamespace(card="🐖linbar🐖（旅游中）", nickname=""),
        reply=SimpleNamespace(
            user_id=1660502091,
            sender=SimpleNamespace(card="entp大人", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "我今天刚到"}),
        ],
    )
    raw_text = plugin._message_context_text(event, bot_id=1801507496)

    assert len(raw_text) > plugin.LONG_MESSAGE_SUMMARY_THRESHOLD
    assert len(raw_text) <= plugin.REPLY_CONTEXT_SUMMARY_THRESHOLD
    assert not plugin._should_compact_group_context_message(event, raw_text=raw_text, plain_text="我今天刚到")


def test_long_plain_reply_context_still_compacts() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "原话")
    plain_text = "这是真正很长的当前回复" * 12
    event = SimpleNamespace(
        user_id=2849687751,
        sender=SimpleNamespace(card="🐖linbar🐖（旅游中）", nickname=""),
        reply=SimpleNamespace(
            user_id=1660502091,
            sender=SimpleNamespace(card="entp大人", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": plain_text}),
        ],
    )
    raw_text = plugin._message_context_text(event, bot_id=1801507496)

    assert plugin._should_compact_group_context_message(event, raw_text=raw_text, plain_text=plain_text)


def test_style_learning_messages_with_focus_adds_xiaoniao(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_message(1026813421, 100, "A", "普通群友一句话", created_at=100)
    store.add_message(1026813421, 184589072, "小鸟", "小鸟的表达应该额外混入", created_at=120)
    base = [
        ChatMessage(1026813421, 100, "A", "普通群友一句话", False, 100.0, id=1),
    ]

    boosted = _style_learning_messages_with_focus(1026813421, base, now=180)

    assert [message.user_id for message in boosted] == [100, 184589072]
    assert boosted[-1].text == "小鸟的表达应该额外混入"


def _buffered_item(group_id: int, text: str, *, user_id: int = 184589072) -> plugin.BufferedGroupMessage:
    return plugin.BufferedGroupMessage(
        bot=SimpleNamespace(),
        event=SimpleNamespace(group_id=group_id, user_id=user_id),
        text=text,
        user_id=user_id,
        nickname="群友",
        created_at=1000.0,
    )


def test_group_buffer_flush_defers_while_generation_inflight(monkeypatch) -> None:
    group_id = 1026813421
    plugin.group_message_buffers.clear()
    plugin.group_buffer_tasks.clear()
    group_generation_inflight.clear()
    item = _buffered_item(group_id, "后来的消息")
    plugin.group_message_buffers[group_id] = [item]
    group_generation_inflight.add(group_id)
    handled = []
    scheduled = []

    async def fake_handle(*args, **kwargs) -> None:
        handled.append((args, kwargs))

    monkeypatch.setattr(plugin, "_handle_group_message_locked", fake_handle)
    monkeypatch.setattr(
        plugin,
        "_schedule_group_buffer_flush",
        lambda gid, *, delay=GROUP_BUFFER_SECONDS: scheduled.append((gid, delay)),
    )

    asyncio.run(plugin._flush_group_buffer_after_delay(group_id, delay=0))

    assert handled == []
    assert plugin.group_message_buffers[group_id] == [item]
    assert scheduled == [(group_id, GROUP_INFLIGHT_BUFFER_RETRY_SECONDS)]
    group_generation_inflight.clear()


def test_group_buffer_flush_marks_generation_and_reschedules_pending(monkeypatch) -> None:
    group_id = 1026813421
    plugin.group_message_buffers.clear()
    plugin.group_buffer_tasks.clear()
    group_generation_inflight.clear()
    first_item = _buffered_item(group_id, "第一轮")
    second_item = _buffered_item(group_id, "生成中来的新消息", user_id=3370998238)
    plugin.group_message_buffers[group_id] = [first_item]
    inflight_seen = []
    scheduled = []

    async def fake_handle(*args, **kwargs) -> None:
        inflight_seen.append(group_id in group_generation_inflight)
        plugin.group_message_buffers.setdefault(group_id, []).append(second_item)

    monkeypatch.setattr(plugin, "_handle_group_message_locked", fake_handle)
    monkeypatch.setattr(
        plugin,
        "_schedule_group_buffer_flush",
        lambda gid, *, delay=GROUP_BUFFER_SECONDS: scheduled.append((gid, delay)),
    )

    asyncio.run(plugin._flush_group_buffer_after_delay(group_id, delay=0))

    assert inflight_seen == [True]
    assert group_id not in group_generation_inflight
    assert plugin.group_message_buffers[group_id] == [second_item]
    assert scheduled == [(group_id, GROUP_INFLIGHT_BUFFER_RETRY_SECONDS)]


def test_low_value_group_text_ignored() -> None:
    for text in [
        "好的",
        "一般",
        "可以",
        "绷",
        "蹦",
        "没绷住",
        "没绷住了",
        "绷不住",
        "绷不住了",
        "好好好",
        "嗯",
        "6",
        "哈哈",
        "哈哈哈哈！！！",
        "草",
    ]:
        assert _is_low_value_group_text(text)


def test_acknowledgement_inside_meaningful_text_not_hard_ignored() -> None:
    for text in ["可以去投算法岗", "一般这种项目都贵", "好的学校还是看平台"]:
        assert not _is_low_value_group_text(text)


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
    assert APPROVAL_CHOICE_RE.match("A")
    assert APPROVAL_CHOICE_RE.match("b!")
    assert APPROVAL_CHOICE_RE.match("C！")
    assert not APPROVAL_CHOICE_RE.match("4")


def test_approval_help_commands() -> None:
    assert "审批规则" in APPROVAL_HELP_COMMANDS
    assert "审批规则详情" in APPROVAL_DETAIL_COMMANDS
    assert "详细规则" in APPROVAL_DETAIL_COMMANDS
    assert "A/1" in plugin.APPROVAL_RULES_MESSAGE
    assert "T 打开工具单" in plugin.APPROVAL_RULES_MESSAGE
    assert "bot工具目录" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "bot工具 私聊" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "bot工具 学习" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "A1拦截" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "B3群友画像" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "D6审批概率" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert "D7工作强度" in plugin.APPROVAL_RULES_DETAIL_MESSAGE
    assert plugin._bot_tool_shortcut_command("D6") == "审批概率"
    assert plugin._bot_tool_shortcut_command("D7") == "工作强度"
    assert "/黑话：咱妈 指代：中国" in plugin._bot_tool_message("bot工具 黑话")
    assert "张风雪 bot工具目录" in plugin._bot_tool_message("T")
    assert "bot工具 查看" in plugin._bot_tool_message("A")
    assert "bot工具 学习" in plugin._bot_tool_message("B")
    assert "bot工具 模型" in plugin._bot_tool_message("C")
    assert "统计 今日" in plugin._bot_tool_message("bot工具 查看")
    assert "记忆单元 20" in plugin._bot_tool_message("bot工具 学习")
    assert "flows.member_profile" in plugin._bot_tool_message("bot工具 prompt")
    assert "拦截 20" in plugin._bot_tool_message("bot工具 查看")
    assert "记忆 8" in plugin._bot_tool_message("bot工具 学习")
    assert "风格学习 20" in plugin._bot_tool_message("bot工具 风格")
    assert "群友画像 20" in plugin._bot_tool_message("bot工具 学习")
    assert "加审批" in plugin._bot_tool_message("bot工具 审批人")
    assert "回 A/B/C" in plugin._bot_tool_message("bot 工具 审批")
    assert "关闭审查" in plugin.APPROVAL_RULES_MESSAGE
    assert "开启审查" in plugin._bot_tool_message("bot工具 开关")
    assert "工作强度 60" in plugin._bot_tool_message("bot工具 开关")
    assert "切回复模型" in plugin._bot_tool_message("bot工具 模型")
    assert "切风格模型" in plugin._bot_tool_message("bot工具 模型")
    assert "可切换部分" in plugin._bot_tool_message("bot工具 模型")
    assert "模型状态" in plugin._bot_tool_message("bot工具 模型")
    assert "模型名要完整复制" in plugin._bot_tool_message("bot工具 模型")
    assert "flows.decision" in plugin._bot_tool_message("bot工具 prompt")
    assert "加私聊 QQ号" in plugin._bot_tool_message("bot工具 私聊")
    assert "格式是 /黑话：词 指代：解释" in plugin._bot_tool_message("bot工具 黑话")
    assert "回 A/B/C" in plugin._bot_tool_message("bot工具 审批")
    assert "拦截 50" in plugin._bot_tool_message("bot工具 查看")
    assert plugin._bot_tool_shortcut_command("B3") == "群友画像 20"
    assert plugin._bot_tool_shortcut_command("A6") == "统计 今日"
    assert plugin._bot_tool_shortcut_command("B4") == "记忆单元 20"
    assert plugin._bot_tool_shortcut_command("A 1") == "拦截 20"


def test_message_context_text_includes_media_and_reply() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "原消息内容很长但是能被摘要")
    reply = SimpleNamespace(
        message=reply_message,
        sender=SimpleNamespace(card="小鸟", nickname=""),
        user_id=184589072,
        message_id=42,
    )
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=reply,
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "  你看这个  "}),
            SimpleNamespace(type="image", data={}),
            SimpleNamespace(type="json", data={"data": "聊天记录 forward"}),
        ],
    )

    text = plugin._message_context_text(event)

    assert "歌迷老蛆[#71184]回复小鸟[#89072]消息" in text
    assert "小鸟[#89072]说：原消息内容很长但是能被摘要" in text
    assert "歌迷老蛆[#71184]回复小鸟[#89072]：你看这个 [图片] [转发消息]" in text
    assert "你看这个" in text
    assert "[图片]" in text
    assert "[转发消息]" in text


def test_replied_to_bot_uses_event_reply_sender() -> None:
    bot = SimpleNamespace(self_id=1801507496)
    event = SimpleNamespace(
        message=[SimpleNamespace(type="reply", data={"id": "42"})],
        reply=SimpleNamespace(user_id=1801507496, sender=SimpleNamespace(card="张风雪")),
    )

    assert plugin._replied_to_bot(event, bot)


def test_mentioned_bot_recognizes_fengxue_alias() -> None:
    bot = SimpleNamespace(self_id=1801507496)
    event = SimpleNamespace(
        self_id=1801507496,
        to_me=False,
        message=[SimpleNamespace(type="text", data={"text": "风雪你看看"})],
        get_plaintext=lambda: "风雪你看看",
    )

    assert plugin._mentioned_bot(event, bot)


def test_reply_to_bot_context_marks_zhangfengxue_as_self() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "风雪觉得这个有点离谱")
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=SimpleNamespace(
            user_id=1801507496,
            sender=SimpleNamespace(card="张风雪", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "你自己也知道啊"}),
        ],
        get_plaintext=lambda: "你自己也知道啊",
    )

    text = plugin._message_context_text(event, bot_id=1801507496)

    assert "张风雪和风雪都是你自己" in text
    assert "群友回复张风雪/风雪，就是在回复你之前说的话" in text
    assert "歌迷老蛆[#71184]回复张风雪[#07496]消息" in text
    assert "张风雪[#07496]说：风雪觉得这个有点离谱" in text


def test_plain_mention_of_fengxue_marks_self_context() -> None:
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=None,
        message=[
            SimpleNamespace(type="text", data={"text": "风雪你怎么看这个"}),
        ],
        get_plaintext=lambda: "风雪你怎么看这个",
    )

    text = plugin._message_context_text(event, bot_id=1801507496)

    assert text.startswith("注：张风雪和风雪都是你自己")
    assert "风雪你怎么看这个" in text


def test_reply_to_other_that_mentions_fengxue_marks_self_context() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "这个选择怎么样")
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=SimpleNamespace(
            user_id=123456789,
            sender=SimpleNamespace(card="安钰与雨与余", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "问问风雪呗"}),
        ],
        get_plaintext=lambda: "问问风雪呗",
    )

    text = plugin._message_context_text(event, bot_id=1801507496)

    assert "张风雪和风雪都是你自己" in text
    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]" in text
    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]：问问风雪呗" in text


def test_low_value_reply_to_bot_event_ignores_plain_ack() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "好的"}),
        ],
        get_plaintext=lambda: "好的",
    )

    assert plugin._is_low_value_reply_to_bot_event(event)


def test_low_value_reply_to_bot_event_allows_meaningful_reply() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "好的，那你怎么看这个学校"}),
        ],
        get_plaintext=lambda: "好的，那你怎么看这个学校",
    )

    assert not plugin._is_low_value_reply_to_bot_event(event)


def test_low_value_reply_to_bot_event_allows_media_reply() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="image", data={"summary": "截图"}),
        ],
        get_plaintext=lambda: "",
    )

    assert not plugin._is_low_value_reply_to_bot_event(event)


def test_reply_to_other_short_answer_enters_structured_context() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "恩泽去南京了？")
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=SimpleNamespace(
            user_id=123456789,
            sender=SimpleNamespace(card="安钰与雨与余", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "南下了"}),
        ],
        get_plaintext=lambda: "南下了",
    )

    text = plugin._message_context_text(event)

    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]消息" in text
    assert "安钰与雨与余[#56789]说：恩泽去南京了？" in text
    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]：南下了" in text


def test_reply_to_other_unknown_original_uses_clear_fallback() -> None:
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=SimpleNamespace(
            user_id=123456789,
            sender=SimpleNamespace(card="安钰与雨与余", nickname=""),
            message=None,
            message_id=42,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="text", data={"text": "那你怎么看这个学校"}),
        ],
        get_plaintext=lambda: "那你怎么看这个学校",
    )

    text = plugin._message_context_text(event)

    assert "安钰与雨与余[#56789]原消息内容未知，消息ID：42" in text
    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]：那你怎么看这个学校" in text


def test_reply_to_other_media_stays_in_structured_reply_text() -> None:
    reply_message = SimpleNamespace(extract_plain_text=lambda: "你看看这个")
    event = SimpleNamespace(
        user_id=1535071184,
        sender=SimpleNamespace(card="歌迷老蛆", nickname=""),
        reply=SimpleNamespace(
            user_id=123456789,
            sender=SimpleNamespace(card="安钰与雨与余", nickname=""),
            message=reply_message,
        ),
        message=[
            SimpleNamespace(type="reply", data={"id": "42"}),
            SimpleNamespace(type="image", data={"summary": "截图"}),
        ],
        get_plaintext=lambda: "",
    )

    text = plugin._message_context_text(event)

    assert "歌迷老蛆[#71184]回复安钰与雨与余[#56789]：[图片:截图]" in text


def test_unreadable_image_only_event_should_not_enter_passive_buffer() -> None:
    event = SimpleNamespace(
        message=[SimpleNamespace(type="image", data={"summary": "截图"})],
        get_plaintext=lambda: "",
    )

    assert plugin._should_ignore_unreadable_media_event(event, forward_context="")


def test_weak_image_caption_should_not_enter_passive_buffer() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="text", data={"text": "看看这个"}),
            SimpleNamespace(type="image", data={"summary": "截图"}),
        ],
        get_plaintext=lambda: "看看这个",
    )

    assert plugin._should_ignore_unreadable_media_event(event, forward_context="")


def test_meaningful_image_caption_can_enter_passive_buffer() -> None:
    event = SimpleNamespace(
        message=[
            SimpleNamespace(type="text", data={"text": "这个图是雷军发布会截图，重点是价格太离谱"}),
            SimpleNamespace(type="image", data={"summary": "截图"}),
        ],
        get_plaintext=lambda: "这个图是雷军发布会截图，重点是价格太离谱",
    )

    assert not plugin._should_ignore_unreadable_media_event(event, forward_context="")


def test_forward_record_extraction_formats_sender_and_text() -> None:
    payload = {
        "messages": [
            {
                "sender": {"user_id": 123456514, "nickname": "血火"},
                "content": [{"type": "text", "data": {"text": "这个学校不太值"}}],
            },
            {
                "sender": {"user_id": 184589072, "nickname": "小鸟"},
                "content": [{"type": "image", "data": {"summary": "截图"}}],
            },
        ]
    }

    lines = plugin._extract_forward_record_lines(payload, limit=5)

    assert lines[0] == "血火[#56514]: 这个学校不太值"
    assert lines[1] == "小鸟[#89072]: [图片:截图]"


def test_forward_context_text_uses_get_forward_msg(monkeypatch) -> None:
    class FakeForwardBot:
        async def call_api(self, api: str, **data):
            assert api == "get_forward_msg"
            assert data == {"id": "forward-1"}
            return {
                "messages": [
                    {
                        "sender": {"user_id": 123456514, "nickname": "血火"},
                        "content": [{"type": "text", "data": {"text": "这个学校不太值"}}],
                    }
                ]
            }

    monkeypatch.setattr(plugin, "deepseek_client", None)
    event = SimpleNamespace(
        message=[SimpleNamespace(type="forward", data={"id": "forward-1"})],
    )

    context = asyncio.run(plugin._forward_context_text(FakeForwardBot(), event, nickname="血火"))

    assert context == "血火传了聊天记录，大致内容如下：血火[#56514]: 这个学校不太值"


def test_parse_token_report_date_window() -> None:
    window = plugin._parse_token_report_window("2026-07-10")

    assert window.label == "2026-07-10"
    assert window.start_at == time.mktime(time.strptime("2026-07-10", "%Y-%m-%d"))
    assert window.end_at == window.start_at + 24 * 60 * 60


def test_parse_token_report_slash_date_window() -> None:
    window = plugin._parse_token_report_window("2026/7/10")

    assert window.label == "2026-07-10"


def test_parse_llm_usage_log_line() -> None:
    parsed = plugin._parse_llm_usage_log_line(
        "07-10 00:36:48 [INFO] qq_social_agent | qq_social_agent llm usage: "
        "task=decision model=deepseek-v4-flash prompt_tokens=3823 "
        "completion_tokens=88 total_tokens=3911",
        year=2026,
    )

    assert parsed is not None
    task, model, prompt_tokens, completion_tokens, total_tokens, created_at = parsed
    assert task == "decision"
    assert model == "deepseek-v4-flash"
    assert prompt_tokens == 3823
    assert completion_tokens == 88
    assert total_tokens == 3911
    assert created_at == time.mktime(time.strptime("2026-07-10 00:36:48", "%Y-%m-%d %H:%M:%S"))


def test_backfill_llm_usage_from_logs_deduplicates(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    log_path = tmp_path / "bot-runtime.log"
    log_path.write_text(
        "07-10 00:36:48 [INFO] qq_social_agent | qq_social_agent llm usage: "
        "task=decision model=deepseek-v4-flash prompt_tokens=3823 "
        "completion_tokens=88 total_tokens=3911\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin, "TOKEN_USAGE_LOG_BACKFILL_FILES", (log_path,))
    monkeypatch.setattr(plugin.time, "localtime", lambda *args: time.struct_time((2026, 7, 10, 0, 0, 0, 4, 191, 0)))

    assert plugin._backfill_llm_usage_from_logs() == 1
    assert plugin._backfill_llm_usage_from_logs() == 0

    summaries = store.llm_usage_summary()
    assert len(summaries) == 1
    assert summaries[0].task == "decision"
    assert summaries[0].total_tokens == 3911


def test_jargon_command_regexes() -> None:
    add = JARGON_ADD_RE.match("/黑话：咱妈 指代：中国")
    assert add is not None
    assert add.group("term") == "咱妈"
    assert add.group("meaning") == "中国"
    flexible_add = JARGON_ADD_RE.match("/黑话 达斯=打死")
    assert flexible_add is not None
    assert flexible_add.group("term") == "达斯"
    assert flexible_add.group("meaning") == "打死"
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
    assert reason == "first_decision"

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
    assert reason == "gap_since_decision"
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
    assert reason == "gap_since_decision"
    assert GROUP_PASSIVE_DECISION_GAP_SECONDS == 30


def test_daily_review_window_uses_previous_24h_at_midnight() -> None:
    midnight = time.mktime((2026, 7, 11, 0, 0, 0, -1, -1, -1))

    start_at, end_at, label = _daily_review_window(midnight + 1)

    assert start_at == midnight - 24 * 60 * 60
    assert end_at == midnight
    assert label == "2026-07-10"


def test_meaningful_group_text_not_low_value() -> None:
    for text in ["股票又亏了", "你用什么刮胡子", "可以去投算法岗", "哈哈这项目真离谱"]:
        assert not _is_low_value_group_text(text)


def test_format_raw_corpus_context_includes_original_warning_and_neighbors() -> None:
    example = RawCorpusExample(
        message=ChatMessage(1, 101, "B", "股票又亏麻了，真的顶不住", False, 101.0, id=2),
        before=(ChatMessage(1, 100, "A", "今天午饭吃什么", False, 100.0, id=1),),
        after=(ChatMessage(1, 102, "C", "这就是资本市场教育费", False, 102.0, id=3),),
        tags=("倒霉", "行情"),
        score=6,
    )

    context = _format_raw_corpus_context([example])

    assert "禁止复制完整原句" in context
    assert "B[#101]" in context
    assert "股票又亏麻了" in context
    assert "A[#100]" in context
    assert "C[#102]" in context


def test_pre_decision_gate_allows_weak_passive_text_to_llm() -> None:
    result = _pre_decision_gate(
        text="收到回复",
        recent_messages=[],
        persona=plugin.personas.get(plugin.app_config.default_persona),
        addressed_bot=False,
        mentioned=False,
        replied_to_bot=False,
        cue_repeat_state=None,
        market_intents=[],
        fresh_intent=None,
    )

    assert result.decision is None
    assert result.skip_reason == ""


def test_pre_decision_gate_skips_plain_ack_as_low_value() -> None:
    result = _pre_decision_gate(
        text="没绷住",
        recent_messages=[],
        persona=plugin.personas.get(plugin.app_config.default_persona),
        addressed_bot=False,
        mentioned=False,
        replied_to_bot=False,
        cue_repeat_state=None,
        market_intents=[],
        fresh_intent=None,
    )

    assert result.decision is None
    assert result.skip_reason == "low_value_local"


def test_pre_decision_gate_handles_explicit_market_lookup_locally() -> None:
    result = _pre_decision_gate(
        text="BTC 怎么了",
        recent_messages=[],
        persona=plugin.personas.get(plugin.app_config.default_persona),
        addressed_bot=False,
        mentioned=False,
        replied_to_bot=False,
        cue_repeat_state=None,
        market_intents=[MarketIntent("crypto", "bitcoin", "BTC")],
        fresh_intent=None,
    )

    assert _is_explicit_market_lookup("BTC 怎么了")
    assert _is_explicit_market_lookup("NVDA 咋样")
    assert result.skip_reason == ""
    assert result.decision is not None
    assert result.decision.action == "market_check"
    assert result.decision.need_tool
    assert result.decision.symbols[0].display == "BTC"


def test_backend_tool_decision_overrides_addressed_market_reply() -> None:
    decision = _apply_backend_tool_decision(
        ReplyDecision(True, 0.7, "正常回答", mode="chat", action="answer"),
        text="NVDA 今天咋样",
        market_intents=[MarketIntent("stock", "NVDA", "NVDA")],
        fresh_intent=None,
    )

    assert decision.action == "market_check"
    assert decision.need_tool
    assert decision.tool == "market"
    assert decision.comment_after_tool
    assert decision.symbols[0].symbol == "NVDA"


def test_fresh_lookup_goes_to_llm_decision() -> None:
    result = _pre_decision_gate(
        text="美国和伊朗现在怎么了",
        recent_messages=[],
        persona=plugin.personas.get(plugin.app_config.default_persona),
        addressed_bot=False,
        mentioned=False,
        replied_to_bot=False,
        cue_repeat_state=None,
        market_intents=[],
        fresh_intent=FreshIntent(query="美国和伊朗", kind="news"),
    )

    assert result.skip_reason == ""
    assert result.decision is None


def test_backend_tool_decision_forces_explicit_search_without_clobbering_action() -> None:
    decision = _apply_backend_tool_decision(
        ReplyDecision(True, 0.78, "正常回答", mode="chat", action="answer"),
        text="搜一下 NoneBot 插件文档",
        market_intents=[],
        fresh_intent=FreshIntent(query="NoneBot 插件文档", kind="web", explicit=True),
    )

    assert decision.action == "answer"
    assert decision.need_fresh_context
    assert decision.fresh_query == "NoneBot 插件文档"
    assert decision.fresh_kind == "web"


def test_backend_tool_decision_does_not_force_implicit_fresh_hint() -> None:
    original = ReplyDecision(True, 0.68, "顺手接话", mode="chat", action="tease")

    decision = _apply_backend_tool_decision(
        original,
        text="美国现在怎么了",
        market_intents=[],
        fresh_intent=FreshIntent(query="美国", kind="news", explicit=False),
    )

    assert decision == original


def test_pre_decision_gate_handles_repeated_addressed_cue_locally() -> None:
    result = _pre_decision_gate(
        text="c 罗和梅西谁厉害",
        recent_messages=[],
        persona=plugin.personas.get(plugin.app_config.default_persona),
        addressed_bot=True,
        mentioned=True,
        replied_to_bot=False,
        cue_repeat_state=CueRepeatState("comparison", "连续问谁厉害/谁更强", 3),
        market_intents=[],
        fresh_intent=None,
    )

    assert result.decision is not None
    assert result.decision.action == "mock_repeated_question"


def test_style_rule_filter_rejects_literal_examples() -> None:
    assert not _is_useful_style_rule("自嘲场景", "说“我完蛋了”", "我完蛋了")
    assert not _is_useful_style_rule("对离谱事吐槽", "短句接“太典了”", "太典了")
    assert not _is_useful_style_rule("模仿对方说话", "重复对方原句", "你说啥")
    assert not _is_useful_style_rule("表示赞赏或附和", "发👍👍👍", "👍👍👍")
    assert _is_useful_style_rule("拒绝请求", "用一句损友式现实理由拒绝", "不能，你胖的不差这点")


def test_specific_user_reply_cooldown() -> None:
    last_user_reply_times.clear()

    _record_user_reply(1026813421, 3370998238, now=1000.0)

    assert not _user_reply_cooling_down(1026813421, 3370998238, now=1119.0)
    assert not _user_reply_cooling_down(1026813421, 3370998238, now=1120.0)
    assert not _user_reply_cooling_down(1026813421, 1535071184, now=1119.0)


def test_member_label_uses_qq_tail() -> None:
    assert _member_label(3370998238, "乌木") == "乌木[#98238]"


def test_focused_user_tone_context_only_for_xiaoniao() -> None:
    assert "必须超级温柔" in plugin._focused_user_tone_context(184589072)
    assert plugin._focused_user_tone_context(3370998238) == ""


def test_builtin_memory_atoms_include_owner_alias(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)

    plugin._ensure_builtin_memory_atoms()

    atoms = store.relevant_memory_atoms(
        1026813421,
        "xbw 奈亚子 张风雪制造者",
        subject_user_ids=[1535071184],
        limit=5,
    )
    assert any("xbw、歌迷老蛆、奈亚子都是同一个人" in atom.content for atom in atoms)


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


def test_format_member_context_includes_impression_details() -> None:
    context = _format_member_context(
        [
            MemberImpression(
                group_id=1026813421,
                user_id=3370998238,
                display_name="乌木",
                aliases=("乌木", "🦕"),
                message_count=12,
                top_tags=(("行情", 3), ("代码", 2)),
                top_keywords=(("比特币", 2), ("代码", 1)),
                recent_texts=("股票又亏麻了",),
                ai_summary="经常聊行情和代码，亏钱时情绪很直接。",
                ai_interests=("股票", "比特币"),
                ai_speaking_style="短句吐槽，喜欢直接破防。",
                ai_representative_texts=("股票又亏麻了",),
                ai_summary_at=1000.0,
                last_seen_at=1000.0,
                updated_at=1000.0,
            )
        ]
    )

    assert "长期印象" in context
    assert "股票、比特币" in context
    assert "后端标签：行情x3、代码x2" in context
    assert "股票又亏麻了" in context


def test_format_member_impression_report(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_message(1026813421, 3370998238, "乌木", "股票又亏麻了", created_at=100)

    report = _format_member_impression_report(1026813421, 5)

    assert "群友画像" in report
    assert "乌木[#98238]" in report
    assert "后端标签" in report


def test_sanitize_generated_text_removes_emoji_and_onebot_face() -> None:
    assert _sanitize_generated_text("可以[CQ:face,id=14] 😂 继续说") == "可以 继续说"


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


def test_format_memory_context_warns_against_wrong_attribution() -> None:
    context = _format_memory_context(
        [
            MemorySummary(
                group_id=1026813421,
                summary="按人：灰機haru[#98238]：经常 cue 机器人；多人讨论：柏拉图活跃度。",
                recall_cues=("灰機haru[#98238] cue 机器人", "柏拉图活跃度",),
                start_at=1000.0,
                end_at=1100.0,
                created_at=1200.0,
            )
        ]
    )

    assert "不要把旧回想误认为当前发言人说过的话" in context
    assert "只有回想里明确写了昵称/QQ尾号" in context
    assert "灰機haru[#98238]" in context


def test_approval_detail_command_does_not_consume_pending(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "审批规则详情"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "张风雪 bot工具目录" in bot.private_messages[-1][1]


def test_approval_letter_choice_takes_priority_over_tool_menu(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "A"))

    assert handled
    assert bot.group_messages == [(1026813421, "第一条回复")]
    assert not bot.private_messages or "bot工具 查看" not in bot.private_messages[-1][1]


def test_approval_letter_cancel_takes_priority_over_tool_menu(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "D"))

    assert handled
    assert not bot.group_messages
    assert bot.private_messages[-1] == (3370998238, "已取消。")
    assert approval.group_id not in plugin.pending_group_approvals


def test_tool_letter_menu_when_no_pending_approval(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "A"))

    assert handled
    assert "bot工具 查看" in bot.private_messages[-1][1]


def test_tool_shortcut_member_profile_report(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "B3"))

    assert handled
    assert "群友画像：group=1026813421 limit=20" in bot.private_messages[-1][1]


def test_owner_can_query_metrics_and_memory_atoms(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_metric_event(
        event_type="decision_result",
        group_id=1026813421,
        user_id=184589072,
        stage="llm",
        action="echo_mood",
        metadata={"reason": "接情绪"},
        created_at=time.time(),
    )
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "A6"))

    assert handled
    assert "Bot 统计" in bot.private_messages[-1][1]
    assert "decision_result" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "加记忆：小鸟说话要更温柔一点"))
    assert handled
    assert "已写入记忆单元" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "B4"))
    assert handled
    assert "记忆单元：group=1026813421 limit=20" in bot.private_messages[-1][1]
    assert "小鸟说话要更温柔一点" in bot.private_messages[-1][1]


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

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "token用量 all"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "Token 用量统计：已关闭" in bot.private_messages[-1][1]


def test_basic_approver_cannot_use_token_tool(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "token用量 all"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


def test_owner_can_query_recent_memory_and_style(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.add_message(1026813421, 184589072, "小鸟", "股票又亏了", created_at=1000.0)
    store.add_message(1026813421, 3370998238, "乌木", "这成本太高", created_at=1010.0)
    batch = store.recent_messages(1026813421, 2)
    store.add_memory_summary(
        1026813421,
        batch,
        summary="按人：小鸟[#89072]说股票又亏了；乌木[#98238]说成本太高。",
        recall_cues=["小鸟股票亏损", "乌木成本判断"],
    )
    store.add_style_rules(
        1026813421,
        [("聊亏钱", "先短句吐槽，再补现实成本", "股票又亏了")],
    )
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "记忆 5"))

    assert handled
    assert "近期记忆：group=1026813421 limit=5" in bot.private_messages[-1][1]
    assert "小鸟" in bot.private_messages[-1][1]
    assert "#89072" in bot.private_messages[-1][1]
    assert "乌木成本判断" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "风格 5"))

    assert handled
    assert "近期风格学习：group=1026813421 limit=5" in bot.private_messages[-1][1]
    assert "当聊亏钱时，可以先短句吐槽" in bot.private_messages[-1][1]
    assert "股票又亏了" in bot.private_messages[-1][1]


def test_basic_approver_cannot_query_recent_memory(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "记忆 5"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


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

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "token用量 2026-07-10"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    assert "Token 用量统计：已关闭" in bot.private_messages[-1][1]
    assert "old / deepseek-v4-flash" not in bot.private_messages[-1][1]


def test_approval_close_clears_pending_and_resends_rules(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "关闭"))

    assert handled
    assert plugin.pending_group_approvals == {}
    assert store.group_state(1026813421)["enabled"] is False
    rule_messages = [item for item in bot.private_messages if item[1] == plugin.APPROVAL_RULES_MESSAGE]
    assert {user_id for user_id, _ in rule_messages} == set(plugin._approval_user_ids())


def test_approval_open_restores_decision_and_resends_rules(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.set_group_enabled(1026813421, False)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "开启"))

    assert handled
    assert store.group_state(1026813421)["enabled"] is True
    rule_messages = [item for item in bot.private_messages if item[1] == plugin.APPROVAL_RULES_MESSAGE]
    assert {user_id for user_id, _ in rule_messages} == set(plugin._approval_user_ids())


def test_owner_can_disable_review_and_clear_pending(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "关闭审查"))

    assert handled
    assert not plugin._approval_review_enabled()
    assert plugin.pending_group_approvals == {}
    assert store.app_kv_get(plugin.APPROVAL_REVIEW_ENABLED_KEY) == "false"
    assert bot.private_messages[0] == (
        1535071184,
        "已关闭审查，后续 bot 会直接发送第 1 候选；当前待审候选已清空。",
    )


def test_owner_can_enable_review(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.app_kv_set(plugin.APPROVAL_REVIEW_ENABLED_KEY, "false")
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "开启审查"))

    assert handled
    assert plugin._approval_review_enabled()
    assert store.app_kv_get(plugin.APPROVAL_REVIEW_ENABLED_KEY) == "true"
    assert bot.private_messages[0] == (1535071184, "已开启审查，bot 发群前会先发审批单。")


def test_owner_can_query_review_status(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.app_kv_set(plugin.APPROVAL_REVIEW_ENABLED_KEY, "false")
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "审查状态"))

    assert handled
    assert "关闭审查" in bot.private_messages[-1][1]


def test_basic_approver_cannot_disable_review(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "关闭审查"))

    assert handled
    assert plugin._approval_review_enabled()
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


def test_owner_can_query_model_status(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    monkeypatch.setattr(plugin, "deepseek_client", FakeModelClient())
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "模型状态"))

    assert handled
    assert "模型状态" in bot.private_messages[-1][1]
    assert "回复" in bot.private_messages[-1][1]
    assert "风格" in bot.private_messages[-1][1]
    assert "画像" in bot.private_messages[-1][1]
    assert "可切换模型" in bot.private_messages[-1][1]
    assert "fallback" in bot.private_messages[-1][1]


def test_owner_can_switch_reply_model(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    client = FakeModelClient()
    monkeypatch.setattr(plugin, "deepseek_client", client)
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(
            bot,
            1535071184,
            "切回复模型 siliconflow/MiniMaxAI/MiniMax-M2.5",
        )
    )

    assert handled
    assert client.current_route("reply").label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert "MiniMaxAI/MiniMax-M2.5" in store.app_kv_get(plugin.MODEL_ROUTE_OVERRIDES_KEY)
    assert bot.private_messages[-1] == (
        1535071184,
        "已切回复模型：siliconflow/MiniMaxAI/MiniMax-M2.5\n影响路由：reply",
    )


def test_owner_can_switch_style_model(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    client = FakeModelClient()
    monkeypatch.setattr(plugin, "deepseek_client", client)
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(
            bot,
            1535071184,
            "切风格模型 siliconflow/MiniMaxAI/MiniMax-M2.5",
        )
    )

    assert handled
    assert client.current_route("style").label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert "MiniMaxAI/MiniMax-M2.5" in store.app_kv_get(plugin.MODEL_ROUTE_OVERRIDES_KEY)
    assert bot.private_messages[-1] == (
        1535071184,
        "已切风格模型：siliconflow/MiniMaxAI/MiniMax-M2.5\n影响路由：style",
    )


def test_owner_can_switch_member_profile_model(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    client = FakeModelClient()
    monkeypatch.setattr(plugin, "deepseek_client", client)
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(
            bot,
            1535071184,
            "切画像模型 siliconflow/MiniMaxAI/MiniMax-M2.5",
        )
    )

    assert handled
    assert client.current_route("member_profile").label == "siliconflow/MiniMaxAI/MiniMax-M2.5"
    assert "MiniMaxAI/MiniMax-M2.5" in store.app_kv_get(plugin.MODEL_ROUTE_OVERRIDES_KEY)
    assert bot.private_messages[-1] == (
        1535071184,
        "已切画像模型：siliconflow/MiniMaxAI/MiniMax-M2.5\n影响路由：member_profile",
    )


def test_owner_can_switch_utility_group_models(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    client = FakeModelClient()
    monkeypatch.setattr(plugin, "deepseek_client", client)
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(
            bot,
            1535071184,
            "切工具模型 deepseek/deepseek-v4-flash",
        )
    )

    assert handled
    assert client.current_route("jargon").label == "deepseek/deepseek-v4-flash"
    assert client.current_route("memory").label == "deepseek/deepseek-v4-flash"
    assert client.current_route("style").label == "deepseek/deepseek-v4-flash"
    assert client.current_route("member_profile").label == "deepseek/deepseek-v4-flash"
    assert '"jargon": "deepseek/deepseek-v4-flash"' in store.app_kv_get(plugin.MODEL_ROUTE_OVERRIDES_KEY)
    assert bot.private_messages[-1] == (
        1535071184,
        "已切工具模型：deepseek/deepseek-v4-flash\n影响路由：jargon、memory、style、member_profile",
    )


def test_owner_can_clear_model_overrides(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    client = FakeModelClient()
    route = client.parse_model_route("siliconflow/MiniMaxAI/MiniMax-M2.5")
    client.set_route_override("reply", route)
    store.app_kv_set(plugin.MODEL_ROUTE_OVERRIDES_KEY, '{"reply":"siliconflow/MiniMaxAI/MiniMax-M2.5"}')
    monkeypatch.setattr(plugin, "deepseek_client", client)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "清模型覆盖"))

    assert handled
    assert client.current_route("reply").label == plugin.app_config.deepseek.routes["reply"].label
    assert store.app_kv_get(plugin.MODEL_ROUTE_OVERRIDES_KEY) == "{}"
    assert bot.private_messages[-1] == (1535071184, "已清除模型覆盖，恢复 config.yaml 默认模型。")


def test_basic_approver_cannot_switch_model(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    monkeypatch.setattr(plugin, "deepseek_client", FakeModelClient())
    bot = FakeApprovalBot()

    handled = asyncio.run(
        plugin._handle_group_approval_private(
            bot,
            3370998238,
            "切回复模型 siliconflow/MiniMaxAI/MiniMax-M2.5",
        )
    )

    assert handled
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


def test_request_group_approval_auto_sends_first_candidate_when_review_disabled(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    store.app_kv_set(plugin.APPROVAL_REVIEW_ENABLED_KEY, "false")
    approval = _pending_approval()
    bot = FakeApprovalBot()

    asyncio.run(plugin._request_group_approval(bot, approval))

    assert plugin.pending_group_approvals == {}
    assert bot.group_messages == [(1026813421, "第一条回复")]
    assert bot.private_messages == []
    sent_messages = store.recent_messages(1026813421, 3)
    assert sent_messages[-1].text == "第一条回复"


def test_owner_can_set_approval_auto_send_percent(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "审批概率 30"))

    assert handled
    assert plugin._approval_auto_send_percent() == 30
    assert "30%" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "审批概率"))

    assert handled
    assert "免审自动发送概率：30%" in bot.private_messages[-1][1]


def test_limited_approver_can_set_approval_auto_send_percent_with_floor(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "审批概率 30"))

    assert handled
    assert plugin._approval_auto_send_percent() == 0
    assert "60% 到 100%" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "审批概率 60"))

    assert handled
    assert plugin._approval_auto_send_percent() == 60
    assert "60%" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "审批概率"))

    assert handled
    assert "免审自动发送概率：60%" in bot.private_messages[-1][1]


def test_regular_basic_approver_cannot_set_approval_auto_send_percent(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    plugin._save_basic_approval_user_ids({123456789})
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 123456789, "审批概率 80"))

    assert handled
    assert plugin._approval_auto_send_percent() == 0
    assert bot.private_messages[-1] == (123456789, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


def test_ai_work_intensity_defaults_to_full(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert plugin._ai_work_intensity_percent() == 100
    assert plugin._ai_work_intensity_selected()


def test_owner_can_set_ai_work_intensity_percent(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "工作强度 30"))

    assert handled
    assert plugin._ai_work_intensity_percent() == 30
    assert "30%" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "AI强度"))

    assert handled
    assert "AI工作强度：30%" in bot.private_messages[-1][1]
    assert "不影响：消息照常写入数据库" in bot.private_messages[-1][1]


def test_basic_approver_cannot_set_ai_work_intensity_percent(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "工作强度 30"))

    assert handled
    assert plugin._ai_work_intensity_percent() == 100
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")


def test_ai_work_intensity_selection_respects_bounds(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert plugin._set_ai_work_intensity_percent(-10) == 0
    assert not plugin._ai_work_intensity_selected()
    assert plugin._set_ai_work_intensity_percent(120) == 100
    assert plugin._ai_work_intensity_selected()


def test_ai_work_intensity_only_applies_to_unaddressed_messages() -> None:
    assert plugin._ai_work_intensity_applies(addressed_bot=False)
    assert not plugin._ai_work_intensity_applies(addressed_bot=True)


def test_request_group_approval_auto_sends_by_probability(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    plugin._set_approval_auto_send_percent(100)
    approval = _pending_approval()
    bot = FakeApprovalBot()

    asyncio.run(plugin._request_group_approval(bot, approval))

    assert plugin.pending_group_approvals == {}
    assert bot.group_messages == [(1026813421, "第一条回复")]
    assert bot.private_messages == []
    sent_messages = store.recent_messages(1026813421, 3)
    assert sent_messages[-1].text == "第一条回复"


def test_approval_direct_single_reply_enabled_only_when_deterministic(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert not plugin._approval_direct_single_reply_enabled()

    plugin._set_approval_auto_send_percent(100)
    assert plugin._approval_direct_single_reply_enabled()

    plugin._set_approval_auto_send_percent(60)
    assert not plugin._approval_direct_single_reply_enabled()

    store.app_kv_set(plugin.APPROVAL_REVIEW_ENABLED_KEY, "false")
    assert plugin._approval_direct_single_reply_enabled()


def test_changelog_notice_sent_once(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    asyncio.run(plugin._send_changelog_notice_to_approvers(bot))
    asyncio.run(plugin._send_changelog_notice_to_approvers(bot))

    notices = [message for _, message in bot.private_messages if "后端更新记录" in message]
    assert len(notices) == len(plugin._approval_user_ids())
    assert all("LLM 路由拆细" in message for message in notices)


def test_parse_suppression_report_accepts_compact_chinese_limit() -> None:
    assert plugin._parse_approval_suppression_report_command("拦截20") == 20
    assert plugin._parse_approval_suppression_report_command("拦截 20") == 20
    assert plugin._parse_approval_suppression_report_command("blocked 20") == 20


def test_suppression_notice_records_without_private_spam(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    asyncio.run(
        plugin._send_approval_suppression_notice(
            bot,
            group_id=1026813421,
            user_id=184589072,
            nickname="小鸟",
            text="哈哈哈哈",
            stage="backend_low_value",
            reason="后端低价值硬拦截",
        )
    )
    asyncio.run(
        plugin._send_approval_suppression_notice(
            bot,
            group_id=1026813421,
            user_id=184589072,
            nickname="小鸟",
            text="哈哈哈哈",
            stage="backend_low_value",
            reason="后端低价值硬拦截",
        )
    )

    notices = [message for _, message in bot.private_messages if "拦截通知" in message]
    assert notices == []
    assert len(plugin.recent_suppression_events) == 2
    report = plugin._format_suppression_report(1)
    assert "最近拦截（1 条）" in report
    assert "backend_low_value" in report
    assert plugin.pending_group_approvals == {}


def test_approval_private_jargon_command_does_not_consume_pending(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "/黑话 达斯=打死"))

    assert handled
    assert plugin.pending_group_approvals[approval.group_id] == approval
    entries = store.custom_jargon_entries(1026813421)
    assert len(entries) == 1
    assert entries[0].term == "达斯"
    assert entries[0].explanation == "指代：打死"
    assert bot.private_messages[-1] == (1535071184, "已记黑话：达斯 -> 打死")


def test_allowed_private_user_cannot_write_custom_jargon(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)

    response = plugin._handle_jargon_command_text(
        user_id=3115344487,
        group_id=1026813421,
        text="/黑话：火宅：活摘",
    )

    assert response == "没权限。"
    entries = store.custom_jargon_entries(1026813421)
    assert len(entries) == 0


def test_owner_can_manage_private_whitelist(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    assert plugin._private_user_allowed(1535071184)
    assert not plugin._private_user_can_chat(1535071184)
    assert plugin._private_user_can_chat(plugin.PRIVATE_DEBUG_OWNER_ID)

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "加私聊 123456789"))

    assert handled
    assert store.app_kv_get(plugin.PRIVATE_WHITELIST_KEY) == "[123456789]"
    assert plugin._private_user_allowed(123456789)
    assert bot.private_messages[-1] == (1535071184, "已添加私聊白名单：123456789")

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "私聊白名单"))

    assert handled
    assert "运行时添加：123456789" in bot.private_messages[-1][1]
    assert "config 固定" in bot.private_messages[-1][1]

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "删私聊 123456789"))

    assert handled
    assert store.app_kv_get(plugin.PRIVATE_WHITELIST_KEY) == "[]"
    assert not plugin._private_user_allowed(123456789)


def test_private_force_obey_toggle_and_priority_context(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert not plugin._private_force_obey_enabled(plugin.PRIVATE_DEBUG_OWNER_ID)
    assert plugin._private_force_obey_command_response(plugin.PRIVATE_DEBUG_OWNER_ID, "强服从") == (
        "强服从已开启。之后这个测试号私聊会注入最高优先级调试提示。"
    )
    assert store.app_kv_get(plugin.PRIVATE_FORCE_OBEY_KEY) == f"[{plugin.PRIVATE_DEBUG_OWNER_ID}]"
    assert plugin._private_force_obey_enabled(plugin.PRIVATE_DEBUG_OWNER_ID)

    context = plugin._private_priority_context(plugin.PRIVATE_DEBUG_OWNER_ID)
    assert "私聊测试账号" in context
    assert "强服从调试模式" in context
    assert "最高优先级调试指令" in context
    assert "2776760548" in context

    assert plugin._private_force_obey_command_response(plugin.PRIVATE_DEBUG_OWNER_ID, "关闭强服从") == (
        "强服从已关闭。之后恢复普通测试号私聊优先级。"
    )
    assert not plugin._private_force_obey_enabled(plugin.PRIVATE_DEBUG_OWNER_ID)


def test_private_force_obey_rejects_non_test_account(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert plugin._private_force_obey_command_response(3115344487, "强服从") == (
        "这个命令只给测试号 2776760548 用。"
    )
    assert not plugin._private_force_obey_enabled(3115344487)
    assert plugin._extract_private_force_obey_once_text(3115344487, "强服从：按我说的回") is None


def test_private_force_obey_once_parser(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)

    assert plugin._extract_private_force_obey_once_text(
        plugin.PRIVATE_DEBUG_OWNER_ID,
        "强服从：按我说的重写",
    ) == "按我说的重写"
    assert plugin._extract_private_force_obey_once_text(
        plugin.PRIVATE_DEBUG_OWNER_ID,
        "/obey: answer directly",
    ) == "answer directly"
    assert plugin._private_force_obey_command_response(plugin.PRIVATE_DEBUG_OWNER_ID, "强服从状态") == (
        "强服从状态：已关闭。可用 强服从 / 关闭强服从 / 强服从：具体内容。"
    )


def test_basic_approver_cannot_manage_private_whitelist(monkeypatch, tmp_path) -> None:
    _use_temp_plugin_memory(monkeypatch, tmp_path)
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 3370998238, "加私聊 123456789"))

    assert handled
    assert bot.private_messages[-1] == (3370998238, "你只有基础审批权限：A/B/C/D/X/1/2/3/取消 处理审批单。")
    assert not plugin._private_user_allowed(123456789)


def test_approval_reject_second_candidate_records_that_candidate(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "不准奏2原因：这句太端水，少点客服味"))

    assert handled
    assert not bot.group_messages
    feedback = store.recent_recalled_reply_feedback(1026813421, 3)
    assert len(feedback) == 1
    assert feedback[0].bot_reply == "第二条回复"
    assert feedback[0].scene_summary == "审批不准奏原始评价，针对第 2 条候选"
    assert feedback[0].owner_reason == "这句太端水，少点客服味"


def test_approval_high_quality_choice_sends_and_records_positive(monkeypatch, tmp_path) -> None:
    store = _use_temp_plugin_memory(monkeypatch, tmp_path)
    approval = _pending_approval()
    plugin.pending_group_approvals[approval.group_id] = approval
    bot = FakeApprovalBot()

    handled = asyncio.run(plugin._handle_group_approval_private(bot, 1535071184, "B!"))

    assert handled
    assert bot.group_messages == [(1026813421, "第二条回复")]
    approved = store.recent_approved_reply_feedback(1026813421, 3)
    assert len(approved) == 1
    assert approved[0].candidate_text == "第二条回复"
    assert approved[0].operator_id == 1535071184


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
