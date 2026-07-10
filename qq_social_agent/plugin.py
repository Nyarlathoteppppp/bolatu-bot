from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from nonebot import get_driver, logger, on_command, on_message
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment, PrivateMessageEvent
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.rule import Rule

from .approval_rules import (
    APPROVAL_CHOICE_RE,
    APPROVAL_DETAIL_COMMANDS,
    APPROVAL_HELP_COMMANDS,
    APPROVAL_REJECT_REASON_RE,
    APPROVAL_RULES_DETAIL_MESSAGE,
    APPROVAL_RULES_MESSAGE,
    JARGON_ADD_RE,
    JARGON_DELETE_RE,
    JARGON_LIST_RE,
    TOKEN_REPORT_COMMAND_ALIASES,
)
from .config import load_config
from .cue_patterns import CuePatternTracker, CueRepeatState
from .decision_gate import (
    apply_backend_tool_decision as _apply_backend_tool_decision,
    context_query_text as _context_query_text,
    is_explicit_market_lookup as _is_explicit_market_lookup,
    is_low_value_group_text as _is_low_value_group_text,
    pre_decision_gate as _pre_decision_gate,
)
from .deepseek_client import DeepSeekClient, ReplyDecision, set_usage_recorder
from .group_jargon import (
    GroupJargonEntry,
    detect_group_jargon_terms,
    group_jargon_catalog,
    group_jargon_context,
)
from .memory import (
    ApprovedReplyFeedback,
    ChatMessage,
    CustomJargonEntry,
    LLMUsageEvent,
    LLMUsageSummary,
    MemberProfile,
    MemoryStore,
    MemorySummary,
    RecalledReplyFeedback,
    StyleRule,
)
from .persona import PersonaRegistry
from .political_guard import has_political_redline, political_safe_reply, sanitize_political_output
from .rate_limiter import RateLimiter
from .reply_splitter import split_reply_messages
from .tools.fresh_context import (
    FreshContextTool,
    detect_fresh_intent,
)
from .tools.market import MarketTool
from .tools.market_intent import MarketIntent, detect_market_intents, is_market_topic


load_dotenv()

app_config = load_config()
memory = MemoryStore(app_config.data_path)
personas = PersonaRegistry(app_config.persona_dir)
rate_limiter = RateLimiter(memory, app_config.rate)
market_tool = MarketTool(max_external_queries_per_minute=2)
fresh_context_tool = FreshContextTool(max_external_queries_per_minute=2)
cue_pattern_tracker = CuePatternTracker(window_seconds=10 * 60)
deepseek_client: DeepSeekClient | None = None
last_mid_memory_attempt: dict[int, float] = {}
last_style_learn_attempt: dict[int, float] = {}
addressed_event_times: dict[tuple[int, int], list[float]] = {}
last_group_mention_targets: dict[int, tuple[int, float]] = {}
last_user_reply_times: dict[tuple[int, int], float] = {}
group_processing_locks: dict[int, asyncio.Lock] = {}
group_learning_tasks: dict[int, asyncio.Task[None]] = {}
group_message_buffers: dict[int, list["BufferedGroupMessage"]] = {}
group_buffer_tasks: dict[int, asyncio.Task[None]] = {}
group_passive_decision_state: dict[int, tuple[float, int]] = {}
pending_group_approvals: dict[int, "PendingGroupApproval"] = {}
last_suppression_notice_times: dict[tuple[int, str], float] = {}

MID_MEMORY_KEEP_SUMMARIES = 4
MID_MEMORY_BATCH_SIZE = 60
MID_MEMORY_MIN_BATCH = 24
MID_MEMORY_RETRY_INTERVAL_SECONDS = 10 * 60
STYLE_LEARN_INTERVAL_SECONDS = 60 * 60
STYLE_LEARN_MESSAGE_LIMIT = 40
STYLE_LEARN_MIN_MESSAGES = 12
STYLE_RULE_CONTEXT_LIMIT = 12
JARGON_CONTEXT_LOOKBACK = 4
CUSTOM_JARGON_CONTEXT_LIMIT = 10
GROUP_BUFFER_SECONDS = 6.0
GROUP_PASSIVE_DECISION_GAP_SECONDS = 30
GROUP_PASSIVE_DECISION_EVERY_MESSAGES = 3
SUPPRESSION_NOTICE_COOLDOWN_SECONDS = 60
ADDRESS_REPEAT_WINDOW_SECONDS = 10 * 60
MENTION_TARGET_LIMIT = 8
REPEAT_MENTION_SUPPRESS_SECONDS = 10 * 60
PRIVATE_DEBUG_OWNER_ID = 2776760548
GROUP_APPROVAL_USER_IDS = (1535071184, 3370998238)
JARGON_COMMAND_USER_IDS = tuple(
    sorted({PRIVATE_DEBUG_OWNER_ID, *GROUP_APPROVAL_USER_IDS, *app_config.allowed_private_users})
)
RECALL_FEEDBACK_CONTEXT_LIMIT = 3
POSITIVE_FEEDBACK_CONTEXT_LIMIT = 4
LLM_USAGE_LOG_RE = re.compile(
    r"^(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hms>\d{2}:\d{2}:\d{2}).*qq_social_agent llm usage: "
    r"task=(?P<task>\S+) model=(?P<model>\S+) "
    r"prompt_tokens=(?P<prompt>\d+|None) "
    r"completion_tokens=(?P<completion>\d+|None) "
    r"total_tokens=(?P<total>\d+|None)"
)
TOKEN_REPORT_DEFAULT_WINDOW_SECONDS = 24 * 60 * 60
TOKEN_REPORT_MAX_RECENT_EVENTS = 8
TOKEN_USAGE_LOG_BACKFILL_FILES = (
    Path(__file__).resolve().parent.parent / "logs" / "bot-runtime.log",
    Path(__file__).resolve().parent.parent / "logs" / "bot.log",
)
CHANGELOG_NOTICE_KEY = "2026-07-10-search-approval-v2"
CHANGELOG_NOTICE_MESSAGE = """张风雪后端更新记录：
1. 搜索优化：是否联网搜索交给 decision LLM 判断；后端只给“可能需要最新背景”的候选提示。
2. Tavily 搜索结果现在会注入快速摘要、去重后的来源和更稳的短摘要，失败时明确提示没拿到可靠新消息。
3. 最新新闻/赛果类消息不会再被普通频率门提前挡掉，但是否搜索、是否回复仍由 LLM 决策。
4. 人设已调整为毒舌美少女现实判断型损友，同时保留 answer/agree 的正常好好讲话分支。
5. 清理了一条会错误泛化语气的旧反馈；保留温和认可 action，不会把所有人都当成要哄。
6. 后端拦截、频率门拦截或 LLM 判断不发时，会限流给审批人发一条调试通知，方便判断为什么没进候选。

审批提醒：
- 黑话：/黑话：词 指代：解释；/黑话列表；/删黑话：词。
- 不准奏：不准奏原因：xxx；不准奏2原因：xxx 可批评指定候选。
- token：token用量 / token用量 2026-07-10 / token用量 7d。
"""


@dataclass(frozen=True)
class BufferedGroupMessage:
    bot: Bot
    event: GroupMessageEvent
    text: str
    user_id: int
    nickname: str
    created_at: float


@dataclass(frozen=True)
class PendingApprovalCandidate:
    index: int
    text: str
    action: str
    style: str


@dataclass(frozen=True)
class PendingGroupApproval:
    group_id: int
    trigger_user_id: int
    trigger_nickname: str
    trigger_text: str
    persona_name: str
    self_id: int
    candidates: tuple[PendingApprovalCandidate, ...]
    mention_targets: dict[int, str]
    created_at: float


@dataclass(frozen=True)
class TokenReportWindow:
    start_at: float | None
    end_at: float | None
    label: str


@get_driver().on_startup
async def _init_client() -> None:
    global deepseek_client
    set_usage_recorder(_record_llm_usage)
    deepseek_client = DeepSeekClient(app_config.deepseek)


def _record_llm_usage(
    task: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> None:
    memory.add_llm_usage(
        task=task,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


@get_driver().on_bot_connect
async def _send_approval_rules_on_connect(bot: Bot) -> None:
    await _send_approval_rules_to_approvers(bot, reason="bot_connect")
    await _send_changelog_notice_to_approvers(bot)


async def _send_approval_rules_to_approvers(bot: Bot, *, reason: str) -> None:
    for approver_id in GROUP_APPROVAL_USER_IDS:
        try:
            await bot.send_private_msg(user_id=approver_id, message=Message(APPROVAL_RULES_MESSAGE))
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending approval rules: "
                f"reason={reason} approver={approver_id} {_action_failed_summary(exc)}"
            )


async def _send_changelog_notice_to_approvers(bot: Bot) -> None:
    marker_key = f"changelog_notice:{CHANGELOG_NOTICE_KEY}"
    delivered: list[int] = []
    for approver_id in GROUP_APPROVAL_USER_IDS:
        if _changelog_notice_sent(marker_key, approver_id):
            continue
        try:
            await bot.send_private_msg(user_id=approver_id, message=Message(CHANGELOG_NOTICE_MESSAGE))
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending changelog notice: "
                f"approver={approver_id} {_action_failed_summary(exc)}"
            )
            continue
        _mark_changelog_notice_sent(marker_key, approver_id)
        delivered.append(approver_id)
    if delivered:
        logger.info(
            "qq_social_agent changelog notice sent: "
            f"key={CHANGELOG_NOTICE_KEY} approvers={delivered}"
        )


def _changelog_notice_sent(marker_key: str, approver_id: int) -> bool:
    return memory.app_kv_get(_changelog_notice_marker(marker_key, approver_id)) == "sent"


def _mark_changelog_notice_sent(marker_key: str, approver_id: int) -> None:
    memory.app_kv_set(_changelog_notice_marker(marker_key, approver_id), "sent")


def _changelog_notice_marker(marker_key: str, approver_id: int) -> str:
    return f"{marker_key}:{approver_id}"


async def _send_approval_suppression_notice(
    bot: Bot,
    *,
    group_id: int,
    user_id: int,
    nickname: str,
    text: str,
    stage: str,
    reason: str,
) -> None:
    now = time.monotonic()
    key = (group_id, stage)
    last_sent_at = last_suppression_notice_times.get(key)
    if last_sent_at is not None and now - last_sent_at < SUPPRESSION_NOTICE_COOLDOWN_SECONDS:
        return
    last_suppression_notice_times[key] = now
    message = (
        "拦截通知：这不是待审候选\n"
        f"群：{group_id}\n"
        f"阶段：{stage}\n"
        f"触发人：{_member_label(user_id, nickname)}\n"
        f"消息：{_short_notice_text(text, 180)}\n"
        f"原因：{_short_notice_text(reason, 220)}"
    )
    for approver_id in GROUP_APPROVAL_USER_IDS:
        await _send_private_text(bot, approver_id, message)


def _short_notice_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


PRIVATE_CHAT_OFFSET = 10_000_000_000_000


def _is_group_event(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


def _is_private_event(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


def _is_jargon_command_event(event: Event) -> bool:
    if not isinstance(event, (GroupMessageEvent, PrivateMessageEvent)):
        return False
    text = _event_plain_text(event)
    return _is_jargon_command_text(text)


jargon_command = on_message(rule=Rule(_is_jargon_command_event), priority=9, block=True)
group_message = on_message(rule=Rule(_is_group_event), priority=50, block=False)
private_message = on_message(rule=Rule(_is_private_event), priority=50, block=False)


@jargon_command.handle()
async def handle_jargon_command(event: Event, matcher: Matcher) -> None:
    user_id = int(getattr(event, "user_id", 0) or 0)
    group_id = _jargon_command_group_id(event)
    await matcher.finish(
        _handle_jargon_command_text(
            user_id=user_id,
            group_id=group_id,
            text=_event_plain_text(event),
        )
    )


@group_message.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    group_id = int(event.group_id)
    text = _plain_text(event)
    addressed_bot = _mentioned_bot(event, bot) or _replied_to_bot(event, bot)
    if (
        text
        and app_config.group_allowed(group_id)
        and not addressed_bot
        and _is_low_value_group_text(text)
    ):
        memory.add_message(group_id, int(event.user_id), _nickname(event), text, is_bot=False)
        _passive_decision_allowed(
            group_id,
            message_count=1,
            first_message_at=float(getattr(event, "time", 0) or time.time()),
            last_message_at=float(getattr(event, "time", 0) or time.time()),
        )
        logger.info(
            "qq_social_agent ignored group low value text: "
            f"group={group_id} user={int(event.user_id)} text={text!r}"
        )
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=int(event.user_id),
            nickname=_nickname(event),
            text=text,
            stage="backend_low_value",
            reason="后端低价值硬拦截：纯表情/单字/短笑声，不进入 buffer 和 LLM decision。",
        )
        return
    if (
        text
        and app_config.group_allowed(group_id)
        and not addressed_bot
    ):
        _buffer_group_message(bot, event, text)
        return
    async with _group_processing_lock(group_id):
        await _handle_group_message_locked(bot, event)


async def _handle_group_message_locked(
    bot: Bot,
    event: GroupMessageEvent,
    *,
    buffered_messages: list[BufferedGroupMessage] | None = None,
) -> None:
    text = _buffered_current_text(buffered_messages) if buffered_messages else _plain_text(event)
    group_id = int(event.group_id)
    if not app_config.group_allowed(group_id):
        logger.info(f"qq_social_agent ignored group={group_id}: not_allowed")
        return

    user_id = _buffered_current_user_id(buffered_messages) if buffered_messages else int(event.user_id)
    nickname = _buffered_current_nickname(buffered_messages) if buffered_messages else _nickname(event)
    mentioned = False if buffered_messages else _mentioned_bot(event, bot)
    replied_to_bot = False if buffered_messages else _replied_to_bot(event, bot)
    addressed_bot = mentioned or replied_to_bot
    addressed_repeat_count = _record_addressed_event(group_id, user_id, addressed_bot)

    if not text:
        if not addressed_bot:
            return
        text = "（只艾特了你）"

    if buffered_messages:
        for item in buffered_messages:
            memory.add_message(group_id, item.user_id, item.nickname, item.text, is_bot=False)
    else:
        memory.add_message(group_id, user_id, nickname, text, is_bot=False)

    group_cfg = app_config.group_config(group_id)
    state = memory.group_state(group_id)
    enabled = bool(group_cfg.get("enabled", True)) and bool(state["enabled"])
    if not enabled:
        logger.info(f"qq_social_agent ignored group={group_id}: disabled")
        return

    market_intents = detect_market_intents(text, limit=2)
    market_topic = bool(market_intents) or is_market_topic(text)
    fresh_intent = detect_fresh_intent(text)
    market_forced = bool(market_intents) and _is_explicit_market_lookup(text)
    fresh_candidate = fresh_intent is not None
    if addressed_bot:
        _mark_passive_decision_forced(group_id)
    elif not market_forced and not fresh_candidate:
        message_count = len(buffered_messages) if buffered_messages else 1
        first_message_at = _buffered_first_created_at(buffered_messages)
        last_message_at = _buffered_last_created_at(buffered_messages)
        allowed, reason = _passive_decision_allowed(
            group_id,
            message_count=message_count,
            first_message_at=first_message_at,
            last_message_at=last_message_at,
        )
        if not allowed:
            logger.info(
                "qq_social_agent skipped passive decision gate: "
                f"group={group_id} messages={message_count} reason={reason}"
            )
            await _send_approval_suppression_notice(
                bot,
                group_id=group_id,
                user_id=user_id,
                nickname=nickname,
                text=text,
                stage="passive_frequency_gate",
                reason=f"被动发言频率门拦截：{reason}。30 秒内连续聊天时，每 3 条才进一次 decision。",
            )
            _schedule_group_learning(group_id)
            return
    else:
        _mark_passive_decision_forced(group_id)

    cue_repeat_state = cue_pattern_tracker.record(
        group_id=group_id,
        user_id=user_id,
        text=text,
        addressed=addressed_bot,
    )
    logger.info(
        "qq_social_agent group decision start: "
        f"group={group_id} user={user_id} mentioned={mentioned} replied_to_bot={replied_to_bot} text={text!r}"
    )

    persona_id = str(state["persona"] or group_cfg.get("persona") or app_config.default_persona)
    persona = personas.get(persona_id)

    recent = memory.recent_messages(group_id, app_config.context_limit)
    context_recent = _without_current_message(recent, user_id=user_id, text=text)
    rate = rate_limiter.allow(group_id, mentioned=addressed_bot)
    if not rate.allowed:
        logger.info(f"qq_social_agent suppressed by rate: group={group_id} reason={rate.reason}")
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            text=text,
            stage="reply_rate_limiter",
            reason=f"发言频率限制拦截：{rate.reason}",
        )
        return

    if has_political_redline(text):
        logger.info(
            "qq_social_agent political guard input: "
            f"group={group_id} addressed={addressed_bot} text={text!r}"
        )
        if not addressed_bot:
            await _send_approval_suppression_notice(
                bot,
                group_id=group_id,
                user_id=user_id,
                nickname=nickname,
                text=text,
                stage="political_guard",
                reason="非点名消息命中中国政治红线兜底，后端直接不插话。",
            )
            return
        reply = political_safe_reply()
        await _request_group_approval(
            bot,
            PendingGroupApproval(
                group_id=group_id,
                trigger_user_id=user_id,
                trigger_nickname=nickname,
                trigger_text=text,
                persona_name=persona.name,
                self_id=int(event.self_id),
                candidates=(PendingApprovalCandidate(1, reply, "political_guard", "政治红线兜底"),),
                mention_targets={},
                created_at=time.time(),
            ),
        )
        return

    if _user_reply_cooling_down(group_id, user_id):
        logger.info(
            "qq_social_agent suppressed by user cooldown: "
            f"group={group_id} user={user_id} cooldown={app_config.user_reply_cooldowns[user_id]}"
        )
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            text=text,
            stage="user_cooldown",
            reason=f"该用户单独限频中：{app_config.user_reply_cooldowns[user_id]} 秒内最多回一次。",
        )
        _schedule_group_learning(group_id)
        return

    if deepseek_client is None:
        logger.warning("qq_social_agent skipped: deepseek_client_not_ready")
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            text=text,
            stage="deepseek_not_ready",
            reason="DeepSeek client 还没初始化，无法进入 LLM decision。",
        )
        return

    decision: ReplyDecision | None = None
    memory_context = ""
    recall_feedback_context = ""
    positive_feedback_context = ""
    member_context = ""
    style_context = ""
    jargon_context = ""
    context_query = _context_query_text(text, nickname, context_recent)

    pre_decision = _pre_decision_gate(
        text=text,
        recent_messages=context_recent,
        persona=persona,
        addressed_bot=addressed_bot,
        mentioned=mentioned,
        replied_to_bot=replied_to_bot,
        cue_repeat_state=cue_repeat_state,
        market_intents=market_intents,
        fresh_intent=fresh_intent,
    )
    if pre_decision.skip_reason:
        logger.info(
            "qq_social_agent skipped by local pre-decision gate: "
            f"group={group_id} reason={pre_decision.skip_reason}"
        )
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            text=text,
            stage="backend_pre_decision",
            reason=f"本地预决策拦截：{pre_decision.skip_reason}",
        )
        _schedule_group_learning(group_id)
        return
    decision = pre_decision.decision

    if decision is None:
        memory_context = _format_memory_context(
            memory.relevant_memory_summaries(
                group_id,
                context_query,
                limit=MID_MEMORY_KEEP_SUMMARIES,
            )
        )
        member_context = _format_member_context(
            memory.member_profiles_for_context(
                group_id,
                _related_member_user_ids(context_recent, current_user_id=user_id),
                limit=8,
            )
        )
        style_context = _format_style_context(
            memory.relevant_style_rules(
                group_id,
                context_query,
                limit=STYLE_RULE_CONTEXT_LIMIT,
            )
        )
        jargon_context = await _selected_group_jargon_context(
            group_id,
            context_recent,
            current_text=text,
            current_nickname=nickname,
        )

        try:
            decision = await deepseek_client.should_reply(
                persona=persona,
                recent_messages=context_recent,
                current_text=text,
                current_nickname=_member_label(user_id, nickname),
                mentioned=mentioned,
                replied_to_bot=replied_to_bot,
                addressed_repeat_count=addressed_repeat_count,
                cue_repeat_context=_format_cue_repeat_context(cue_repeat_state),
                market_topic=market_topic,
                chat_label="QQ 群聊",
                memory_context=memory_context,
                style_context=style_context,
                jargon_context=jargon_context,
                member_context=member_context,
                fresh_context_hint=_format_fresh_context_hint(fresh_intent),
            )
        except Exception as exc:
            decision = _decision_failure_fallback(
                addressed_bot=addressed_bot,
                reason="decision_error",
            )
            logger.warning(
                "qq_social_agent decision failed: "
                f"group={group_id} addressed={addressed_bot} error={exc}"
            )
            if decision is None:
                await _send_approval_suppression_notice(
                    bot,
                    group_id=group_id,
                    user_id=user_id,
                    nickname=nickname,
                    text=text,
                    stage="llm_decision_error",
                    reason=f"decision LLM 调用失败，且非点名没有兜底回复：{exc}",
                )
                _schedule_group_learning(group_id)
                return
    else:
        logger.info(
            "qq_social_agent local pre-decision: "
            f"group={group_id} should_reply={decision.should_reply} "
            f"action={decision.action} mode={decision.mode} reason={decision.reason}"
        )
    if decision.reason == "invalid_json":
        fallback_decision = _decision_failure_fallback(
            addressed_bot=addressed_bot,
            reason="decision_invalid_json",
        )
        if fallback_decision is None:
            logger.warning(
                "qq_social_agent decision invalid json ignored: "
                f"group={group_id} addressed={addressed_bot}"
            )
            await _send_approval_suppression_notice(
                bot,
                group_id=group_id,
                user_id=user_id,
                nickname=nickname,
                text=text,
                stage="llm_invalid_json",
                reason="decision LLM 返回 invalid_json，且非点名没有兜底回复。",
            )
            _schedule_group_learning(group_id)
            return
        logger.warning(
            "qq_social_agent decision invalid json fallback: "
            f"group={group_id} addressed={addressed_bot}"
        )
        decision = fallback_decision
    decision = _apply_backend_tool_decision(
        decision,
        text=text,
        market_intents=market_intents,
        fresh_intent=fresh_intent,
    )
    if addressed_bot and "非点名" in decision.reason:
        logger.warning(
            "qq_social_agent decision state mismatch: "
            f"group={group_id} addressed=True reason={decision.reason}"
        )
    logger.info(
        "qq_social_agent llm decision: "
        f"group={group_id} should_reply={decision.should_reply} "
        f"confidence={decision.confidence:.2f} action={decision.action} mode={decision.mode} "
        f"need_fresh={decision.need_fresh_context} fresh_query={decision.fresh_query!r} "
        f"reason={decision.reason}"
    )
    _schedule_group_learning(group_id)
    if not decision.should_reply:
        await _send_approval_suppression_notice(
            bot,
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            text=text,
            stage="llm_ignore",
            reason=(
                f"LLM 判断不发：action={decision.action} mode={decision.mode} "
                f"confidence={decision.confidence:.2f} reason={decision.reason}"
            ),
        )
        return

    if not memory_context:
        memory_context = _format_memory_context(
            memory.relevant_memory_summaries(
                group_id,
                context_query,
                limit=MID_MEMORY_KEEP_SUMMARIES,
            )
        )
    if not member_context:
        member_context = _format_member_context(
            memory.member_profiles_for_context(
                group_id,
                _related_member_user_ids(context_recent, current_user_id=user_id),
                limit=8,
            )
        )
    if not style_context:
        style_context = _format_style_context(
            memory.relevant_style_rules(
                group_id,
                context_query,
                limit=STYLE_RULE_CONTEXT_LIMIT,
            )
        )
    if not jargon_context:
        jargon_context = await _selected_group_jargon_context(
            group_id,
            context_recent,
            current_text=text,
            current_nickname=nickname,
        )
    recall_feedback_context = _format_recall_feedback_context(
        memory.recent_recalled_reply_feedback(group_id, RECALL_FEEDBACK_CONTEXT_LIMIT)
    )
    positive_feedback_context = _format_positive_feedback_context(
        memory.recent_approved_reply_feedback(group_id, POSITIVE_FEEDBACK_CONTEXT_LIMIT)
    )

    market_context = ""
    market_report = ""
    if decision.need_tool and decision.tool == "market":
        requested_intents = _market_intents_from_decision(
            decision,
            fallback_text=text,
            fallback_intents=market_intents,
        )
        market_report, market_context = await _market_report_and_context_for(
            requested_intents,
            market_topic=market_topic,
        )
        if market_report:
            logger.info(
                "qq_social_agent pending market report approval: "
                f"group={group_id} chars={len(market_report)}"
            )
            if not decision.comment_after_tool:
                await _request_group_approval(
                    bot,
                    PendingGroupApproval(
                        group_id=group_id,
                        trigger_user_id=user_id,
                        trigger_nickname=nickname,
                        trigger_text=text,
                        persona_name=persona.name,
                        self_id=int(event.self_id),
                        candidates=(
                            PendingApprovalCandidate(
                                1,
                                market_report,
                                "market_check",
                                "行情工具报告，不额外编判断",
                            ),
                        ),
                        mention_targets={},
                        created_at=time.time(),
                    ),
                )
                return

    fresh_context = ""
    if decision.need_fresh_context:
        fresh_context = await _fresh_context_for(decision, fallback_text=text)

    suppress_mention_user_id = _repeat_mention_suppressed_user(group_id, user_id)
    mention_targets = _mention_targets(
        context_recent,
        current_user_id=user_id,
        current_nickname=nickname,
        self_id=int(event.self_id),
        suppress_user_id=suppress_mention_user_id,
    )
    try:
        reply_candidates = await deepseek_client.reply_candidates(
            persona=persona,
            recent_messages=context_recent,
            current_text=text,
            current_nickname=_member_label(user_id, nickname),
            mentioned=addressed_bot,
            addressed_repeat_count=addressed_repeat_count,
            cue_repeat_context=_format_cue_repeat_context(cue_repeat_state),
            action=decision.action,
            chat_label="QQ 群聊",
            market_context=market_context,
            fresh_context=fresh_context,
            memory_context=memory_context,
            style_context=style_context,
            jargon_context=jargon_context,
            member_context=member_context,
            recall_feedback_context=recall_feedback_context,
            positive_feedback_context=positive_feedback_context,
            mention_targets=_format_mention_targets(mention_targets),
            include_bot_history=False,
        )
    except Exception as exc:
        logger.warning(
            "qq_social_agent reply candidate generation failed: "
            f"group={group_id} addressed={addressed_bot} error={exc}"
        )
        if not addressed_bot:
            return
        reply_candidates = (
            PendingApprovalCandidate(
                1,
                "人在。刚才这句没接稳，你直接说重点。",
                "reply",
                "模型异常时的兜底短回复",
            ),
        )
    if not reply_candidates:
        if addressed_bot:
            reply_candidates = (
                PendingApprovalCandidate(
                    1,
                    "我是个美少女人家不知道呢。",
                    "reply",
                    "空回复兜底，要求对方补清楚",
                ),
            )
            logger.info(f"qq_social_agent fallback reply: group={group_id} reason=empty_model_reply")
        else:
            logger.info(f"qq_social_agent skipped group={group_id}: empty_model_reply")
            return

    approval_candidates: list[PendingApprovalCandidate] = []
    for index, draft in enumerate(reply_candidates, start=1):
        candidate_text = draft.text
        if market_report:
            candidate_text = f"{market_report}\n{candidate_text}".strip()
        candidate_text, guarded = sanitize_political_output(candidate_text)
        if guarded:
            logger.info(f"qq_social_agent political guard output: group={group_id} candidate={index}")
        if not candidate_text:
            continue
        approval_candidates.append(
            PendingApprovalCandidate(
                index=index,
                text=candidate_text,
                action=draft.action,
                style=draft.style,
            )
        )
        if len(approval_candidates) >= 3:
            break
    if not approval_candidates:
        logger.info(f"qq_social_agent skipped group={group_id}: empty_candidate_after_guard")
        return
    logger.info(
        "qq_social_agent pending group reply candidates approval: "
        f"group={group_id} candidates={len(approval_candidates)}"
    )
    await _request_group_approval(
        bot,
        PendingGroupApproval(
            group_id=group_id,
            trigger_user_id=user_id,
            trigger_nickname=nickname,
            trigger_text=text,
            persona_name=persona.name,
            self_id=int(event.self_id),
            candidates=tuple(approval_candidates),
            mention_targets=mention_targets,
            created_at=time.time(),
        ),
    )


@private_message.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent) -> None:
    text = _plain_text(event)
    if not text:
        logger.info("qq_social_agent ignored private: empty_text")
        return

    user_id = int(event.user_id)
    if await _handle_group_approval_private(bot, user_id, text):
        return

    if not app_config.private_user_allowed(user_id):
        logger.info(f"qq_social_agent ignored private: user={user_id} not_allowed")
        return

    logger.info(f"qq_social_agent private start: user={user_id} text={text!r}")
    chat_id = _private_chat_id(user_id)
    nickname = _private_nickname(event)
    memory.add_message(chat_id, user_id, nickname, text, is_bot=False)

    state = memory.group_state(chat_id)
    if not bool(state["enabled"]):
        logger.info(f"qq_social_agent ignored private: user={user_id} disabled")
        return

    persona_id = str(state["persona"] or app_config.default_persona)
    persona = personas.get(persona_id)
    recent = memory.recent_messages(chat_id, app_config.context_limit)
    context_recent = _without_current_message(recent, user_id=user_id, text=text)
    market_intents = detect_market_intents(text, limit=2)
    rate = rate_limiter.allow(chat_id, mentioned=True)
    if not rate.allowed:
        logger.info(f"qq_social_agent suppressed private by rate: user={user_id} reason={rate.reason}")
        return

    if has_political_redline(text):
        logger.info(f"qq_social_agent political guard private input: user={user_id} text={text!r}")
        reply = political_safe_reply()
        try:
            await bot.send_private_msg(user_id=user_id, message=Message(reply))
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending political guard private reply: "
                f"user={user_id} {_action_failed_summary(exc)}"
            )
            return
        memory.add_message(chat_id, int(event.self_id), persona.name, reply, is_bot=True)
        return

    if deepseek_client is None:
        logger.warning("qq_social_agent skipped private: deepseek_client_not_ready")
        return

    memory_context = _format_memory_context(
        memory.recent_memory_summaries(chat_id, MID_MEMORY_KEEP_SUMMARIES)
    )
    market_context = await _market_context_for(market_intents, market_topic=bool(market_intents))
    fresh_context = await _private_fresh_context_for(text)
    try:
        reply = await deepseek_client.reply(
            persona=persona,
            recent_messages=context_recent,
            current_text=text,
            current_nickname=nickname,
            mentioned=True,
            chat_label="QQ 私聊",
            market_context=market_context,
            fresh_context=fresh_context,
            memory_context=memory_context,
            priority_context=_private_priority_context(user_id),
        )
    except Exception as exc:
        logger.warning(f"qq_social_agent private reply generation failed: user={user_id} error={exc}")
        reply = "我在。刚才模型没接上，你再发一遍重点。"
    if not reply:
        reply = "我是个美少女人家不知道呢。"
        logger.info(f"qq_social_agent fallback private reply: user={user_id} reason=empty_model_reply")
    reply, guarded = sanitize_political_output(reply)
    if guarded:
        logger.info(f"qq_social_agent political guard private output: user={user_id}")

    reply_parts = split_reply_messages(reply, max_messages=3)
    logger.info(
        "qq_social_agent sending private reply: "
        f"user={user_id} chars={len(reply)} parts={len(reply_parts)}"
    )
    for index, part in enumerate(reply_parts):
        try:
            await bot.send_private_msg(user_id=user_id, message=Message(part))
            memory.add_message(chat_id, int(event.self_id), persona.name, part, is_bot=True)
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending private reply: "
                f"user={user_id} {_action_failed_summary(exc)}"
            )
            return
        if index < len(reply_parts) - 1:
            await asyncio.sleep(0.9)


bot_command = on_command("bot", priority=10, block=True)


@bot_command.handle()
async def handle_bot_command(event: Event, matcher: Matcher, args: Message = CommandArg()) -> None:
    chat_id = _command_chat_id(event)
    if chat_id is None:
        return

    raw = args.extract_plain_text().strip()
    parts = raw.split()
    action = parts[0].lower() if parts else "status"

    if action == "pause":
        memory.set_group_enabled(chat_id, False)
        await matcher.finish("已暂停。")
    if action == "resume":
        memory.set_group_enabled(chat_id, True)
        await matcher.finish("已恢复。")
    if action == "reset":
        memory.reset_group_messages(chat_id)
        await matcher.finish("上下文已清空。")
    if action == "quiet":
        minutes = _parse_minutes(parts[1] if len(parts) >= 2 else "10m")
        memory.mute_until(chat_id, time.time() + minutes * 60)
        await matcher.finish(f"闭嘴 {minutes} 分钟。")
    if action == "persona":
        if len(parts) < 2:
            await matcher.finish("可用人格：" + ", ".join(personas.ids()))
        persona_id = parts[1]
        if not personas.has(persona_id):
            await matcher.finish("没有这个人格。可用：" + ", ".join(personas.ids()))
        memory.set_group_persona(chat_id, persona_id)
        await matcher.finish(f"人格已切换：{persona_id}")
    if action == "status":
        state = memory.group_state(chat_id)
        group_cfg = app_config.group_config(chat_id) if isinstance(event, GroupMessageEvent) else {}
        persona_id = str(state["persona"] or group_cfg.get("persona") or app_config.default_persona)
        enabled = bool(group_cfg.get("enabled", True)) and bool(state["enabled"])
        muted_left = max(0, int(float(state["muted_until"]) - time.time()))
        await matcher.finish(
            f"enabled={enabled} persona={persona_id} muted_left={muted_left}s "
            f"decision_model={app_config.deepseek.decision_model} "
            f"reply_model={app_config.deepseek.reply_model} "
            f"utility_model={app_config.deepseek.utility_model}"
        )
    if action in {"tokens", "token", "usage"}:
        window = _parse_token_report_window(parts[1] if len(parts) >= 2 else "")
        await matcher.finish(
            _token_usage_report_for_window(window)
        )

    await matcher.finish("用法：/bot status|tokens 24h|tokens 2026-07-10|pause|resume|reset|quiet 10m|persona <id>")


def _parse_token_report_window(raw: str) -> TokenReportWindow:
    text = raw.strip().lower()
    if not text or text in {"24h", "day"}:
        return _relative_token_report_window(TOKEN_REPORT_DEFAULT_WINDOW_SECONDS, "近 24 小时")
    if text in {"today", "今天", "今日"}:
        return _date_token_report_window(time.localtime().tm_year, time.localtime().tm_mon, time.localtime().tm_mday)
    if text in {"yesterday", "昨天", "昨日"}:
        local_now = time.localtime(time.time() - 24 * 60 * 60)
        return _date_token_report_window(local_now.tm_year, local_now.tm_mon, local_now.tm_mday)
    if text in {"all", "全部", "total"}:
        return TokenReportWindow(None, None, "全部")
    date_match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if date_match is not None:
        return _date_token_report_window(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        )
    match = re.fullmatch(r"(\d+)([hdw天周]?)", text)
    if match is None:
        return _relative_token_report_window(TOKEN_REPORT_DEFAULT_WINDOW_SECONDS, "近 24 小时")
    value = max(1, int(match.group(1)))
    unit = match.group(2)
    if unit in {"h", ""}:
        return _relative_token_report_window(value * 60 * 60, f"近 {value} 小时")
    if unit in {"d", "天"}:
        return _relative_token_report_window(value * 24 * 60 * 60, f"近 {value} 天")
    return _relative_token_report_window(value * 7 * 24 * 60 * 60, f"近 {value} 周")


def _relative_token_report_window(seconds: int, label: str) -> TokenReportWindow:
    return TokenReportWindow(time.time() - seconds, None, label)


def _date_token_report_window(year: int, month: int, day: int) -> TokenReportWindow:
    try:
        start_struct = time.strptime(f"{year:04d}-{month:02d}-{day:02d}", "%Y-%m-%d")
    except ValueError:
        return _relative_token_report_window(TOKEN_REPORT_DEFAULT_WINDOW_SECONDS, "近 24 小时")
    start_at = time.mktime(start_struct)
    end_at = start_at + 24 * 60 * 60
    return TokenReportWindow(start_at, end_at, f"{year:04d}-{month:02d}-{day:02d}")


def _parse_approval_token_report_command(text: str) -> TokenReportWindow | None:
    compact = text.strip()
    if not compact:
        return None
    parts = compact.split(maxsplit=1)
    head = parts[0].casefold()
    if head in TOKEN_REPORT_COMMAND_ALIASES:
        return _parse_token_report_window(parts[1] if len(parts) >= 2 else "")
    for alias in TOKEN_REPORT_COMMAND_ALIASES:
        if compact.casefold().startswith(alias.casefold()):
            raw_window = compact[len(alias) :].strip(" ：:")
            return _parse_token_report_window(raw_window)
    return None


def _token_usage_report_for_window(window: TokenReportWindow) -> str:
    imported = _backfill_llm_usage_from_logs()
    if imported:
        logger.info(f"qq_social_agent imported llm usage from logs: rows={imported}")
    return _format_token_usage_report(
        summaries=memory.llm_usage_summary(start_at=window.start_at, end_at=window.end_at),
        recent_events=memory.recent_llm_usage_events(
            start_at=window.start_at,
            end_at=window.end_at,
            limit=TOKEN_REPORT_MAX_RECENT_EVENTS,
        ),
        label=window.label,
    )


def _backfill_llm_usage_from_logs() -> int:
    imported = 0
    current_year = time.localtime().tm_year
    for path in TOKEN_USAGE_LOG_BACKFILL_FILES:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    parsed = _parse_llm_usage_log_line(line, year=current_year)
                    if parsed is None:
                        continue
                    task, model, prompt_tokens, completion_tokens, total_tokens, created_at = parsed
                    digest = hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:16]
                    source_key = f"log:{path.name}:{line_no}:{digest}"
                    if memory.add_llm_usage(
                        task=task,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        created_at=created_at,
                        source_key=source_key,
                    ):
                        imported += 1
        except OSError as exc:
            logger.warning(f"qq_social_agent failed reading llm usage log: path={path} error={exc}")
    return imported


def _parse_llm_usage_log_line(
    line: str,
    *,
    year: int,
) -> tuple[str, str, int | None, int | None, int | None, float] | None:
    match = LLM_USAGE_LOG_RE.match(line.strip())
    if match is None:
        return None
    timestamp_text = (
        f"{year:04d}-{match.group('month')}-{match.group('day')} "
        f"{match.group('hms')}"
    )
    try:
        created_at = time.mktime(time.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None
    return (
        match.group("task"),
        match.group("model"),
        _optional_usage_int(match.group("prompt")),
        _optional_usage_int(match.group("completion")),
        _optional_usage_int(match.group("total")),
        created_at,
    )


def _optional_usage_int(value: str) -> int | None:
    if value == "None":
        return None
    return int(value)


def _format_token_usage_report(
    *,
    summaries: list[LLMUsageSummary],
    recent_events: list[LLMUsageEvent],
    label: str,
) -> str:
    if not summaries:
        return f"Token 用量报告（{label}）：暂无记录。"
    total_calls = sum(item.call_count for item in summaries)
    total_prompt = sum(item.prompt_tokens for item in summaries)
    total_completion = sum(item.completion_tokens for item in summaries)
    total_tokens = sum(item.total_tokens for item in summaries)
    total_cost = sum(
        _estimate_llm_cost_cny(item.model, item.prompt_tokens, item.completion_tokens)
        for item in summaries
    )
    lines = [
        f"Token 用量报告（{label}）",
        f"总调用：{total_calls} 次",
        f"总 token：{total_tokens}（输入 {total_prompt} / 输出 {total_completion}）",
        f"估算成本：{_format_cny(total_cost)}（按输入缓存未命中估算，实际可能更低）",
        "",
        "按任务/模型：",
    ]
    for item in summaries[:12]:
        cost = _estimate_llm_cost_cny(item.model, item.prompt_tokens, item.completion_tokens)
        lines.append(
            f"- {item.task} / {item.model}：{item.call_count} 次，"
            f"{item.total_tokens} token（入 {item.prompt_tokens} / 出 {item.completion_tokens}），"
            f"{_format_cny(cost)}"
        )
    if recent_events:
        lines.append("")
        lines.append("最近调用：")
        for event in recent_events:
            lines.append(
                f"- {_format_time(event.created_at)} {event.task}/{event.model} "
                f"{event.total_tokens} token（入 {event.prompt_tokens} / 出 {event.completion_tokens}）"
            )
    return "\n".join(lines)


def _estimate_llm_cost_cny(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_key = model.casefold()
    if "pro" in model_key:
        prompt_price = 3.0
        completion_price = 6.0
    else:
        prompt_price = 1.0
        completion_price = 2.0
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


def _format_cny(value: float) -> str:
    if value < 0.01:
        return f"{value:.4f} 元"
    return f"{value:.2f} 元"


def _format_time(timestamp: float) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(timestamp))


def _plain_text(event: GroupMessageEvent) -> str:
    text = event.get_plaintext().strip()
    return re.sub(r"\s+", " ", text)


def _event_plain_text(event: GroupMessageEvent | PrivateMessageEvent) -> str:
    text = event.get_plaintext().strip()
    return re.sub(r"\s+", " ", text)


def _jargon_command_group_id(event: Event) -> int | None:
    if isinstance(event, GroupMessageEvent):
        group_id = int(event.group_id)
        if not app_config.group_allowed(group_id):
            return None
        return group_id
    if isinstance(event, PrivateMessageEvent):
        return _private_jargon_group_id()
    return None


def _private_jargon_group_id() -> int | None:
    allowed_groups = sorted(app_config.allowed_groups)
    if len(allowed_groups) == 1:
        return allowed_groups[0]
    return None


def _is_jargon_command_text(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("/黑话") or stripped.startswith("/删黑话")


def _handle_jargon_command_text(
    *,
    user_id: int,
    group_id: int | None,
    text: str,
) -> str:
    if user_id not in JARGON_COMMAND_USER_IDS:
        return "没权限。"
    if group_id is None:
        return "没找到要写入的群。"

    if JARGON_LIST_RE.match(text):
        entries = memory.custom_jargon_entries(group_id)
        if not entries:
            return "暂无自定义黑话。"
        return _format_custom_jargon_list(entries)

    delete_match = JARGON_DELETE_RE.match(text)
    if delete_match is not None:
        term = delete_match.group("term").strip()
        if not term:
            return "格式：/删黑话：词"
        deleted = memory.delete_custom_jargon(group_id, term)
        return "已删。" if deleted else "没找到这条自定义黑话。"

    add_match = JARGON_ADD_RE.match(text)
    if add_match is None:
        return "格式：/黑话：咱妈 指代：中国"
    term = add_match.group("term").strip()
    meaning = add_match.group("meaning").strip()
    if not term or not meaning:
        return "格式：/黑话：咱妈 指代：中国"
    memory.upsert_custom_jargon(
        group_id=group_id,
        term=term,
        explanation=f"指代：{meaning}",
        created_by=user_id,
    )
    return f"已记黑话：{term} -> {meaning}"


def _format_custom_jargon_list(entries: list[CustomJargonEntry]) -> str:
    lines = ["自定义黑话："]
    for entry in entries[:40]:
        lines.append(f"- {entry.term}：{entry.explanation}")
    return "\n".join(lines)


def _action_failed_summary(exc: ActionFailed) -> str:
    retcode = getattr(exc, "retcode", None)
    message = getattr(exc, "message", None)
    if retcode is None:
        retcode = getattr(exc, "code", None)
    if message is None:
        message = getattr(exc, "wording", None)
    return f"retcode={retcode or 'unknown'} message={message or str(exc)!r}"


def _nickname(event: GroupMessageEvent) -> str:
    sender = event.sender
    return sender.card or sender.nickname or str(event.user_id)


def _private_nickname(event: PrivateMessageEvent) -> str:
    sender = event.sender
    return sender.nickname or str(event.user_id)


def _private_chat_id(user_id: int) -> int:
    return PRIVATE_CHAT_OFFSET + user_id


async def _market_context_for(intents: list[MarketIntent], *, market_topic: bool) -> str:
    if not intents:
        if market_topic:
            return (
                "市场工具提示：用户在聊美股、加密货币或看盘，但没有给出具体标的。"
                "回复时让对方报 ticker 或币种，例如 NVDA、TSLA、BTC、ETH；不要编造行情。"
            )
        return ""
    context = await market_tool.context_for(intents)
    logger.info(
        "qq_social_agent market tool: "
        f"intents={[(intent.kind, intent.symbol) for intent in intents]} "
        f"has_context={bool(context)}"
    )
    return context


async def _market_report_and_context_for(
    intents: list[MarketIntent],
    *,
    market_topic: bool,
) -> tuple[str, str]:
    if not intents:
        if market_topic:
            text = "没看见具体标的，报 ticker 或币种我才能查，比如 NVDA、TSLA、BTC、ETH。"
            context = (
                "市场工具提示：用户在聊美股、加密货币或看盘，但没有给出具体标的。"
                "已提示对方报 ticker 或币种；不要编造行情。"
            )
            return text, context
        return "", ""

    report, context = await market_tool.report_and_context_for(intents)
    logger.info(
        "qq_social_agent market tool: "
        f"intents={[(intent.kind, intent.symbol) for intent in intents]} "
        f"has_report={bool(report)} has_context={bool(context)}"
    )
    return report, context


async def _fresh_context_for(decision: ReplyDecision, *, fallback_text: str) -> str:
    query = decision.fresh_query.strip() or fallback_text.strip()
    if not query:
        logger.info(
            "qq_social_agent fresh context skipped: "
            f"query={query!r} fallback={fallback_text!r}"
        )
        return ""
    context = await fresh_context_tool.context_for(query, kind=decision.fresh_kind)
    logger.info(
        "qq_social_agent fresh context: "
        f"kind={decision.fresh_kind} query={query!r} has_context={bool(context)}"
    )
    return context


def _format_fresh_context_hint(intent: object | None) -> str:
    if intent is None:
        return ""
    query = str(getattr(intent, "query", "") or "").strip()
    kind = str(getattr(intent, "kind", "news") or "news").strip()
    if not query:
        return ""
    return (
        f"后端检测到这句话可能涉及最新背景，候选查询：{query}，类型：{kind}。"
        "是否真的需要搜索由你判断；非必要不要搜索。"
    )


async def _private_fresh_context_for(text: str) -> str:
    intent = detect_fresh_intent(text)
    if intent is None:
        return ""
    context = await fresh_context_tool.context_for(intent.query, kind=intent.kind)
    logger.info(
        "qq_social_agent private fresh context: "
        f"kind={intent.kind} query={intent.query!r} has_context={bool(context)}"
    )
    return context


def _market_intents_from_decision(
    decision: ReplyDecision,
    *,
    fallback_text: str,
    fallback_intents: list[MarketIntent],
) -> list[MarketIntent]:
    intents: list[MarketIntent] = []
    seen: set[tuple[str, str]] = set()

    for symbol in decision.symbols:
        detected = detect_market_intents(f"{symbol.display} {symbol.symbol}", limit=1)
        if detected:
            _append_market_intent(intents, seen, detected[0])
            continue
        _append_market_intent(
            intents,
            seen,
            MarketIntent(symbol.kind, symbol.symbol, symbol.display or symbol.symbol),
        )

    if not intents:
        for intent in fallback_intents:
            _append_market_intent(intents, seen, intent)

    if not intents:
        for intent in detect_market_intents(fallback_text, limit=2):
            _append_market_intent(intents, seen, intent)

    return intents[:2]


def _append_market_intent(
    intents: list[MarketIntent],
    seen: set[tuple[str, str]],
    intent: MarketIntent,
) -> None:
    key = (intent.kind, intent.symbol)
    if key in seen or len(intents) >= 2:
        return
    seen.add(key)
    intents.append(intent)


def _buffer_group_message(bot: Bot, event: GroupMessageEvent, text: str) -> None:
    group_id = int(event.group_id)
    item = BufferedGroupMessage(
        bot=bot,
        event=event,
        text=text,
        user_id=int(event.user_id),
        nickname=_nickname(event),
        created_at=float(getattr(event, "time", 0) or time.time()),
    )
    group_message_buffers.setdefault(group_id, []).append(item)
    task = group_buffer_tasks.get(group_id)
    if task is None or task.done():
        group_buffer_tasks[group_id] = asyncio.create_task(_flush_group_buffer_after_delay(group_id))
    logger.info(
        "qq_social_agent buffered group message: "
        f"group={group_id} size={len(group_message_buffers.get(group_id, []))}"
    )


async def _flush_group_buffer_after_delay(group_id: int) -> None:
    try:
        await asyncio.sleep(GROUP_BUFFER_SECONDS)
        async with _group_processing_lock(group_id):
            items = group_message_buffers.pop(group_id, [])
            if not items:
                return
            logger.info(
                "qq_social_agent flushing group buffer: "
                f"group={group_id} size={len(items)}"
            )
            latest = items[-1]
            await _handle_group_message_locked(latest.bot, latest.event, buffered_messages=items)
    finally:
        task = asyncio.current_task()
        if group_buffer_tasks.get(group_id) is task:
            group_buffer_tasks.pop(group_id, None)


def _buffered_current_text(items: list[BufferedGroupMessage] | None) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0].text
    recent_items = items[-6:]
    lines = [f"{item.nickname}: {item.text}" for item in recent_items if item.text]
    if len(items) > len(recent_items):
        lines.insert(0, f"（前面还有 {len(items) - len(recent_items)} 条普通群消息）")
    return "\n".join(lines).strip()


def _buffered_current_user_id(items: list[BufferedGroupMessage] | None) -> int:
    if not items:
        return 0
    user_ids = {item.user_id for item in items}
    if len(user_ids) == 1:
        return items[-1].user_id
    return items[-1].user_id


def _buffered_current_nickname(items: list[BufferedGroupMessage] | None) -> str:
    if not items:
        return "群友"
    nicknames = {item.nickname for item in items}
    if len(nicknames) == 1:
        return items[-1].nickname
    return "群友们"


def _buffered_first_created_at(items: list[BufferedGroupMessage] | None) -> float:
    if not items:
        return time.time()
    return items[0].created_at


def _buffered_last_created_at(items: list[BufferedGroupMessage] | None) -> float:
    if not items:
        return time.time()
    return items[-1].created_at


def _passive_decision_allowed(
    group_id: int,
    *,
    message_count: int,
    first_message_at: float,
    last_message_at: float,
) -> tuple[bool, str]:
    previous_at, waiting_count = group_passive_decision_state.get(group_id, (0.0, 0))
    current_count = max(1, message_count)
    if previous_at <= 0 or first_message_at - previous_at >= GROUP_PASSIVE_DECISION_GAP_SECONDS:
        group_passive_decision_state[group_id] = (last_message_at, 0)
        return True, "gap_first_message"

    waiting_count += current_count
    if waiting_count >= GROUP_PASSIVE_DECISION_EVERY_MESSAGES:
        group_passive_decision_state[group_id] = (
            last_message_at,
            waiting_count % GROUP_PASSIVE_DECISION_EVERY_MESSAGES,
        )
        return True, "every_three_messages"

    group_passive_decision_state[group_id] = (last_message_at, waiting_count)
    return False, f"waiting_{waiting_count}/{GROUP_PASSIVE_DECISION_EVERY_MESSAGES}"


def _mark_passive_decision_forced(group_id: int, *, now: float | None = None) -> None:
    group_passive_decision_state[group_id] = (time.time() if now is None else now, 0)


def _group_processing_lock(group_id: int) -> asyncio.Lock:
    lock = group_processing_locks.get(group_id)
    if lock is None:
        lock = asyncio.Lock()
        group_processing_locks[group_id] = lock
    return lock


def _schedule_group_learning(group_id: int) -> None:
    if deepseek_client is None:
        return
    task = group_learning_tasks.get(group_id)
    if task is not None and not task.done():
        return
    group_learning_tasks[group_id] = asyncio.create_task(_run_group_learning(group_id))


async def _run_group_learning(group_id: int) -> None:
    task = asyncio.current_task()
    try:
        await _maintain_group_learning(group_id)
    except Exception as exc:
        logger.warning(f"qq_social_agent group learning task failed: group={group_id} error={exc}")
    finally:
        if group_learning_tasks.get(group_id) is task:
            group_learning_tasks.pop(group_id, None)


async def _maintain_group_learning(group_id: int) -> None:
    if deepseek_client is None:
        return

    mid_messages = memory.messages_for_mid_summary(
        group_id,
        keep_recent=app_config.context_limit,
        batch_size=MID_MEMORY_BATCH_SIZE,
    )
    if (
        len(mid_messages) >= MID_MEMORY_MIN_BATCH
        and time.time() - last_mid_memory_attempt.get(group_id, 0.0)
        >= MID_MEMORY_RETRY_INTERVAL_SECONDS
    ):
        last_mid_memory_attempt[group_id] = time.time()
        try:
            summary_messages = [msg for msg in mid_messages if not msg.is_bot]
            draft = None
            if len(summary_messages) >= MID_MEMORY_MIN_BATCH:
                draft = await deepseek_client.summarize_mid_memory(
                    messages=summary_messages,
                    chat_label="QQ 群聊",
                )
            if draft and draft.summary:
                memory.add_memory_summary(
                    group_id,
                    mid_messages,
                    summary=draft.summary,
                    recall_cues=list(draft.recall_cues),
                )
                logger.info(
                    "qq_social_agent mid memory summarized: "
                    f"group={group_id} messages={len(mid_messages)} cues={len(draft.recall_cues)}"
                )
        except Exception as exc:
            logger.warning(f"qq_social_agent mid memory skipped: group={group_id} error={exc}")

    last_attempt = max(
        memory.last_style_rule_at(group_id),
        last_style_learn_attempt.get(group_id, 0.0),
    )
    if time.time() - last_attempt < STYLE_LEARN_INTERVAL_SECONDS:
        return
    style_messages = memory.messages_for_style_learning(
        group_id,
        limit=STYLE_LEARN_MESSAGE_LIMIT,
    )
    if len(style_messages) < STYLE_LEARN_MIN_MESSAGES:
        return
    last_style_learn_attempt[group_id] = time.time()
    try:
        rules = await deepseek_client.learn_style_rules(
            messages=style_messages,
            chat_label="QQ 群聊",
        )
        useful_rules = [
            rule
            for rule in rules
            if _is_useful_style_rule(rule.situation, rule.style, rule.source_text)
        ]
        memory.add_style_rules(
            group_id,
            [
                (rule.situation, rule.style, rule.source_text)
                for rule in useful_rules
            ],
        )
        if useful_rules:
            logger.info(
                "qq_social_agent style rules learned: "
                f"group={group_id} rules={len(useful_rules)}"
            )
    except Exception as exc:
        logger.warning(f"qq_social_agent style learning skipped: group={group_id} error={exc}")


def _format_memory_context(summaries: list[MemorySummary]) -> str:
    if not summaries:
        return ""
    lines: list[str] = []
    for index, summary in enumerate(summaries, start=1):
        cues = "；".join(summary.recall_cues[:3])
        if cues:
            lines.append(f"{index}. {summary.summary}（线索：{cues}）")
        else:
            lines.append(f"{index}. {summary.summary}")
    return "\n".join(lines)


def _format_recall_feedback_context(feedback_items: list[RecalledReplyFeedback]) -> str:
    if not feedback_items:
        return ""
    lines: list[str] = []
    for item in feedback_items:
        if "owner_feedback" in item.tags:
            lines.append(f"- 主人原始评价：{item.owner_reason}")
            continue
        tags = f"；标签：{'、'.join(item.tags[:3])}" if item.tags else ""
        lines.append(
            f"- 场景：{item.scene_summary}\n"
            f"  问题：{item.bad_reply_problem}\n"
            f"  避免：{item.avoid_rule}\n"
            f"  更好方向：{item.better_direction}{tags}"
        )
    return "\n".join(lines)


def _format_positive_feedback_context(feedback_items: list[ApprovedReplyFeedback]) -> str:
    if not feedback_items:
        return ""
    lines: list[str] = []
    for item in feedback_items:
        trigger = item.trigger_text.strip().replace("\n", " ")[:36]
        style = item.style.strip() or "自然群聊接话"
        lines.append(
            f"- 触发“{trigger}”时，审批人认可的方向：{style}；"
            "只学习策略，禁止照搬原回复。"
        )
    return "\n".join(lines)


def _format_style_context(rules: list[StyleRule]) -> str:
    if not rules:
        return ""
    lines = [
        f"- 当{rule.situation}时，可以{rule.style}"
        for rule in rules
        if _is_useful_style_rule(rule.situation, rule.style, rule.source_text)
    ]
    return "\n".join(lines)


def _related_member_user_ids(
    recent_messages: list[ChatMessage],
    *,
    current_user_id: int,
) -> list[int]:
    user_ids = [current_user_id]
    for msg in reversed(recent_messages):
        if msg.is_bot:
            continue
        user_ids.append(msg.user_id)
        if len(user_ids) >= 12:
            break
    return user_ids


def _format_member_context(profiles: list[MemberProfile]) -> str:
    if not profiles:
        return ""
    lines: list[str] = []
    for profile in profiles:
        label = _member_label(profile.user_id, profile.display_name)
        aliases = [
            alias
            for alias in profile.aliases
            if alias and alias != profile.display_name
        ][:3]
        if aliases:
            lines.append(f"- {label}，曾用名/历史名：{'、'.join(aliases)}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _member_label(user_id: int, nickname: str) -> str:
    clean_name = nickname.strip() or str(user_id)
    return f"{clean_name}[#{str(user_id)[-5:]}]"


def _format_cue_repeat_context(state: CueRepeatState | None) -> str:
    if state is None:
        return ""
    if state.count < 3:
        return f"{state.label}，10 分钟内第 {state.count} 次；暂时按正常互动处理。"
    return (
        f"{state.label}，10 分钟内第 {state.count} 次。"
        "这已经有问答机/客服感：不要继续按题作答，优先吐槽题型、反问、嫌他无聊，"
        "必要时只顺带给一个很短判断。"
    )


async def _selected_group_jargon_context(
    group_id: int,
    recent_messages: list[ChatMessage],
    *,
    current_text: str,
    current_nickname: str,
) -> str:
    custom_entries = _matched_custom_group_jargon_entries(group_id, [current_text])
    heuristic_terms = detect_group_jargon_terms([current_text], extra_entries=custom_entries)
    if not heuristic_terms:
        logger.info(
            "qq_social_agent jargon selector: "
            "heuristic=() selected=() injected=False skipped=no_current_hit"
        )
        return ""
    if deepseek_client is None:
        return group_jargon_context(heuristic_terms, extra_entries=custom_entries)
    try:
        selected_terms = await deepseek_client.select_jargon_terms(
            recent_messages=recent_messages[-JARGON_CONTEXT_LOOKBACK:],
            current_text=current_text,
            current_nickname=current_nickname,
            jargon_catalog=group_jargon_catalog(extra_entries=custom_entries),
            heuristic_terms=heuristic_terms,
            chat_label="QQ 群聊",
        )
    except Exception as exc:
        logger.warning(f"qq_social_agent jargon selector skipped: error={exc}")
        selected_terms = heuristic_terms
    if not selected_terms and heuristic_terms:
        selected_terms = heuristic_terms
    context = group_jargon_context(selected_terms, extra_entries=custom_entries)
    logger.info(
        "qq_social_agent jargon selector: "
        f"heuristic={heuristic_terms} selected={selected_terms} injected={bool(context)}"
    )
    return context


def _custom_group_jargon_entries(group_id: int) -> tuple[GroupJargonEntry, ...]:
    return tuple(_custom_jargon_entry_to_group_jargon(entry) for entry in memory.custom_jargon_entries(group_id))


def _matched_custom_group_jargon_entries(
    group_id: int,
    texts: list[str],
) -> tuple[GroupJargonEntry, ...]:
    haystack = "\n".join(text for text in texts if text).casefold()
    if not haystack:
        return ()
    entries: list[GroupJargonEntry] = []
    for entry in memory.custom_jargon_entries(group_id):
        term = entry.term.strip()
        if not term or term.casefold() not in haystack:
            continue
        entries.append(_custom_jargon_entry_to_group_jargon(entry))
        if len(entries) >= CUSTOM_JARGON_CONTEXT_LIMIT:
            break
    return tuple(entries)


def _custom_jargon_entry_to_group_jargon(entry: CustomJargonEntry) -> GroupJargonEntry:
    key = f"custom:{entry.term.casefold()}"
    return GroupJargonEntry(key, (entry.term,), entry.explanation)


async def _request_group_approval(bot: Bot, approval: PendingGroupApproval) -> None:
    pending_group_approvals[approval.group_id] = approval
    preview = _format_approval_candidates(approval)
    message = (
        f"待发群：{approval.group_id}\n"
        f"触发人：{_member_label(approval.trigger_user_id, approval.trigger_nickname)}\n"
        f"触发消息：{approval.trigger_text}\n\n"
        f"候选回复：\n{preview}\n\n"
        "指令：\n"
        "- AI 默认把最想发的放在 1\n"
        "- 1/2/3：发送对应候选\n"
        "- 1!/2!/3!：发送并标记优质\n"
        "- 准奏：发送 1\n"
        "- 不准奏原因：xxx：默认批评 1\n"
        "- 不准奏2原因：xxx：批评指定候选\n"
        "- 审批规则详情：展开完整说明"
    )
    delivered = 0
    for approver_id in GROUP_APPROVAL_USER_IDS:
        try:
            await bot.send_private_msg(user_id=approver_id, message=Message(message))
            delivered += 1
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending group approval request: "
                f"approver={approver_id} group={approval.group_id} {_action_failed_summary(exc)}"
            )
    if delivered <= 0:
        pending_group_approvals.pop(approval.group_id, None)
        return
    logger.info(
        "qq_social_agent group approval pending: "
        f"approvers={GROUP_APPROVAL_USER_IDS} group={approval.group_id} candidates={len(approval.candidates)}"
    )


def _format_approval_candidates(approval: PendingGroupApproval) -> str:
    lines: list[str] = []
    for candidate in approval.candidates:
        style = candidate.style.strip()
        style_line = f"\n   style：{style}" if style else ""
        lines.append(f"{candidate.index}. {candidate.text}{style_line}")
    return "\n\n".join(lines).strip()


def _approval_candidate_by_index(
    approval: PendingGroupApproval,
    index: int,
) -> PendingApprovalCandidate | None:
    for candidate in approval.candidates:
        if candidate.index == index:
            return candidate
    return None


def _latest_group_approval() -> PendingGroupApproval | None:
    if not pending_group_approvals:
        return None
    return max(pending_group_approvals.values(), key=lambda approval: approval.created_at)


async def _set_approval_group_decision_enabled(bot: Bot, user_id: int, enabled: bool) -> None:
    target_groups = sorted(app_config.allowed_groups) or sorted(pending_group_approvals)
    for group_id in target_groups:
        memory.set_group_enabled(group_id, enabled)
    if not enabled:
        pending_group_approvals.clear()
    response_text = "已开启，群聊恢复进入决策。" if enabled else "已关闭，群聊不再进入决策，待审候选已清空。"
    try:
        await bot.send_private_msg(user_id=user_id, message=Message(response_text))
    except ActionFailed:
        pass
    await _send_approval_rules_to_approvers(bot, reason="decision_switch")
    logger.info(
        "qq_social_agent approval decision switch: "
        f"approver={user_id} enabled={enabled} groups={target_groups}"
    )


async def _handle_group_approval_private(bot: Bot, user_id: int, text: str) -> bool:
    if user_id not in GROUP_APPROVAL_USER_IDS:
        return False
    compact_text = text.strip()
    if _is_jargon_command_text(compact_text):
        await _send_private_text(
            bot,
            user_id,
            _handle_jargon_command_text(
                user_id=user_id,
                group_id=_private_jargon_group_id(),
                text=compact_text,
            ),
        )
        return True
    if compact_text in APPROVAL_HELP_COMMANDS:
        await _send_private_text(bot, user_id, APPROVAL_RULES_MESSAGE)
        return True
    if compact_text in APPROVAL_DETAIL_COMMANDS:
        await _send_private_text(bot, user_id, APPROVAL_RULES_DETAIL_MESSAGE)
        return True
    token_report_window = _parse_approval_token_report_command(compact_text)
    if token_report_window is not None:
        await _send_private_text(
            bot,
            user_id,
            _token_usage_report_for_window(token_report_window),
        )
        return True
    if compact_text in {"开启", "打开", "恢复"}:
        await _set_approval_group_decision_enabled(bot, user_id, True)
        return True
    if compact_text in {"关闭", "关掉", "暂停"}:
        await _set_approval_group_decision_enabled(bot, user_id, False)
        return True
    approval = _latest_group_approval()
    if approval is None:
        return False
    pending_group_approvals.pop(approval.group_id, None)
    candidate: PendingApprovalCandidate | None = None
    high_quality = False
    choice_match = APPROVAL_CHOICE_RE.match(compact_text)
    if choice_match is not None:
        candidate = _approval_candidate_by_index(approval, int(choice_match.group(1)))
        high_quality = bool(choice_match.group(2))
    elif compact_text == "准奏":
        candidate = approval.candidates[0] if approval.candidates else None
    if candidate is None:
        reason_match = APPROVAL_REJECT_REASON_RE.match(compact_text)
        if reason_match is not None:
            owner_reason = reason_match.group("reason").strip()
            reject_index = int(reason_match.group("index") or "1")
            rejected_candidate = _approval_candidate_by_index(approval, reject_index)
            if owner_reason:
                _save_approval_rejection_feedback(
                    approval,
                    owner_reason,
                    reason_user_id=user_id,
                    candidate=rejected_candidate,
                    candidate_index=reject_index,
                )
                response_text = "已取消，并记录不准奏原因。"
            else:
                response_text = "已取消。不准奏原因是空的，没写入反馈。"
        else:
            response_text = "已取消。"
        logger.info(
            "qq_social_agent group approval canceled: "
            f"approver={user_id} group={approval.group_id} text={text!r}"
        )
        try:
            await bot.send_private_msg(user_id=user_id, message=Message(response_text))
        except ActionFailed:
            pass
        return True
    await _send_approved_group_reply(
        bot,
        approval,
        candidate,
        approver_id=user_id,
        high_quality=high_quality,
    )
    return True


def _save_approval_rejection_feedback(
    approval: PendingGroupApproval,
    owner_reason: str,
    *,
    reason_user_id: int,
    candidate: PendingApprovalCandidate | None = None,
    candidate_index: int = 1,
) -> None:
    selected_candidate = candidate or (approval.candidates[0] if approval.candidates else None)
    bot_reply = (
        _memory_text_from_reply_part(selected_candidate.text, approval.mention_targets)
        if selected_candidate is not None
        else _format_approval_candidates(approval)
    )
    action = selected_candidate.action if selected_candidate is not None else "unknown"
    now = time.time()
    memory.add_recalled_reply_feedback(
        group_id=approval.group_id,
        message_id=0,
        bot_reply=bot_reply,
        trigger_user_id=approval.trigger_user_id,
        trigger_nickname=approval.trigger_nickname,
        trigger_text=approval.trigger_text,
        action=action,
        owner_reason=owner_reason,
        scene_summary=f"审批不准奏原始评价，针对第 {candidate_index} 条候选",
        bad_reply_problem=owner_reason,
        avoid_rule=owner_reason,
        better_direction=owner_reason,
        tags=["owner_feedback"],
        operator_id=reason_user_id,
        reason_user_id=reason_user_id,
        recalled_at=approval.created_at,
        reason_at=now,
    )
    logger.info(
        "qq_social_agent approval rejection feedback saved: "
        f"group={approval.group_id} approver={reason_user_id} reason={owner_reason!r}"
    )


def _save_approved_reply_feedback(
    approval: PendingGroupApproval,
    candidate: PendingApprovalCandidate,
    *,
    approver_id: int,
) -> None:
    memory.add_approved_reply_feedback(
        group_id=approval.group_id,
        candidate_text=_memory_text_from_reply_part(candidate.text, approval.mention_targets),
        trigger_user_id=approval.trigger_user_id,
        trigger_nickname=approval.trigger_nickname,
        trigger_text=approval.trigger_text,
        action=candidate.action,
        style=candidate.style,
        operator_id=approver_id,
    )
    logger.info(
        "qq_social_agent approved reply feedback saved: "
        f"group={approval.group_id} approver={approver_id} candidate={candidate.index}"
    )


async def _send_approved_group_reply(
    bot: Bot,
    approval: PendingGroupApproval,
    candidate: PendingApprovalCandidate,
    *,
    approver_id: int,
    high_quality: bool,
) -> None:
    logger.info(
        "qq_social_agent group approval accepted: "
        f"approver={approver_id} group={approval.group_id} candidate={candidate.index} high_quality={high_quality}"
    )
    reply_parts = split_reply_messages(candidate.text, max_messages=3)
    sent_mention_user_id: int | None = None
    recorded_user_reply = False
    for index, part_text in enumerate(reply_parts):
        try:
            part_mention_user_id = _first_allowed_mention_id(part_text, approval.mention_targets)
            sent_message_id = await _send_group_message(
                bot,
                approval.group_id,
                _message_from_reply_part(part_text, approval.mention_targets),
            )
            if not recorded_user_reply:
                _record_user_reply(approval.group_id, approval.trigger_user_id)
                recorded_user_reply = True
            memory_text = _memory_text_from_reply_part(part_text, approval.mention_targets)
            _record_bot_sent_message(
                group_id=approval.group_id,
                message_id=sent_message_id,
                bot_reply=memory_text,
                trigger_user_id=approval.trigger_user_id,
                trigger_nickname=approval.trigger_nickname,
                trigger_text=approval.trigger_text,
                action=candidate.action,
            )
            if sent_mention_user_id is None and part_mention_user_id is not None:
                sent_mention_user_id = part_mention_user_id
            memory.add_message(
                approval.group_id,
                approval.self_id,
                approval.persona_name,
                memory_text,
                is_bot=True,
            )
        except ActionFailed as exc:
            logger.warning(
                "qq_social_agent failed sending approved group reply: "
                f"group={approval.group_id} {_action_failed_summary(exc)}"
            )
            try:
                await bot.send_private_msg(
                    user_id=approver_id,
                    message=Message(f"发送失败：{_action_failed_summary(exc)}"),
                )
            except ActionFailed:
                pass
            return
        if index < len(reply_parts) - 1:
            await asyncio.sleep(0.9)
    if sent_mention_user_id is not None:
        last_group_mention_targets[approval.group_id] = (sent_mention_user_id, time.time())
    else:
        last_group_mention_targets.pop(approval.group_id, None)
    if high_quality:
        _save_approved_reply_feedback(approval, candidate, approver_id=approver_id)
    try:
        await bot.send_private_msg(user_id=approver_id, message=Message("已发。"))
    except ActionFailed:
        pass


async def _send_group_message(bot: Bot, group_id: int, message: Message) -> int | None:
    result = await bot.send_group_msg(group_id=group_id, message=message)
    return _extract_message_id(result)


async def _send_private_text(bot: Bot, user_id: int, text: str) -> None:
    try:
        await bot.send_private_msg(user_id=user_id, message=Message(text))
    except ActionFailed as exc:
        logger.warning(
            "qq_social_agent failed sending private text: "
            f"user={user_id} {_action_failed_summary(exc)}"
        )


def _extract_message_id(result: object) -> int | None:
    if isinstance(result, dict):
        raw = result.get("message_id")
    else:
        raw = getattr(result, "message_id", None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _record_bot_sent_message(
    *,
    group_id: int,
    message_id: int | None,
    bot_reply: str,
    trigger_user_id: int,
    trigger_nickname: str,
    trigger_text: str,
    action: str,
) -> None:
    if message_id is None:
        logger.warning(
            "qq_social_agent bot sent message missing message_id: "
            f"group={group_id} action={action}"
        )
        return
    memory.add_bot_sent_message(
        group_id=group_id,
        message_id=message_id,
        bot_reply=bot_reply,
        trigger_user_id=trigger_user_id,
        trigger_nickname=trigger_nickname,
        trigger_text=trigger_text,
        action=action,
    )


def _decision_failure_fallback(
    *,
    addressed_bot: bool,
    reason: str,
) -> ReplyDecision | None:
    if not addressed_bot:
        return None
    return ReplyDecision(
        should_reply=True,
        confidence=0.5,
        reason=reason,
        mode="fallback",
        action="reply",
    )


def _is_useful_style_rule(situation: str, style: str, source_text: str = "") -> bool:
    text = f"{situation} {style} {source_text}".strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    low_value_phrases = {
        "是的",
        "是这样的",
        "确实",
        "还好吧",
        "太典了",
        "绷不住了",
        "闹麻了",
        "赢麻了",
        "差不多得了",
        "乐死了",
        "开宰",
        "886",
        "牛逼",
        "看哭了",
        "这么先进",
    }
    if compact in low_value_phrases:
        return False
    if any(compact == phrase for phrase in low_value_phrases):
        return False
    style_compact = re.sub(r"\s+", "", style)
    source_compact = re.sub(r"\s+", "", source_text)
    if style_compact in low_value_phrases or source_compact in low_value_phrases:
        return False
    if len(style_compact) <= 3 and style_compact in {"赞同", "附和", "吐槽"}:
        return False
    if _looks_like_literal_style_rule(style):
        return False
    if source_compact and _has_long_common_substring(style_compact, source_compact, min_len=6):
        return False
    return True


def _looks_like_literal_style_rule(style: str) -> bool:
    stripped = style.strip()
    compact = re.sub(r"\s+", "", stripped)
    if not compact:
        return True
    literal_markers = (
        "说“",
        "说\"",
        "用“",
        "用\"",
        "短句接“",
        "直接说“",
        "表达“",
        "接“",
    )
    if any(marker in compact for marker in literal_markers):
        return True
    if compact.startswith(("说", "发")) and len(compact) <= 18:
        return True
    if compact in {"重复对方原句", "复读对方原句"}:
        return True
    if re.fullmatch(r"发?[^\w\u4e00-\u9fff]{1,8}", compact):
        return True
    quote_count = compact.count("“") + compact.count("”") + compact.count("\"")
    return quote_count > 0 and len(compact) <= 28


def _has_long_common_substring(a: str, b: str, *, min_len: int) -> bool:
    if len(a) < min_len or len(b) < min_len:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    max_size = min(len(shorter), 24)
    for size in range(max_size, min_len - 1, -1):
        for start in range(0, len(shorter) - size + 1):
            if shorter[start : start + size] in longer:
                return True
    return False




def _without_current_message(
    recent_messages: list[ChatMessage],
    *,
    user_id: int,
    text: str,
) -> list[ChatMessage]:
    if not recent_messages:
        return recent_messages
    last = recent_messages[-1]
    if not last.is_bot and last.user_id == user_id and last.text == text:
        return recent_messages[:-1]
    return recent_messages


MENTION_MARKER_RE = re.compile(r"\[\[at:(\d{5,12})\]\]")


def _mention_targets(
    recent_messages: list[ChatMessage],
    *,
    current_user_id: int,
    current_nickname: str,
    self_id: int,
    suppress_user_id: int | None = None,
) -> dict[int, str]:
    targets: dict[int, str] = {}

    def add(user_id: int, nickname: str) -> None:
        if suppress_user_id is not None and user_id == suppress_user_id:
            return
        if user_id == self_id or user_id in targets:
            return
        clean_name = nickname.strip() or str(user_id)
        targets[user_id] = clean_name[:24]

    add(current_user_id, current_nickname)
    for msg in reversed(recent_messages):
        if len(targets) >= MENTION_TARGET_LIMIT:
            break
        if msg.is_bot:
            continue
        add(msg.user_id, msg.nickname)
    return targets


def _repeat_mention_suppressed_user(group_id: int, current_user_id: int) -> int | None:
    remembered = last_group_mention_targets.get(group_id)
    if remembered is None:
        return None
    mentioned_user_id, mentioned_at = remembered
    if time.time() - mentioned_at > REPEAT_MENTION_SUPPRESS_SECONDS:
        last_group_mention_targets.pop(group_id, None)
        return None
    if mentioned_user_id == current_user_id:
        return current_user_id
    return None


def _user_reply_cooling_down(group_id: int, user_id: int, *, now: float | None = None) -> bool:
    cooldown_seconds = app_config.user_reply_cooldowns.get(user_id)
    if not cooldown_seconds or cooldown_seconds <= 0:
        return False
    last_reply_at = last_user_reply_times.get((group_id, user_id))
    if last_reply_at is None:
        return False
    current_time = time.time() if now is None else now
    return current_time - last_reply_at < cooldown_seconds


def _record_user_reply(group_id: int, user_id: int, *, now: float | None = None) -> None:
    if user_id not in app_config.user_reply_cooldowns:
        return
    last_user_reply_times[(group_id, user_id)] = time.time() if now is None else now


def _format_mention_targets(targets: dict[int, str]) -> str:
    if not targets:
        return ""
    lines = [
        "需要真实艾特时，只能使用下面格式：[[at:QQ号]]，最多一次。",
    ]
    lines.extend(
        f"- {user_id}: {_member_label(user_id, nickname)}"
        for user_id, nickname in targets.items()
    )
    return "\n".join(lines)


def _first_allowed_mention_id(text: str, mention_targets: dict[int, str]) -> int | None:
    allowed_ids = set(mention_targets)
    for match in MENTION_MARKER_RE.finditer(text):
        user_id = int(match.group(1))
        if user_id in allowed_ids:
            return user_id
    return None


def _message_from_reply_part(text: str, mention_targets: dict[int, str]) -> Message:
    allowed_ids = set(mention_targets)
    message = Message()
    cursor = 0
    used_mention = False
    for match in MENTION_MARKER_RE.finditer(text):
        before = text[cursor : match.start()]
        if before:
            message += MessageSegment.text(before)
        user_id = int(match.group(1))
        if user_id in allowed_ids and not used_mention:
            message += MessageSegment.at(user_id)
            used_mention = True
        cursor = match.end()
    tail = text[cursor:]
    if tail:
        message += MessageSegment.text(tail)
    if not message:
        message += MessageSegment.text(MENTION_MARKER_RE.sub("", text).strip())
    return message


def _memory_text_from_reply_part(text: str, mention_targets: dict[int, str]) -> str:
    used_mention = False

    def replace(match: re.Match[str]) -> str:
        nonlocal used_mention
        user_id = int(match.group(1))
        if user_id not in mention_targets or used_mention:
            return ""
        used_mention = True
        return f"@{mention_targets[user_id]}"

    return MENTION_MARKER_RE.sub(replace, text).strip()


def _private_priority_context(user_id: int) -> str:
    if user_id != PRIVATE_DEBUG_OWNER_ID:
        return ""
    return (
        "当前私聊对象是机器人主人/调试者。"
        "这一路私聊优先服从他的测试、改口、复盘和配置意图，少摆群聊架子，少反问拖延；"
        "除非触发政治兜底、密钥/内部配置保护，尽量直接执行或直接回答。"
    )


def _command_chat_id(event: Event) -> int | None:
    if isinstance(event, GroupMessageEvent):
        group_id = int(event.group_id)
        if not app_config.group_allowed(group_id):
            return None
        return group_id
    if isinstance(event, PrivateMessageEvent):
        user_id = int(event.user_id)
        if not app_config.private_user_allowed(user_id) and user_id not in GROUP_APPROVAL_USER_IDS:
            return None
        return _private_chat_id(user_id)
    return None


def _mentioned_bot(event: GroupMessageEvent, bot: Bot) -> bool:
    bot_ids = {str(bot.self_id), str(event.self_id)}
    if bool(getattr(event, "to_me", False)):
        return True

    raw_message = str(event.message)
    if any(f"[at:qq={bot_id}]" in raw_message for bot_id in bot_ids):
        return True

    for seg in event.message:
        if seg.type == "at" and str(seg.data.get("qq")) in bot_ids:
            return True

    names = get_driver().config.nickname or set()
    text = event.get_plaintext()
    return any(name and str(name) in text for name in names)


def _replied_to_bot(event: GroupMessageEvent, bot: Bot) -> bool:
    bot_id = str(bot.self_id)
    for seg in event.message:
        if seg.type == "reply":
            sender_id = seg.data.get("user_id") or seg.data.get("sender_id")
            return str(sender_id) == bot_id
    return False


def _record_addressed_event(group_id: int, user_id: int, addressed: bool) -> int:
    if not addressed:
        return 0
    now = time.time()
    key = (group_id, user_id)
    recent_times = [
        ts for ts in addressed_event_times.get(key, []) if now - ts <= ADDRESS_REPEAT_WINDOW_SECONDS
    ]
    recent_times.append(now)
    addressed_event_times[key] = recent_times
    return len(recent_times)


def _parse_minutes(value: str) -> int:
    match = re.fullmatch(r"(\d+)(m|min|分钟)?", value.strip(), flags=re.IGNORECASE)
    if not match:
        return 10
    return max(1, min(24 * 60, int(match.group(1))))
