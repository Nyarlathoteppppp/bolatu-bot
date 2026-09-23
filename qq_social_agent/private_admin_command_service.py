from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot

from .approval_rules import (
    APPROVAL_DETAIL_COMMANDS,
    APPROVAL_HELP_COMMANDS,
    APPROVAL_RULES_DETAIL_MESSAGE,
    APPROVAL_RULES_MESSAGE,
    APPROVAL_REVIEW_OFF_COMMANDS,
    APPROVAL_REVIEW_ON_COMMANDS,
    APPROVAL_REVIEW_STATUS_COMMANDS,
)


@dataclass(frozen=True)
class PrivateAdminCommandServices:
    send_private_text: Callable[[Bot, int, str], Awaitable[None]]
    basic_denied_message: str
    is_jargon_command_text: Callable[[str], bool]
    private_jargon_group_id: Callable[[], int]
    handle_jargon_command_text: Callable[..., str]
    bot_tool_message: Callable[[str], str | None]
    handle_approver_management_command: Callable[[Bot, int, str], Awaitable[bool]]
    is_private_tool_text: Callable[[str], bool]
    handle_private_whitelist_command: Callable[[Bot, int, str], Awaitable[bool]]
    handle_model_route_command: Callable[[Bot, int, str], Awaitable[bool]]
    parse_memory_report_limit: Callable[..., int | None]
    memory_report_command_re: Any
    format_recent_memory_report: Callable[[int, int], str]
    style_report_command_re: Any
    format_recent_style_report: Callable[[int, int], str]
    member_impression_report_command_re: Any
    format_member_impression_report: Callable[[int, int], str]
    handle_memory_atom_command_text: Callable[[int, int, str], str | None]
    memory_atom_report_command_re: Any
    format_memory_atom_report: Callable[[int, int], str]
    rag_admin: Any
    parse_metric_report_command: Callable[[str], Any | None]
    format_metric_report: Callable[..., str]
    parse_token_report_command: Callable[[str], Any | None]
    token_usage_report_for_window: Callable[[Any], str]
    parse_suppression_report_command: Callable[[str], int | None]
    format_suppression_report: Callable[[int], str]
    can_manage_approval_auto_send_percent: Callable[[int], bool]
    approval_auto_send_percent_re: Any
    format_approval_review_status: Callable[[], str]
    set_approval_auto_send_percent: Callable[[int], int]
    ai_work_intensity_percent_re: Any
    format_ai_work_intensity_status: Callable[[], str]
    set_ai_work_intensity_percent: Callable[[int], int]
    can_manage_approval_review: Callable[[int], bool]
    set_approval_review_enabled: Callable[[Bot, int, bool], Awaitable[None]]
    set_approval_group_decision_enabled: Callable[[Bot, int, bool], Awaitable[None]]
    is_approval_control_text: Callable[[str], bool]


async def handle_private_admin_command(
    bot: Bot,
    user_id: int,
    compact_text: str,
    *,
    is_admin: bool,
    pending_approval_control: bool,
    can_manage_auto_send_percent: bool,
    can_manage_review: bool,
    services: PrivateAdminCommandServices,
) -> bool:
    if services.is_jargon_command_text(compact_text):
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.send_private_text(
            bot,
            user_id,
            services.handle_jargon_command_text(
                user_id=user_id,
                group_id=services.private_jargon_group_id(),
                text=compact_text,
            ),
        )
        return True
    if not pending_approval_control and compact_text in APPROVAL_HELP_COMMANDS:
        await services.send_private_text(bot, user_id, APPROVAL_RULES_MESSAGE)
        return True
    if not pending_approval_control:
        bot_tool_message = services.bot_tool_message(compact_text)
        if bot_tool_message is not None or compact_text in APPROVAL_DETAIL_COMMANDS:
            await services.send_private_text(
                bot,
                user_id,
                bot_tool_message or APPROVAL_RULES_DETAIL_MESSAGE,
            )
            return True
    if await services.handle_approver_management_command(bot, user_id, compact_text):
        return True
    if (
        not pending_approval_control
        and services.is_private_tool_text(compact_text)
        and not is_admin
        and not can_manage_auto_send_percent
        and not can_manage_review
    ):
        await services.send_private_text(bot, user_id, services.basic_denied_message)
        return True
    if await services.handle_private_whitelist_command(bot, user_id, compact_text):
        return True
    if await services.handle_model_route_command(bot, user_id, compact_text):
        return True

    memory_report_limit = services.parse_memory_report_limit(compact_text, services.memory_report_command_re)
    if memory_report_limit is not None:
        group_id = services.private_jargon_group_id()
        await services.send_private_text(
            bot,
            user_id,
            services.format_recent_memory_report(group_id, memory_report_limit),
        )
        return True
    style_report_limit = services.parse_memory_report_limit(compact_text, services.style_report_command_re)
    if style_report_limit is not None:
        group_id = services.private_jargon_group_id()
        await services.send_private_text(
            bot,
            user_id,
            services.format_recent_style_report(group_id, style_report_limit),
        )
        return True
    member_report_limit = services.parse_memory_report_limit(
        compact_text,
        services.member_impression_report_command_re,
    )
    if member_report_limit is not None:
        group_id = services.private_jargon_group_id()
        await services.send_private_text(
            bot,
            user_id,
            services.format_member_impression_report(group_id, member_report_limit),
        )
        return True
    group_id = services.private_jargon_group_id()
    atom_command_response = services.handle_memory_atom_command_text(user_id, group_id, compact_text)
    if atom_command_response is not None:
        await services.send_private_text(bot, user_id, atom_command_response)
        return True
    atom_report_limit = services.parse_memory_report_limit(compact_text, services.memory_atom_report_command_re)
    if atom_report_limit is not None:
        await services.send_private_text(
            bot,
            user_id,
            services.format_memory_atom_report(group_id, atom_report_limit),
        )
        return True
    rag_admin_result = await services.rag_admin.handle(
        compact_text,
        group_id=group_id,
        operator_id=user_id,
    )
    if rag_admin_result.handled:
        await services.send_private_text(bot, user_id, rag_admin_result.text)
        return True

    metric_window = services.parse_metric_report_command(compact_text)
    if metric_window is not None:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.send_private_text(
            bot,
            user_id,
            services.format_metric_report(metric_window, group_id=group_id),
        )
        return True
    token_report_window = services.parse_token_report_command(compact_text)
    if token_report_window is not None:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.send_private_text(bot, user_id, services.token_usage_report_for_window(token_report_window))
        return True
    suppression_report_limit = services.parse_suppression_report_command(compact_text)
    if suppression_report_limit is not None:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.send_private_text(
            bot,
            user_id,
            services.format_suppression_report(suppression_report_limit),
        )
        return True

    auto_send_match = services.approval_auto_send_percent_re.match(compact_text)
    if auto_send_match is not None:
        if not services.can_manage_approval_auto_send_percent(user_id):
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        raw_percent = auto_send_match.group("percent")
        if raw_percent is None:
            await services.send_private_text(bot, user_id, services.format_approval_review_status())
            return True
        percent = services.set_approval_auto_send_percent(int(raw_percent))
        await services.send_private_text(
            bot,
            user_id,
            (
                f"已设置免审自动发送概率：{percent}%。\n"
                "审查开启时，命中概率的候选会直接发送第 1 条；未命中仍发审批单。\n"
                "设置为 100% 时改用单条直发 prompt，不再生成三候选。"
            ),
        )
        return True
    work_intensity_match = services.ai_work_intensity_percent_re.match(compact_text)
    if work_intensity_match is not None:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        raw_percent = work_intensity_match.group("percent")
        if raw_percent is None:
            await services.send_private_text(bot, user_id, services.format_ai_work_intensity_status())
            return True
        percent = services.set_ai_work_intensity_percent(int(raw_percent))
        await services.send_private_text(
            bot,
            user_id,
            (
                f"已设置 AI 工作强度：{percent}%。\n"
                "群消息仍会写入上下文和学习素材；只有命中的触发批次会进入硬筛选、decision、搜索/行情和生成。"
            ),
        )
        return True
    if compact_text in APPROVAL_REVIEW_STATUS_COMMANDS:
        if not services.can_manage_approval_review(user_id):
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.send_private_text(bot, user_id, services.format_approval_review_status())
        return True
    if compact_text in APPROVAL_REVIEW_ON_COMMANDS:
        if not services.can_manage_approval_review(user_id):
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.set_approval_review_enabled(bot, user_id, True)
        return True
    if compact_text in APPROVAL_REVIEW_OFF_COMMANDS:
        if not services.can_manage_approval_review(user_id):
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.set_approval_review_enabled(bot, user_id, False)
        return True
    if compact_text in {"开启", "打开", "恢复"}:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.set_approval_group_decision_enabled(bot, user_id, True)
        return True
    if compact_text in {"关闭", "关掉", "暂停"}:
        if not is_admin:
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        await services.set_approval_group_decision_enabled(bot, user_id, False)
        return True
    if not services.is_approval_control_text(compact_text):
        if (
            services.is_private_tool_text(compact_text)
            and not is_admin
            and not can_manage_auto_send_percent
            and not can_manage_review
        ):
            await services.send_private_text(bot, user_id, services.basic_denied_message)
            return True
        return False
    return False
