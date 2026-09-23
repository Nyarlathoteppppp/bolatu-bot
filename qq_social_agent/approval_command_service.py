from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Awaitable, Callable

from nonebot.adapters.onebot.v11 import Bot

from .approval_rules import APPROVAL_CANCEL_COMMANDS, APPROVAL_CHOICE_RE, APPROVAL_REJECT_REASON_RE
from .approval_state_service import ApprovalControl, ApprovalStateService, ApprovalStateServices


def is_approval_control_text(text: str) -> bool:
    return (
        APPROVAL_CHOICE_RE.match(text) is not None
        or text == "准奏"
        or text in APPROVAL_CANCEL_COMMANDS
        or APPROVAL_REJECT_REASON_RE.match(text) is not None
    )


def is_basic_approval_control_text(text: str) -> bool:
    choice_match = APPROVAL_CHOICE_RE.match(text)
    return (choice_match is not None and not choice_match.group(2)) or text in APPROVAL_CANCEL_COMMANDS


def approval_choice_index(raw: str | None, *, default: int = 1) -> int:
    if raw is None:
        return default
    return {"1": 1, "a": 1, "2": 2, "b": 2, "3": 3, "c": 3}.get(raw.strip().casefold(), default)


def parse_approval_control(text: str) -> ApprovalControl:
    choice_match = APPROVAL_CHOICE_RE.match(text)
    if choice_match is not None:
        return ApprovalControl(
            kind="select",
            candidate_index=approval_choice_index(choice_match.group(1)),
            high_quality=bool(choice_match.group(2)),
            raw_text=text,
        )
    if text == "准奏":
        return ApprovalControl(kind="approve_first", raw_text=text)
    reason_match = APPROVAL_REJECT_REASON_RE.match(text)
    if reason_match is not None:
        return ApprovalControl(
            kind="cancel",
            candidate_index=approval_choice_index(reason_match.group("index")),
            rejection_requested=True,
            rejection_reason=reason_match.group("reason").strip(),
            raw_text=text,
        )
    return ApprovalControl(kind="cancel", raw_text=text)


@dataclass(frozen=True)
class PrivateApprovalCommandServices:
    state: ApprovalStateService
    state_services: ApprovalStateServices
    is_approval_user: Callable[[int], bool]
    is_tool_admin_user: Callable[[int], bool]
    is_owner_user: Callable[[int], bool]
    is_basic_approval_user: Callable[[int], bool]
    latest_approval: Callable[[], Any]
    bot_tool_shortcut_command: Callable[[str], str | None]
    can_manage_auto_send_percent: Callable[[int], bool]
    auto_send_percent_re: Any
    can_manage_approval_review: Callable[[int], bool]
    review_command_texts: frozenset[str]
    handle_admin_command: Callable[..., Awaitable[bool]]
    send_private_text: Callable[[Bot, int, str], Awaitable[None]]
    basic_denied_message: str
    cooldown_seconds: float
    now: Callable[[], float] = time.time


async def handle_private_approval_command(
    bot: Bot,
    user_id: int,
    text: str,
    *,
    services: PrivateApprovalCommandServices,
) -> bool:
    if not services.is_approval_user(user_id) and not services.is_tool_admin_user(user_id):
        return False
    compact_text = text.strip()
    is_admin = services.is_tool_admin_user(user_id) or services.is_owner_user(user_id)
    auto_send_match = services.auto_send_percent_re.match(compact_text)
    can_manage_auto_send_percent = auto_send_match is not None and services.can_manage_auto_send_percent(user_id)
    can_manage_review = (
        services.can_manage_approval_review(user_id)
        and compact_text in services.review_command_texts
    )
    pending_approval_control = (
        services.latest_approval() is not None and is_approval_control_text(compact_text)
    )
    if not pending_approval_control:
        shortcut_command = services.bot_tool_shortcut_command(compact_text)
        if shortcut_command is not None:
            compact_text = shortcut_command
            auto_send_match = services.auto_send_percent_re.match(compact_text)
            can_manage_auto_send_percent = auto_send_match is not None and services.can_manage_auto_send_percent(user_id)
            can_manage_review = (
                services.can_manage_approval_review(user_id)
                and compact_text in services.review_command_texts
            )
    if await services.handle_admin_command(
        bot,
        user_id,
        compact_text,
        is_admin=is_admin,
        pending_approval_control=pending_approval_control,
        can_manage_auto_send_percent=can_manage_auto_send_percent,
        can_manage_review=can_manage_review,
    ):
        return True
    if not is_approval_control_text(compact_text):
        return False
    if services.is_basic_approval_user(user_id) and not is_basic_approval_control_text(compact_text):
        await services.send_private_text(bot, user_id, services.basic_denied_message)
        return True
    if services.now() < services.state.choice_cooldowns.get(user_id, 0.0):
        await services.send_private_text(
            bot,
            user_id,
            "上一条审批刚被处理，这次审批指令已忽略，避免串到下一条。",
        )
        return True
    await services.state.handle_control(
        bot,
        user_id,
        parse_approval_control(compact_text),
        is_admin=is_admin,
        services=services.state_services,
    )
    return True
