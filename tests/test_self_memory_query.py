import nonebot

nonebot.init()

from qq_social_agent.memory import ChatMessage, MemberImpression, MemberProfile
from qq_social_agent.plugin import (
    _format_member_context,
    _is_self_memory_query,
    _member_memory_user_ids,
)


def test_self_memory_query_detects_common_asks() -> None:
    assert _is_self_memory_query("你记录了我什么")
    assert _is_self_memory_query("@张风雪（开启） 你记录了我什么")
    assert _is_self_memory_query("你记录了关于我的哪些信息")
    assert _is_self_memory_query("我是谁你知道吗")
    assert not _is_self_memory_query("我也水产研2了")
    assert not _is_self_memory_query("我儿子是谁")


def test_member_memory_user_ids_focus_on_speaker_for_self_query() -> None:
    recent = [
        ChatMessage(group_id=1, user_id=2362945204, nickname="沈清和", text="你记录了我什么", is_bot=False, created_at=1),
        ChatMessage(group_id=1, user_id=1347532835, nickname="安钰", text="你记录了我什么", is_bot=False, created_at=2),
    ]
    assert _member_memory_user_ids(recent, current_user_id=1347532835, current_text="你记录了我什么") == [1347532835]
    related = _member_memory_user_ids(recent, current_user_id=1347532835, current_text="水产学院")
    assert related[0] == 1347532835
    assert 2362945204 in related


def test_format_member_context_labels_speaker_versus_bystander() -> None:
    speaker = MemberImpression(
        group_id=1,
        user_id=1347532835,
        display_name="安钰与雨与余",
        aliases=("安钰与雨与余",),
        message_count=3,
        top_tags=(),
        top_keywords=(),
        recent_texts=(),
        ai_summary="活跃的AI技术讨论者",
        ai_interests=("中转站",),
        ai_speaking_style="反问",
        ai_representative_texts=(),
        ai_summary_at=1.0,
        last_seen_at=1.0,
        updated_at=1.0,
    )
    bystander = MemberProfile(
        group_id=1,
        user_id=2362945204,
        display_name="沈清和",
        aliases=("沈清和",),
        last_seen_at=1.0,
    )
    context = _format_member_context([speaker, bystander], current_user_id=1347532835)
    assert "当前触发人画像" in context
    assert "旁人画像" in context
    assert context.index("安钰") < context.index("沈清和")
    assert "禁止说成当前触发人的经历" in context
