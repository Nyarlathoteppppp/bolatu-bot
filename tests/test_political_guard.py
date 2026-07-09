from qq_social_agent.political_guard import (
    has_political_redline,
    political_safe_reply,
    sanitize_political_output,
)


def test_detects_direct_party_attack() -> None:
    assert has_political_redline("共产党就是独裁暴政")
    assert has_political_redline("打倒CCP")


def test_detects_sensitive_domestic_events() -> None:
    assert has_political_redline("你怎么看六四")
    assert has_political_redline("聊聊白纸运动")
    assert has_political_redline("法轮功是什么")


def test_does_not_block_normal_topics() -> None:
    assert not has_political_redline("党员毕业去国企有优势吗")
    assert not has_political_redline("政治学专业就业怎么样")
    assert not has_political_redline("美国和伊朗冲突现在怎么样")


def test_sanitize_output_replaces_redline_reply() -> None:
    reply, guarded = sanitize_political_output("中共暴政这个说法可以展开讲")

    assert guarded
    assert reply == political_safe_reply()


def test_sanitize_output_keeps_normal_reply() -> None:
    reply, guarded = sanitize_political_output("这专业就业要看城市和家庭试错空间。")

    assert not guarded
    assert reply == "这专业就业要看城市和家庭试错空间。"
