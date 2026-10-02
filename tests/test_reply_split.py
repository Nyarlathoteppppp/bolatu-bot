from qq_social_agent.reply_splitter import split_reply_messages


def test_split_two_meaningful_sentences() -> None:
    text = "这个专业别光看名字，核心还是就业出口。家里试错空间不大，就别拿四年去赌一个听起来高级的方向。"

    assert split_reply_messages(text, max_messages=2) == [
        "这个专业别光看名字，核心还是就业出口。",
        "家里试错空间不大，就别拿四年去赌一个听起来高级的方向。",
    ]


def test_does_not_split_short_fragments() -> None:
    assert split_reply_messages("可以。别急。", max_messages=2) == ["可以。别急。"]


def test_group_reply_caps_three_sentences_to_two_messages() -> None:
    text = "先别急着下结论。这个波动更像情绪盘。真要看还得等它站回关键位。"

    assert split_reply_messages(text, max_messages=2) == [
        "先别急着下结论。",
        "这个波动更像情绪盘。真要看还得等它站回关键位。",
    ]


def test_directed_reply_can_split_three_messages() -> None:
    text = "先别急着下结论。这个波动更像情绪盘。真要看还得等它站回关键位。"

    assert split_reply_messages(text, max_messages=3) == [
        "先别急着下结论。",
        "这个波动更像情绪盘。",
        "真要看还得等它站回关键位。",
    ]


def test_split_real_long_reply_at_chinese_ellipsis() -> None:
    text = (
        "年包80万的大厂算法岗，你得先补完体系结构、数据结构、算法那堆东西再说啊……"
        "而且现在算法岗卷成啥样了，双非本科基本简历都过不去，"
        "建议先刷LeetCode+考研冲个985硕，或者看看AI infra运维岗能不能曲线救国"
    )

    assert split_reply_messages(text, max_messages=3) == [
        "年包80万的大厂算法岗，你得先补完体系结构、数据结构、算法那堆东西再说啊……",
        (
            "而且现在算法岗卷成啥样了，双非本科基本简历都过不去，"
            "建议先刷LeetCode+考研冲个985硕，或者看看AI infra运维岗能不能曲线救国"
        ),
    ]


def test_split_comma_only_reply_below_model_character_limit() -> None:
    text = "第一段信息很长，" + "需要继续解释清楚，" * 7 + "最后给出一个明确建议"

    parts = split_reply_messages(text, max_messages=3)

    assert len(text) < 120
    assert len(parts) >= 2
    # Split points drop the dangling comma but lose no content.
    assert "，".join(parts) == text


def test_clause_split_does_not_leave_a_dangling_comma() -> None:
    from qq_social_agent.reply_splitter import split_reply_messages

    text = (
        "82% 服从率就是测它听指令办事的通过率，日常玩或者单次任务完全够用；但剩下那 18% 是随机翻车，"
        "真拿去跑批量自动化就不稳了，最好拿你自己的场景实测一下，别只看平均数"
    )
    parts = split_reply_messages(text, max_messages=2)
    assert len(parts) == 2
    assert not parts[0].endswith(("，", "；", ","))
