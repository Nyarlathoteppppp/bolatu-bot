from qq_social_agent.group_jargon import GroupJargonEntry, detect_group_jargon_terms, group_jargon_context


def test_detect_group_jargon_terms() -> None:
    terms = detect_group_jargon_terms(["柏拉图今天又在聊 zbzy 和剩余价值"])

    assert terms == ("plato", "capitalism", "marx_political_economy")


def test_detect_member_alias_jargon_terms() -> None:
    terms = detect_group_jargon_terms(["恩泽和乌木都在，科蛆代科无代两个号一起出现了"])

    assert terms == ("member_enze", "member_wumu", "member_kedai")


def test_kedai_same_person_jargon_covers_both_accounts() -> None:
    terms = detect_group_jargon_terms(["科有代和纯真代代晚上都在"])
    context = group_jargon_context(terms)

    assert terms == ("member_kedai",)
    assert "3066256514" in context
    assert "2947279300" in context


def test_detect_departed_member_jargon_terms() -> None:
    terms = detect_group_jargon_terms(["xhn和熊熊是离群老梗"])

    assert terms == ("departed_xhn_xiong",)


def test_departed_member_context_mentions_leiren() -> None:
    context = group_jargon_context(("熊熊",))

    assert "1660502091（雷人）" in context


def test_ximenqing_and_pan_jinlian_are_not_group_jargon() -> None:
    terms = detect_group_jargon_terms(["西门庆潘金莲"])

    assert terms == ()


def test_group_jargon_context_only_injects_selected_terms() -> None:
    context = group_jargon_context(("柏拉图", "zbzy"))

    assert "柏拉图" in context
    assert "zbzy" in context
    assert "王梓" not in context


def test_group_jargon_context_empty_when_no_selected_terms() -> None:
    assert group_jargon_context(()) == ""


def test_custom_group_jargon_entry_detected_and_injected() -> None:
    custom = (GroupJargonEntry("custom:das", ("达斯",), "指代：打死"),)

    terms = detect_group_jargon_terms(["这波达斯了"], extra_entries=custom)
    context = group_jargon_context(terms, extra_entries=custom)

    assert terms == ("custom:das",)
    assert "达斯：指代：打死" in context


def test_political_slang_jargon_terms() -> None:
    terms = detect_group_jargon_terms(["厚米和 homie 还有我儿子、腊爹"])
    assert "houmi" in terms
    assert "my_son" in terms
    assert "larou" in terms
    context = group_jargon_context(("homie", "我儿子"))
    assert "现任国家主席" in context
    assert "邓小平" in context


def test_xuehuo_is_not_qihen() -> None:
    from qq_social_agent.group_jargon import group_jargon_context, detect_group_jargon_terms
    terms = detect_group_jargon_terms(["奇恨博士不是血火"])
    context = group_jargon_context(terms)
    assert "member_xuehuo" in terms
    assert "3603460236" in context
    assert "不是奇恨博士" in context
    assert "血火通常指群友奇恨博士" not in context
