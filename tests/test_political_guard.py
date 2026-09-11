from qq_social_agent.political_guard import has_political_redline, sanitize_political_output


def test_input_redline_never_blocks_chat() -> None:
    assert not has_political_redline("共产党就是独裁暴政")
    assert not has_political_redline("打倒CCP")
    assert not has_political_redline("你怎么看六四")
    assert not has_political_redline("聊聊64和学生运动")
    assert not has_political_redline("聊聊文革和毛泽东")
    assert not has_political_redline("习近平和共产党这个话题别直球")
    assert not has_political_redline("聊聊白纸运动")
    assert not has_political_redline("法轮功是什么")
    assert not has_political_redline("党员毕业去国企有优势吗")
    assert not has_political_redline("政治学专业就业怎么样")
    assert not has_political_redline("美国和伊朗冲突现在怎么样")


def test_does_not_treat_709_inside_message_identifiers_as_sensitive() -> None:
    assert not has_political_redline("张风雪原消息内容未知，消息ID：1392822709；可爱")
    reply, guarded = sanitize_political_output("消息ID：1392822709")
    assert not guarded
    assert reply == "消息ID：1392822709"


def test_sanitize_output_masks_redline_terms() -> None:
    reply, guarded = sanitize_political_output("中共暴政这个说法可以展开讲")

    assert guarded
    assert reply == "*zg**bz*这个说法可以展开讲"


def test_sanitize_output_masks_sensitive_keywords_but_keeps_reply() -> None:
    reply, guarded = sanitize_political_output("文革、64、学生运动、毛泽东、习近平、共产党这几个别直说")

    assert guarded
    assert reply == "*wg*、64、学生运动、*mzd*、*xjp*、*gcd*这几个别直说"


def test_sanitize_output_keeps_normal_reply() -> None:
    reply, guarded = sanitize_political_output("这专业就业要看城市和家庭试错空间。")

    assert not guarded
    assert reply == "这专业就业要看城市和家庭试错空间。"


def test_sanitize_skips_ambiguous_everyday_words() -> None:
    reply, guarded = sanitize_political_output("连任包子喝茶白纸新疆西藏64国家主席政治局")

    assert not guarded
    assert reply == "连任包子喝茶白纸新疆西藏64国家主席政治局"


def test_sanitize_output_masks_expanded_sensitive_keywords() -> None:
    reply, guarded = sanitize_political_output(
        "四人帮、江青、张春桥、姚文元、王洪文、反右、大跃进、红卫兵、批斗、上山下乡、林彪事件"
    )

    assert guarded
    assert reply == "*srb*、*jq*、*zcq*、*ywy*、*whw*、*fy*、*dyj*、*hwb*、*pd*、*ssxx*、*lbsj*"


def test_sanitize_output_masks_recent_region_and_slang_keywords() -> None:
    reply, guarded = sanitize_political_output(
        "六四、白纸运动、乌鲁木齐火灾、赵紫阳、薄熙来、周永康、大纪元、藏独、墙国、赵家人"
    )

    assert guarded
    assert reply == "*ls*、*bzyd*、*wlmqhz*、*zzy*、*bxl*、*zyk*、*djy*、*cd*、*qg*、*zjr*"


def test_sanitize_output_masks_pinyin_abbreviations() -> None:
    reply, guarded = sanitize_political_output("xjp、mzd、gcd、wenge")

    assert guarded
    assert reply == "*xjp*、*mzd*、*gcd*、*wenge*"


def test_short_ascii_terms_need_word_boundaries() -> None:
    reply, guarded = sanitize_political_output("accp toolkit 和 xjpg 不是敏感词")

    assert not guarded
    assert reply == "accp toolkit 和 xjpg 不是敏感词"
