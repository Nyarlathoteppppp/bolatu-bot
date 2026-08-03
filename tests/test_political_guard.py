from qq_social_agent.political_guard import has_political_redline, sanitize_political_output


def test_detects_direct_party_attack() -> None:
    assert has_political_redline("共产党就是独裁暴政")
    assert has_political_redline("打倒CCP")


def test_detects_sensitive_domestic_events() -> None:
    assert has_political_redline("你怎么看六四")
    assert has_political_redline("聊聊64和学生运动")
    assert has_political_redline("聊聊文革和毛泽东")
    assert has_political_redline("习近平和共产党这个话题别直球")
    assert has_political_redline("聊聊白纸运动")
    assert has_political_redline("法轮功是什么")


def test_does_not_block_normal_topics() -> None:
    assert not has_political_redline("党员毕业去国企有优势吗")
    assert not has_political_redline("政治学专业就业怎么样")
    assert not has_political_redline("美国和伊朗冲突现在怎么样")


def test_sanitize_output_masks_redline_terms() -> None:
    reply, guarded = sanitize_political_output("中共暴政这个说法可以展开讲")

    assert guarded
    assert reply == "*g*z这个说法可以展开讲"


def test_sanitize_output_masks_sensitive_keywords_but_keeps_reply() -> None:
    reply, guarded = sanitize_political_output("文革、64、学生运动、毛泽东、习近平、共产党这几个别直说")

    assert guarded
    assert reply == "*g、**、*s*d、*z*、*j*、*c*这几个别直说"


def test_sanitize_output_keeps_normal_reply() -> None:
    reply, guarded = sanitize_political_output("这专业就业要看城市和家庭试错空间。")

    assert not guarded
    assert reply == "这专业就业要看城市和家庭试错空间。"


def test_sanitize_output_masks_expanded_sensitive_keywords() -> None:
    reply, guarded = sanitize_political_output(
        "国家主席、四人帮、江青、张春桥、姚文元、王洪文、反右、大跃进、红卫兵、批斗、上山下乡、林彪事件"
    )

    assert guarded
    assert reply == "*j*x、*r*、*q、*c*、*w*、*h*、*y、*y*、*w*、*d、*s*x、*b*j"


def test_sanitize_output_masks_recent_region_and_slang_keywords() -> None:
    reply, guarded = sanitize_political_output(
        "八九、学潮、天安门、白纸、乌鲁木齐火灾、佳士、709、乌坎、赵紫阳、薄熙来、周永康、政治局、中南海、大纪元、新疆、维吾尔、西藏、修宪、连任、登基、墙国、赵家人"
    )

    assert guarded
    assert reply == "*j、*c、*a*、*z、*l*q*z、*s、***、*k、*z*、*x*、*y*、*z*、*n*、*j*、*j、*w*、*c、*x、*r、*j、*g、*j*"


def test_sanitize_output_masks_pinyin_abbreviations() -> None:
    reply, guarded = sanitize_political_output("xjp、mzd、gcd、wenge")

    assert guarded
    assert reply == "*j*、*z*、*c*、*e*g*"
