from qq_social_agent.tools.fresh_context import detect_fresh_intent, fresh_kind_from_text


def test_fresh_kind_from_news_text() -> None:
    assert fresh_kind_from_text("美国和伊朗现在怎么了") == "news"


def test_fresh_kind_from_sports_text() -> None:
    assert fresh_kind_from_text("世界杯今天比分多少") == "sports"


def test_fresh_kind_ignores_casual_text() -> None:
    assert fresh_kind_from_text("哈哈哈太典了") is None


def test_fresh_kind_ignores_plain_date_question() -> None:
    assert fresh_kind_from_text("今天周几") is None


def test_detect_fresh_intent_cleans_explicit_search_query() -> None:
    intent = detect_fresh_intent("搜一下世界杯今天比分多少")

    assert intent is not None
    assert intent.kind == "sports"
    assert intent.query == "世界杯今天比分多少"


def test_detect_fresh_intent_ignores_low_value_search() -> None:
    assert detect_fresh_intent("随便搜搜") is None
