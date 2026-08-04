from qq_social_agent.tools.market_intent import detect_market_intents, is_market_topic


def test_detect_stock_alias_and_crypto_symbol() -> None:
    intents = detect_market_intents("英伟达今天咋样，BTC多少了")
    assert [(intent.kind, intent.symbol) for intent in intents] == [
        ("crypto", "bitcoin"),
        ("stock", "NVDA"),
    ]


def test_detect_known_stock_ticker_without_market_hint() -> None:
    intents = detect_market_intents("NVDA今天咋样")
    assert len(intents) == 1
    assert intents[0].kind == "stock"
    assert intents[0].symbol == "NVDA"


def test_limits_to_two_intents() -> None:
    intents = detect_market_intents("BTC ETH SOL NVDA TSLA")
    assert len(intents) == 2


def test_market_topic_without_specific_symbol() -> None:
    assert is_market_topic("那是不是美股也可以看盘")


def test_does_not_detect_unknown_ticker_from_member_name_with_amount_hint() -> None:
    text = (
        "土木-血火同源-偶像痴-NjTech本-HHU硕[#60236]说：我存在 "
        "邪恶代代[#56514]说：最后你考了多少分 "
        "邪恶代代[#56514]说：数学一"
    )

    assert detect_market_intents(text) == []


def test_unknown_ticker_requires_explicit_stock_code_signal() -> None:
    cashtag = detect_market_intents("$HHU 多少了")
    prefixed = detect_market_intents("查股票 HHU 现在多少")

    assert [(intent.kind, intent.symbol) for intent in cashtag] == [("stock", "HHU")]
    assert [(intent.kind, intent.symbol) for intent in prefixed] == [("stock", "HHU")]


def test_plain_english_name_with_market_hint_is_not_unknown_ticker() -> None:
    assert detect_market_intents("Jane Street 给多少") == []

