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
