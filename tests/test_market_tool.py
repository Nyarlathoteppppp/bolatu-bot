import pytest

from qq_social_agent.tools.market import (
    MarketSnapshot,
    MarketTool,
    _fetch_crypto_snapshot_from_gate,
    _fetch_stock_snapshot_from_sina,
    _fetch_stock_snapshot_from_yahoo_chart,
    _snapshot_chat_line,
)
from qq_social_agent.tools.market_intent import MarketIntent


@pytest.mark.anyio
async def test_market_tool_reports_rate_limit_failure() -> None:
    tool = MarketTool(max_external_queries_per_minute=0)
    context = await tool.context_for([MarketIntent("crypto", "bitcoin", "BTC")])
    assert "查询失败" in context
    assert "已达上限" in context
    assert "不要编造价格" in context


def test_stock_snapshot_uses_yahoo_chart_api(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "symbol": "NVDA",
                                "regularMarketPrice": 203.2,
                                "chartPreviousClose": 200.0,
                                "currency": "USD",
                                "regularMarketVolume": 123456,
                                "regularMarketTime": 1783634400,
                            }
                        }
                    ]
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str, params: dict[str, str], headers: dict[str, str]) -> FakeResponse:
            assert "NVDA" in url
            assert params == {"range": "1d", "interval": "1m"}
            assert "User-Agent" in headers
            return FakeResponse()

    monkeypatch.setattr("qq_social_agent.tools.market.httpx.Client", FakeClient)

    snapshot = _fetch_stock_snapshot_from_yahoo_chart(MarketIntent("stock", "NVDA", "NVDA"))

    assert snapshot is not None
    assert snapshot.price == 203.2
    assert snapshot.change_percent == pytest.approx(1.6)
    assert snapshot.source == "Yahoo Finance Chart"


def test_market_chat_line_includes_short_insight() -> None:
    line = _snapshot_chat_line(
        MarketSnapshot(
            kind="stock",
            symbol="NVDA",
            display_name="NVDA",
            price=203.2,
            currency="USD",
            change_percent=1.6,
            volume=123456,
            market_cap=None,
            source="Yahoo Finance Chart",
        )
    )

    assert "短评" in line
    assert "短线偏强" in line


def test_stock_snapshot_uses_sina_fallback(monkeypatch) -> None:
    class FakeResponse:
        text = 'var hq_str_gb_googl="谷歌A类股,356.1300,6.73,2026-08-03 19:08:22,0,0,0,0,0,0,1050000,0,2800000000000";'

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            assert "gb_googl" in url
            assert headers["Referer"] == "https://finance.sina.com.cn/"
            return FakeResponse()

    monkeypatch.setattr("qq_social_agent.tools.market.httpx.Client", FakeClient)

    snapshot = _fetch_stock_snapshot_from_sina(MarketIntent("stock", "GOOGL", "谷歌"))

    assert snapshot is not None
    assert snapshot.symbol == "GOOGL"
    assert snapshot.display_name == "谷歌"
    assert snapshot.price == 356.13
    assert snapshot.change_percent == pytest.approx(6.73)
    assert snapshot.source == "新浪财经"


@pytest.mark.anyio
async def test_crypto_snapshot_uses_gate_fallback(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, str]]:
            return [
                {
                    "last": "62595.7",
                    "change_percentage": "-0.93",
                    "quote_volume": "269857424.5",
                }
            ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str, params: dict[str, str], headers: dict[str, str]) -> FakeResponse:
            assert "gateio" in url
            assert params == {"currency_pair": "BTC_USDT"}
            assert "User-Agent" in headers
            return FakeResponse()

    monkeypatch.setattr("qq_social_agent.tools.market.httpx.AsyncClient", FakeAsyncClient)

    snapshot = await _fetch_crypto_snapshot_from_gate(MarketIntent("crypto", "bitcoin", "BTC"))

    assert snapshot is not None
    assert snapshot.symbol == "BTC"
    assert snapshot.price == 62595.7
    assert snapshot.change_percent == pytest.approx(-0.93)
    assert snapshot.source == "Gate.io"
