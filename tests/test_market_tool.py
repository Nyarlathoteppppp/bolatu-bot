import pytest

from qq_social_agent.tools.market import (
    MarketSnapshot,
    MarketTool,
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
