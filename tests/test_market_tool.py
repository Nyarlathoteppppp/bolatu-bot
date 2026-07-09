import pytest

from qq_social_agent.tools.market import MarketTool
from qq_social_agent.tools.market_intent import MarketIntent


@pytest.mark.anyio
async def test_market_tool_reports_rate_limit_failure() -> None:
    tool = MarketTool(max_external_queries_per_minute=0)
    context = await tool.context_for([MarketIntent("crypto", "bitcoin", "BTC")])
    assert "查询失败" in context
    assert "已达上限" in context
    assert "不要编造价格" in context
