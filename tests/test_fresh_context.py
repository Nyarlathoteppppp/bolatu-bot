import pytest

from qq_social_agent.tools.fresh_context import (
    FreshLookup,
    FreshContextTool,
    fact_pack_from_lookup,
    _parse_google_news_rss,
    _parse_tavily_answer,
    _parse_tavily_results,
    _prompt_context_from_lookup,
)


def test_parse_google_news_rss_items() -> None:
    items = _parse_google_news_rss(
        """
        <rss>
          <channel>
            <item>
              <title>世界杯比赛结果表﹝网址：example.com﹞ - Results on X | Live Posts &amp; Updates</title>
              <source>x.com</source>
              <pubDate>Wed, 08 Jul 2026 12:05:00 GMT</pubDate>
              <description>spam</description>
            </item>
            <item>
              <title>美国和伊朗局势升温 - 示例媒体</title>
              <source>示例媒体</source>
              <pubDate>Wed, 08 Jul 2026 13:05:00 GMT</pubDate>
              <description>&lt;a href="https://example.com"&gt;相关报道&lt;/a&gt; 摘要内容</description>
            </item>
          </channel>
        </rss>
        """
    )

    assert len(items) == 1
    assert items[0].title == "美国和伊朗局势升温"
    assert items[0].source == "示例媒体"
    assert items[0].published_at == "2026-07-08 13:05"
    assert "摘要内容" in items[0].summary


def test_parse_tavily_results_items() -> None:
    items = _parse_tavily_results(
        {
            "results": [
                {
                    "title": "世界杯赛果预测和赔率",
                    "url": "https://spam.example.com/a",
                    "content": "bad",
                },
                {
                    "title": "美国和伊朗冲突最新进展",
                    "url": "https://news.example.com/world/iran",
                    "content": "双方局势仍在变化，多个消息源称外交斡旋继续。",
                    "published_date": "2026-07-09",
                    "score": 0.6,
                },
                {
                    "title": "美国和伊朗冲突最新进展",
                    "url": "https://news.example.com/world/iran?utm=1",
                    "content": "重复内容。",
                    "published_date": "2026-07-09",
                    "score": 0.9,
                },
            ]
        }
    )

    assert len(items) == 1
    assert items[0].title == "美国和伊朗冲突最新进展"
    assert items[0].source == "news.example.com"
    assert items[0].published_at == "2026-07-09"
    assert "外交斡旋" in items[0].summary
    assert items[0].score == 0.6


def test_parse_tavily_answer() -> None:
    answer = _parse_tavily_answer({"answer": "  美国和伊朗局势仍在变化。\n外交斡旋继续。  "})

    assert answer == "美国和伊朗局势仍在变化。 外交斡旋继续。"


def test_fresh_context_includes_quick_answer() -> None:
    context = _prompt_context_from_lookup(
        FreshLookup(
            query="美国 伊朗 冲突",
            kind="news",
            items=(),
            status="ok",
            provider="tavily",
            answer="局势仍在变化，外交斡旋继续。",
        )
    )

    assert "快速摘要：局势仍在变化" in context
    assert "多来源共同支持" in context


def test_fresh_fact_pack_structures_sources_and_uncertainty() -> None:
    lookup = FreshLookup(
        query="美国 伊朗 冲突",
        kind="news",
        items=_parse_tavily_results(
            {
                "results": [
                    {
                        "title": "美国和伊朗局势仍在变化",
                        "url": "https://news.example.com/world/iran",
                        "content": "外交斡旋继续，局势仍需观察。",
                        "published_date": "2026-07-09",
                    }
                ]
            }
        ),
        status="ok",
        provider="tavily",
        answer="双方局势仍在变化。",
    )

    pack = fact_pack_from_lookup(lookup)

    assert pack.topic == "美国 伊朗 冲突"
    assert pack.status == "ok"
    assert pack.sources == ("news.example.com",)
    assert any("快速摘要" in fact for fact in pack.facts)
    assert "来源较少" in pack.uncertain[0]


@pytest.mark.anyio
async def test_fresh_context_rate_limit_failure() -> None:
    tool = FreshContextTool(max_external_queries_per_minute=0)
    context = await tool.context_for("美国 伊朗 冲突 最新消息")

    assert "最新背景信息" in context
    assert "查询已达上限" in context
    assert "不要编造最新事实" in context
    assert "不要说“没联网”" in context
