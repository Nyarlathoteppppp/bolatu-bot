import pytest

import qq_social_agent.tools.fresh_context as fresh_context
from qq_social_agent.tools.fresh_context import (
    FreshItem,
    FreshLookup,
    FreshContextTool,
    SearchProviderError,
    _parse_bing_rss,
    fact_pack_from_lookup,
    detect_fresh_intent,
    _parse_google_news_rss,
    _parse_tavily_answer,
    _parse_tavily_results,
    _prompt_context_from_lookup,
    _safe_external_query,
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


@pytest.mark.anyio
async def test_tavily_answer_without_items_keeps_tavily_provider(monkeypatch) -> None:
    async def fake_tavily_lookup(query: str, *, kind: str, api_key: str):
        return "只有摘要，没有列表。", ()

    async def fail_google_news(query: str, *, kind: str):
        raise AssertionError("google news fallback should not run when tavily has answer")

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily_lookup)
    monkeypatch.setattr(fresh_context, "_fetch_google_news_items", fail_google_news)
    tool = FreshContextTool(provider="auto", tavily_api_key="test-key")

    lookup = await tool.lookup("美国 伊朗 冲突 最新消息")

    assert lookup.provider == "tavily"
    assert lookup.status == "ok"
    assert lookup.answer == "只有摘要，没有列表。"


def test_detect_explicit_web_news_and_sports_intents() -> None:
    web = detect_fresh_intent("搜一下 NoneBot 插件开发文档")
    news = detect_fresh_intent("联网查美国和伊朗最新消息")
    sports = detect_fresh_intent("网上找世界杯今天比分")

    assert web is not None and web.explicit and web.kind == "web"
    assert web.query == "NoneBot 插件开发文档"
    assert news is not None and news.explicit and news.kind == "news"
    assert sports is not None and sports.explicit and sports.kind == "sports"


@pytest.mark.parametrize(
    "text",
    [
        "美国挺抽象",
        "我今天吃什么",
        "你现在干嘛",
        "检查一下代码",
        "比赛真难看",
    ],
)
def test_detect_fresh_intent_avoids_casual_false_positives(text: str) -> None:
    assert detect_fresh_intent(text) is None


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("你们北大今年有两个菲奖得主了", "web"),
        ("今年的获奖名单已经公布了", "news"),
        ("本届世界杯冠军已经确定", "sports"),
    ],
)
def test_detect_fresh_intent_requires_verification_for_current_outcomes(
    text: str,
    kind: str,
) -> None:
    intent = detect_fresh_intent(text)

    assert intent is not None
    assert intent.kind == kind
    assert not intent.explicit
    assert intent.required


@pytest.mark.parametrize("text", ["今年好累", "今天吃鹅腿", "目前不想说话"])
def test_current_casual_chat_does_not_require_fresh_verification(text: str) -> None:
    assert detect_fresh_intent(text) is None


def test_current_fact_at_end_of_enriched_reply_wrapper_is_not_truncated() -> None:
    text = (
        "科有代（人类最终毁灭兵器）[#56514]回复张风雪-北本[#07496]消息【"
        "注：张风雪和风雪都是你自己；群友回复张风雪/风雪，就是在回复你之前说的话。"
        "张风雪-北本[#07496]说：代代咋突然发这个呀~；"
        "科有代（人类最终毁灭兵器）[#56514]回复张风雪-北本[#07496]："
        "你们北大今年有两个菲奖得主了】"
    )

    intent = detect_fresh_intent(text)

    assert intent is not None
    assert intent.required
    assert intent.kind == "web"
    assert intent.query == "你们北大今年有两个菲奖得主了"


def test_parse_bing_web_rss_preserves_traceable_metadata() -> None:
    items = _parse_bing_rss(
        """
        <rss><channel><item>
          <title>NoneBot 插件开发指南</title>
          <link>https://nonebot.dev/docs/tutorial/plugin/create-plugin</link>
          <pubDate>Sun, 12 Jul 2026 08:00:00 GMT</pubDate>
          <description>介绍如何创建和加载插件。</description>
        </item></channel></rss>
        """
    )

    assert len(items) == 1
    assert items[0].source == "nonebot.dev"
    assert items[0].url.startswith("https://nonebot.dev/")
    assert items[0].published_at == "2026-07-12 08:00"


def test_prompt_context_numbers_sources_and_marks_untrusted_data() -> None:
    context = _prompt_context_from_lookup(
        FreshLookup(
            query="NoneBot 文档",
            kind="web",
            items=(
                FreshItem(
                    title="插件开发",
                    source="nonebot.dev",
                    published_at="2026-07-12",
                    summary="创建插件的方法。",
                    url="https://nonebot.dev/docs/plugin",
                ),
            ),
            status="ok",
            provider="bing_web",
        )
    )

    assert "[S1]" in context
    assert "https://nonebot.dev/docs/plugin" in context
    assert "不可信外部数据" in context
    assert "不要编造来源" in context


@pytest.mark.anyio
async def test_web_search_falls_back_to_bing_not_google_news(monkeypatch) -> None:
    async def empty_tavily(query: str, *, kind: str, api_key: str):
        return "", ()

    async def fail_google(query: str, *, kind: str):
        raise AssertionError("web search must not use Google News")

    async def fake_bing(query: str):
        return (
            FreshItem("NoneBot 文档", "nonebot.dev", "", url="https://nonebot.dev/docs"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", empty_tavily)
    monkeypatch.setattr(fresh_context, "_fetch_google_news_items", fail_google)
    monkeypatch.setattr(fresh_context, "_fetch_bing_web_items", fake_bing)
    tool = FreshContextTool(provider="auto", tavily_api_key="test-key")

    lookup = await tool.lookup("NoneBot 插件文档", kind="web")

    assert lookup.status == "ok"
    assert lookup.provider == "bing_web"
    assert lookup.attempted_providers == ("tavily", "bing_web")


@pytest.mark.anyio
async def test_tavily_answer_only_uses_success_cache_ttl(monkeypatch) -> None:
    calls = 0

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        nonlocal calls
        calls += 1
        return "可用摘要。", ()

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        cache_ttl_seconds=60,
        failure_ttl_seconds=0,
    )

    first = await tool.lookup("测试主题", kind="news")
    second = await tool.lookup("测试主题", kind="news")

    assert first.status == "ok"
    assert second.cached
    assert calls == 1


@pytest.mark.anyio
async def test_lookup_observes_provider_errors_and_status_without_secrets(monkeypatch) -> None:
    captured_queries: list[str] = []

    async def fail_tavily(query: str, *, kind: str, api_key: str):
        captured_queries.append(query)
        raise SearchProviderError("timeout")

    async def empty_google(query: str, *, kind: str):
        captured_queries.append(query)
        return ()

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fail_tavily)
    monkeypatch.setattr(fresh_context, "_fetch_google_news_items", empty_google)
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    tool = FreshContextTool(provider="auto", tavily_api_key="configured-secret", query_max_chars=72)

    lookup = await tool.lookup(f"美国最新消息 api_key={secret} QQ 123456789", kind="news")
    status = tool.status_snapshot()

    assert lookup.status == "no_result"
    assert lookup.attempted_providers == ("tavily", "google_news")
    assert "tavily:timeout" in lookup.error
    assert all(secret not in query for query in captured_queries)
    assert "123456789" not in lookup.query
    assert "configured-secret" not in str(status)
    assert secret not in str(status)
    assert status["counters"]["external_requests"] == 1
    assert status["last_request"]["latency_ms"] >= 0


def test_safe_external_query_redacts_tokens_and_long_qq_numbers() -> None:
    safe = _safe_external_query(
        "查资料 token=abcdefghijklmno sk-abcdefghijklmnop QQ 123456789",
        max_chars=120,
    )

    assert "abcdefghijklmno" not in safe
    assert "sk-abcdefghijklmnop" not in safe
    assert "123456789" not in safe
    assert "[已隐藏" in safe
    assert "[QQ号]" in safe


def test_from_config_applies_runtime_limits(monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_TAVILY_KEY", "configured")
    tool = FreshContextTool.from_config(
        {
            "provider": "auto",
            "max_queries_per_minute": 7,
            "timeout_seconds": 4.5,
            "max_results": 3,
            "cache_max_entries": 12,
            "query_max_chars": 80,
            "news_cache_ttl_seconds": 300,
            "sports_cache_ttl_seconds": 45,
            "web_cache_ttl_seconds": 1800,
            "tavily": {"api_key_env": "CUSTOM_TAVILY_KEY"},
        }
    )

    assert tool.max_external_queries_per_minute == 7
    assert tool.timeout_seconds == 4.5
    assert tool.max_results == 3
    assert tool.cache_max_entries == 12
    assert tool.query_max_chars == 80
    assert tool.cache_ttl_by_kind == {"news": 300, "sports": 45, "web": 1800}
    assert tool.tavily_api_key == "configured"
