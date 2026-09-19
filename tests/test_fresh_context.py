import asyncio
import time

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
    _compact_search_query,
    _followup_skip_url,
    _httpx_timeout,
    _looks_like_low_quality_result,
    _rank_followup_items,
    _read_wikipedia_extract,
    _second_hop_query,
    _wikipedia_title_from_url,
    _parse_google_news_rss,
    _parse_searxng_results,
    _parse_tavily_answer,
    _parse_tavily_results,
    _prompt_context_from_lookup,
    _related_cache_scope_mismatch,
    _safe_external_query,
)
from qq_social_agent.tools.safe_url_reader import UrlReadResult


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


def test_parse_searxng_results_items() -> None:
    items = _parse_searxng_results(
        {
            "results": [
                {
                    "title": " 美国和伊朗局势最新进展 ",
                    "url": "https://news.example.com/iran",
                    "content": "外交斡旋仍在继续。",
                    "engine": "bing news",
                    "publishedDate": "2026-08-17",
                },
                {
                    "title": "美国和伊朗局势最新进展",
                    "url": "https://news.example.com/iran?ref=duplicate",
                    "content": "重复。",
                },
            ]
        }
    )

    assert len(items) == 1
    assert items[0].source == "bing news"
    assert items[0].published_at == "2026-08-17"
    assert "外交斡旋" in items[0].summary


@pytest.mark.anyio
async def test_searxng_provider_uses_local_json_endpoint(monkeypatch) -> None:
    async def fake_searxng(query: str, *, kind: str, base_url: str):
        assert base_url == "http://searxng:8080"
        return (FreshItem("最新进展", "示例媒体", "2026-08-17", "摘要", "https://example.com"),)

    monkeypatch.setattr(fresh_context, "_fetch_searxng_items", fake_searxng)
    tool = FreshContextTool(provider="searxng", searxng_base_url="http://searxng:8080")

    lookup = await tool.lookup("美国 伊朗 最新消息")

    assert lookup.status == "ok"
    assert lookup.provider == "searxng"
    assert lookup.attempted_providers[0] == "searxng"
    assert all(item.startswith("searxng") for item in lookup.attempted_providers)


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

    assert "<synthesized_answer>" in context
    assert "局势仍在变化" in context
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


@pytest.mark.anyio
async def test_fresh_context_enforces_total_timeout(monkeypatch) -> None:
    async def slow_tavily_lookup(query: str, *, kind: str, api_key: str):
        await asyncio.sleep(2)
        return "不该等到这里", ()

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", slow_tavily_lookup)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        timeout_seconds=1,
    )
    started = asyncio.get_running_loop().time()

    lookup = await tool.lookup("严格超时测试")

    assert asyncio.get_running_loop().time() - started < 1.3
    assert lookup.status == "failed"
    assert "total_timeout" in lookup.error


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
    first_calls = calls
    second = await tool.lookup("测试主题", kind="news")

    assert first.status == "ok"
    assert second.cached
    assert first_calls >= 1
    assert calls == first_calls


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
            "searxng": {"base_url": "http://searxng:8080"},
        }
    )

    assert tool.max_external_queries_per_minute == 7
    assert tool.timeout_seconds == 4.5
    assert tool.max_results == 3
    assert tool.cache_max_entries == 12
    assert tool.query_max_chars == 80
    assert tool.cache_ttl_by_kind == {"news": 300, "sports": 45, "web": 1800}
    assert tool.tavily_api_key == "configured"
    assert tool.searxng_base_url == "http://searxng:8080"

def test_safe_external_query_drops_abstract_group_banter() -> None:
    assert _safe_external_query("你被平行宇宙的美少女权踢了") == ""
    assert _safe_external_query("他被踢了") == ""
    assert _safe_external_query("美国伊朗冲突 最新") == "美国伊朗冲突 最新"
    assert _safe_external_query("比特币现在价格") == "比特币现在价格"


@pytest.mark.anyio
async def test_lookup_reuses_related_successful_cache_without_external_call() -> None:
    tool = FreshContextTool(max_external_queries_per_minute=0, cache_ttl_seconds=60)
    lookup = FreshLookup("美国伊朗冲突 最新", "news", (FreshItem("美国伊朗冲突最新进展", "示例媒体", "2026-08-03"),), "ok", provider="test", answer="双方局势仍在变化。")
    tool._cache[("news", "美国伊朗冲突 最新")] = (time.monotonic(), lookup)
    reused = await tool.lookup("美国伊朗现在冲突", kind="news")
    assert reused.status == "ok"
    assert reused.cached
    assert reused.provider == "test:related_cache"
    assert reused.answer == "双方局势仍在变化。"
    assert tool.status_snapshot()["counters"]["external_requests"] == 0


def test_httpx_timeout_caps_connect_so_dead_providers_leave_fallback_budget() -> None:
    timeout = _httpx_timeout(5.0)
    assert timeout.connect <= 1.5
    assert timeout.read == 5.0
    assert timeout.connect < timeout.read
    reserved = fresh_context._provider_timeout_seconds(5.0, has_later_provider=True)
    assert reserved < 5.0
    assert reserved >= 0.5


class _FakeUrlReader:
    def __init__(self, results: list[UrlReadResult]) -> None:
        self._results = list(results)
        self.urls: list[str] = []

    async def read(self, url: str) -> UrlReadResult:
        self.urls.append(url)
        if not self._results:
            return UrlReadResult("fetch_error", url, error="exhausted")
        return self._results.pop(0)


@pytest.mark.anyio
async def test_followup_reads_one_page_from_this_round_result_urls(monkeypatch) -> None:
    reader = _FakeUrlReader(
        [
            UrlReadResult(
                "ok",
                "https://example.com/nvidia",
                final_url="https://example.com/nvidia",
                title="NVIDIA 10-K",
                text="Fiscal year 2025 total revenue was $130.5 billion.",
            )
        ]
    )

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "data center 752", (
            FreshItem(
                "NVIDIA earnings",
                "example.com",
                "2026-02-01",
                summary="data center 752 billion",
                url="https://example.com/nvidia",
            ),
            FreshItem(
                "Other",
                "example.com",
                "",
                summary="ignore",
                url="https://example.com/other",
            ),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=reader,
        followup_page_max_successes=1,
        followup_search_hops=1,
    )

    lookup = await tool.lookup("NVIDIA 最新财报", kind="news")
    context = _prompt_context_from_lookup(lookup)

    assert lookup.status == "ok"
    assert reader.urls == ["https://example.com/nvidia"]
    assert "130.5 billion" in lookup.page_text
    assert "网页正文" in context
    assert "130.5 billion" in context
    assert context.index("网页正文") < context.index("<facts>")
    assert tool.status_snapshot()["last_request"]["page_status"] == "ok"


@pytest.mark.anyio
async def test_followup_retries_next_url_at_most_twice(monkeypatch) -> None:
    reader = _FakeUrlReader(
        [
            UrlReadResult("fetch_error", "https://example.com/a", error="timeout"),
            UrlReadResult(
                "ok",
                "https://example.com/b",
                final_url="https://example.com/b",
                title="B",
                text="正文第二页有 816 亿美元。",
            ),
            UrlReadResult(
                "ok",
                "https://example.com/c",
                final_url="https://example.com/c",
                title="C",
                text="不该读到第三页",
            ),
        ]
    )

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "", (
            FreshItem("A", "example.com", "", url="https://www.zhihu.com/topics"),
            FreshItem("B", "example.com", "", url="https://example.com/a"),
            FreshItem("C", "example.com", "", url="https://example.com/b"),
            FreshItem("D", "example.com", "", url="https://example.com/c"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=reader,
        followup_page_max_tries=2,
        followup_page_max_successes=1,
        followup_search_hops=1,
    )

    lookup = await tool.lookup("NVIDIA", kind="web")

    assert reader.urls == ["https://example.com/a", "https://example.com/b"]
    assert "816 亿美元" in lookup.page_text
    assert lookup.page_url == "https://example.com/b"


@pytest.mark.anyio
async def test_followup_page_timeout_does_not_eat_full_reader_budget(monkeypatch) -> None:
    class _SlowReader:
        urls: list[str] = []

        async def read(self, url: str) -> UrlReadResult:
            self.urls.append(url)
            await asyncio.sleep(8)
            return UrlReadResult("ok", url, final_url=url, title="late", text="不该等到这里")

    reader = _SlowReader()

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "摘要", (
            FreshItem("A", "example.com", "", url="https://example.com/slow"),
            FreshItem("B", "example.com", "", url="https://example.com/also-slow"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=reader,
        followup_page_timeout_seconds=1.0,
        followup_search_hops=1,
        timeout_seconds=5,
    )
    started = asyncio.get_running_loop().time()
    lookup = await tool.lookup("NVIDIA", kind="web")
    elapsed = asyncio.get_running_loop().time() - started

    assert lookup.status == "ok"
    assert lookup.page_text == ""
    assert lookup.page_status == "timeout"
    assert reader.urls == ["https://example.com/slow", "https://example.com/also-slow"]
    assert elapsed < 3.0


@pytest.mark.anyio
async def test_news_dead_first_provider_still_reaches_fallback(monkeypatch) -> None:
    calls: list[str] = []

    async def hang_tavily(query: str, *, kind: str, api_key: str, timeout_seconds: float = 10.0):
        calls.append(f"tavily:{timeout_seconds:.2f}")
        await asyncio.sleep(10)
        return "不该等到这里", ()

    async def fast_google(query: str, *, kind: str, timeout_seconds: float = 10.0):
        calls.append(f"google:{timeout_seconds:.2f}")
        return (FreshItem("局势", "示例媒体", "2026-09-10", url="https://news.example.com/a"),)

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", hang_tavily)
    monkeypatch.setattr(fresh_context, "_fetch_google_news_items", fast_google)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        timeout_seconds=5,
    )
    started = asyncio.get_running_loop().time()

    lookup = await tool.lookup("美国 伊朗 最新消息", kind="news")
    elapsed = asyncio.get_running_loop().time() - started

    assert lookup.status == "ok"
    assert lookup.provider == "google_news"
    assert elapsed < 5.0
    assert any(item.startswith("google:") for item in calls)


@pytest.mark.anyio
async def test_searxng_empty_or_slow_first_hop_falls_back_to_tavily(monkeypatch) -> None:
    calls: list[str] = []

    async def slow_empty_searxng(
        query: str,
        *,
        kind: str,
        base_url: str,
        timeout_seconds: float = 10.0,
        max_results: int = 5,
    ):
        calls.append(f"searxng:{timeout_seconds:.2f}")
        await asyncio.sleep(4)
        return ()

    async def fast_tavily(
        query: str,
        *,
        kind: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        max_results: int = 4,
    ):
        calls.append(f"tavily:{timeout_seconds:.2f}")
        return "双方局势仍在变化。", (
            FreshItem(
                "局势",
                "示例媒体",
                "2026-09-10",
                summary="外交斡旋继续",
                url="https://news.example.com/a",
            ),
        )

    monkeypatch.setattr(fresh_context, "_fetch_searxng_items", slow_empty_searxng)
    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fast_tavily)
    tool = FreshContextTool(
        provider="searxng",
        searxng_base_url="http://searxng:8080",
        tavily_api_key="test-key",
        timeout_seconds=5,
    )
    started = asyncio.get_running_loop().time()

    lookup = await tool.lookup("美国 伊朗 最新消息", kind="news")
    elapsed = asyncio.get_running_loop().time() - started

    assert lookup.status == "ok"
    assert lookup.provider == "tavily"
    assert elapsed < 4.0
    assert any(item.startswith("searxng:") for item in calls)
    assert any(item.startswith("tavily:") for item in calls)
    searxng_budget = float(next(item.split(":", 1)[1] for item in calls if item.startswith("searxng:")))
    assert searxng_budget <= 1.5


def test_followup_reads_zhihu_articles_but_skips_hubs_and_scrapers() -> None:
    assert _followup_skip_url("https://www.zhihu.com/question/2070082903229347029") is False
    assert _followup_skip_url("https://zhuanlan.zhihu.com/p/2080013448440816928") is False
    assert _followup_skip_url("https://www.zhihu.com/topics") is True
    assert _followup_skip_url("https://www.zhihu.com/explore") is True
    assert _followup_skip_url("https://github.com/justjavac/zhihu-trending-hot-questions") is True
    assert _followup_skip_url("https://prefixx.com/post") is False


def test_related_cache_does_not_reuse_zhihu_scope_for_elsewhere() -> None:
    assert _related_cache_scope_mismatch("知乎 astra", "astra ai 算力 创业公司 最新动态") is True
    assert _related_cache_scope_mismatch("美国伊朗冲突 最新", "美国伊朗现在冲突") is False


@pytest.mark.anyio
async def test_followup_reads_zhihu_article_and_rejects_login_wall(monkeypatch) -> None:
    reader = _FakeUrlReader(
        [
            UrlReadResult(
                "ok",
                "https://zhuanlan.zhihu.com/p/wall",
                final_url="https://zhuanlan.zhihu.com/p/wall",
                title="打开知乎App",
                text="打开知乎App 验证码登录 密码登录 获取短信验证码",
            ),
            UrlReadResult(
                "ok",
                "https://zhuanlan.zhihu.com/p/astra",
                final_url="https://zhuanlan.zhihu.com/p/astra",
                title="GPT-6 Astra",
                text="Astra 是跟算力迭代有关的模型，不是热榜条目。",
            ),
        ]
    )

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "GPT-6 Astra 讨论", (
            FreshItem("hub", "zhihu.com", "", url="https://www.zhihu.com/topics"),
            FreshItem("wall", "zhihu.com", "", url="https://zhuanlan.zhihu.com/p/wall"),
            FreshItem("astra", "zhihu.com", "", url="https://zhuanlan.zhihu.com/p/astra"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=reader,
        followup_search_hops=1,
    )
    lookup = await tool.lookup("知乎 astra", kind="web")
    context = _prompt_context_from_lookup(lookup)

    assert set(reader.urls) == {
        "https://zhuanlan.zhihu.com/p/wall",
        "https://zhuanlan.zhihu.com/p/astra",
    }
    assert "Astra 是跟算力迭代有关的模型" in lookup.page_text
    assert "网页正文" in context
    assert "热门话题" not in context


@pytest.mark.anyio
async def test_related_cache_does_not_copy_page_text_across_queries() -> None:
    tool = FreshContextTool(max_external_queries_per_minute=0, cache_ttl_seconds=60)
    lookup = FreshLookup(
        "美国伊朗冲突 最新",
        "news",
        (FreshItem("美国伊朗冲突最新进展", "示例媒体", "2026-08-03", url="https://news.example.com/a"),),
        "ok",
        provider="test",
        answer="双方局势仍在变化。",
        page_url="https://news.example.com/a",
        page_text="这是上一轮正文，不该带到相似查询。",
        page_status="ok",
    )
    tool._cache[("news", "美国伊朗冲突 最新")] = (time.monotonic(), lookup)
    reused = await tool.lookup("美国伊朗现在冲突", kind="news")
    assert reused.cached
    assert reused.provider == "test:related_cache"
    assert reused.answer == "双方局势仍在变化。"
    assert reused.page_text == ""
    assert reused.page_url == ""
    assert reused.error == ""
    context = _prompt_context_from_lookup(reused)
    assert "复用上一跳条目" in context
    assert "部分信息源失败" not in context


def test_compact_search_query_keeps_short_terms_and_drops_title_stopwords() -> None:
    title = "让触觉成为具身智能的关键能力：如何成为下一代机器人的感知底座"
    compacted = _compact_search_query(title)
    compact_blob = compacted.replace(" ", "")
    assert "触觉" in compacted
    assert "具身智能" in compacted
    assert compact_blob != "让"
    assert compacted.count("让") == 0
    assert _compact_search_query("astra GPT-6 知乎") == "astra GPT-6 知乎"
    intent = detect_fresh_intent("搜一下知乎上面的 astra")
    assert intent is not None
    assert "astra" in intent.query.casefold()


def test_low_quality_filter_uses_host_not_title_substring() -> None:
    assert _looks_like_low_quality_result("HG 高达模型评测", "https://zhuanlan.zhihu.com/p/1") is False
    assert _looks_like_low_quality_result("最新预测", "https://news.example.com/a") is False
    assert _looks_like_low_quality_result("Live posts & updates", "https://x.com/foo") is True
    assert _looks_like_low_quality_result("赛果", "https://twitter.com/foo") is True


def test_rank_followup_items_prefers_query_host_and_wikipedia() -> None:
    items = (
        FreshItem("杂讯", "example.com", "2026-09-01", url="https://example.com/noise"),
        FreshItem("Astra", "zhihu.com", "", url="https://zhuanlan.zhihu.com/p/astra"),
        FreshItem("Astra", "wikipedia.org", "", url="https://zh.wikipedia.org/wiki/OpenAI"),
    )
    ranked = _rank_followup_items(items, query="知乎 astra")
    assert ranked[0].url == "https://zhuanlan.zhihu.com/p/astra"
    wiki_first = _rank_followup_items(items, query="OpenAI 是什么")
    assert wiki_first[0].url == "https://zh.wikipedia.org/wiki/OpenAI"
    assert _followup_skip_url("https://zh.wikipedia.org/wiki/OpenAI") is False
    assert _wikipedia_title_from_url("https://zh.wikipedia.org/wiki/OpenAI") == ("zh", "OpenAI")
    assert "维基百科" in _second_hop_query("Astra 是什么")


@pytest.mark.anyio
async def test_followup_reads_two_pages_in_parallel(monkeypatch) -> None:
    reader = _FakeUrlReader(
        [
            UrlReadResult(
                "ok",
                "https://example.com/a",
                final_url="https://example.com/a",
                title="A",
                text="第一页写 130.5 billion。",
            ),
            UrlReadResult(
                "ok",
                "https://example.com/b",
                final_url="https://example.com/b",
                title="B",
                text="第二页写 Agent 模型。",
            ),
        ]
    )

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "摘要", (
            FreshItem("A", "example.com", "", url="https://example.com/a"),
            FreshItem("B", "example.com", "", url="https://example.com/b"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=reader,
        followup_search_hops=1,
    )
    lookup = await tool.lookup("NVIDIA", kind="web")
    context = _prompt_context_from_lookup(lookup)
    assert reader.urls == ["https://example.com/a", "https://example.com/b"]
    assert lookup.page_texts[0].startswith("第一页")
    assert "第二页写 Agent 模型" in lookup.page_texts[1]
    assert "[S1 正文]" in context
    assert "[S2 正文]" in context


@pytest.mark.anyio
async def test_wikipedia_extract_uses_api_not_html(monkeypatch) -> None:
    class _NoHtmlReader:
        async def read(self, url: str) -> UrlReadResult:
            if "wikipedia.org" in url:
                raise AssertionError(f"should not scrape html {url}")
            return UrlReadResult("fetch_error", url, error="unused")

    async def fake_wiki(url: str, *, timeout_seconds: float):
        if "wikipedia.org" not in url:
            return None
        return UrlReadResult(
            "ok",
            url,
            final_url=url,
            title="OpenAI",
            text="OpenAI 是一家人工智能研究公司。",
            content_type="application/json",
        )

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "", (
            FreshItem("OpenAI", "wikipedia.org", "", url="https://zh.wikipedia.org/wiki/OpenAI"),
            FreshItem("杂讯", "example.com", "", url="https://example.com/noise"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    monkeypatch.setattr(fresh_context, "_read_wikipedia_extract", fake_wiki)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        url_reader=_NoHtmlReader(),
        followup_search_hops=1,
    )
    lookup = await tool.lookup("OpenAI 是什么", kind="web")
    assert "人工智能研究公司" in lookup.page_text
    assert lookup.page_url == "https://zh.wikipedia.org/wiki/OpenAI"

def test_research_queries_split_web_topic_into_angles() -> None:
    queries = fresh_context._research_queries("OpenFOAM 简介", kind="web")
    assert queries[0] == "OpenFOAM 简介"
    assert 2 <= len(queries) <= 4
    assert any("维基百科" in item or "是什么" in item for item in queries)


def test_detect_fresh_intent_drops_weak_search_object() -> None:
    assert detect_fresh_intent("搜一下然后开始想个思路") is None
    intent = detect_fresh_intent("搜一下 OpenFOAM 简介")
    assert intent is not None
    assert "OpenFOAM" in intent.query


@pytest.mark.anyio
async def test_lookup_runs_parallel_angle_queries_without_polluting_attempted_providers(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        seen.append(query)
        return "OpenFOAM 是开源 CFD 工具包。", (
            FreshItem(
                "OpenFOAM",
                "openfoam.org",
                "",
                summary="The OpenFOAM Foundation",
                url="https://openfoam.org",
            ),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=1,
    )
    lookup = await tool.lookup("OpenFOAM 简介", kind="web")
    assert lookup.status == "ok"
    assert lookup.provider == "tavily"
    assert lookup.attempted_providers == ("tavily",)
    assert len(lookup.research_queries) >= 2
    assert len(seen) >= 2
    assert "OpenFOAM 是开源 CFD 工具包" in lookup.answer

def test_should_run_second_research_round_only_when_evidence_is_thin() -> None:
    covered = (
        FreshItem("OpenFOAM 简介", "openfoam.org", "", url="https://openfoam.org/docs"),
    )
    pages = (
        UrlReadResult(
            "ok",
            "https://openfoam.org/docs",
            final_url="https://openfoam.org/docs",
            title="OpenFOAM",
            text="OpenFOAM is an open source CFD toolbox.",
        ),
    )
    assert (
        fresh_context._should_run_second_research_round(
            query="OpenFOAM 简介",
            kind="web",
            items=covered,
            pages=pages,
            hops=2,
            remaining_seconds=3.0,
        )
        is False
    )
    mismatch = (FreshItem("无关标题", "example.com", "", url="https://example.com/x"),)
    assert (
        fresh_context._should_run_second_research_round(
            query="OpenFOAM 简介",
            kind="web",
            items=mismatch,
            pages=(),
            hops=2,
            remaining_seconds=3.0,
        )
        is True
    )
    assert (
        fresh_context._should_run_second_research_round(
            query="OpenFOAM 简介",
            kind="web",
            items=mismatch,
            pages=(),
            hops=1,
            remaining_seconds=3.0,
        )
        is False
    )


def test_research_round2_queries_are_new_angles() -> None:
    used = ("OpenFOAM 简介", "OpenFOAM 简介 是什么")
    queries = fresh_context._research_round2_queries("OpenFOAM 简介", kind="web", used_queries=used)
    assert queries
    assert all(item not in used for item in queries)
    assert any("官网" in item or "wikipedia" in item.lower() for item in queries)


@pytest.mark.anyio
async def test_lookup_second_round_runs_once_when_first_round_misses(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        seen.append(query)
        if "维基" in query or "官网" in query:
            return "OpenFOAM 是开源 CFD 工具包。", (
                FreshItem(
                    "OpenFOAM",
                    "openfoam.org",
                    "",
                    summary="Open source CFD toolbox",
                    url="https://www.openfoam.com",
                ),
            )
        return "无关摘要", (
            FreshItem("杂讯", "example.com", "", summary="noise", url="https://example.com/noise"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=2,
        timeout_seconds=8,
    )
    lookup = await tool.lookup("OpenFOAM 简介", kind="web")
    assert lookup.status == "ok"
    assert lookup.research_rounds == 2
    assert any(item.endswith(":round2") for item in lookup.attempted_providers)
    assert any("维基" in query or "官网" in query for query in seen)
    assert lookup.research_rounds <= 3


def test_planned_research_queries_prefer_model_angles() -> None:
    queries = fresh_context._planned_research_queries(
        "OpenFOAM 简介",
        kind="web",
        planned=("OpenFOAM 官方文档", "OpenFOAM CFD toolbox"),
    )
    assert queries[0] == "OpenFOAM 简介"
    assert "OpenFOAM 官方文档" in queries
    assert len(queries) <= 4


@pytest.mark.anyio
async def test_lookup_uses_model_queries_as_first_round(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        seen.append(query)
        return "OpenFOAM 是开源 CFD 工具包。", (
            FreshItem(
                "OpenFOAM",
                "openfoam.org",
                "",
                summary="Open source CFD toolbox",
                url="https://openfoam.org",
            ),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=1,
    )
    lookup = await tool.lookup(
        "OpenFOAM 简介",
        kind="web",
        queries=("OpenFOAM 简介", "OpenFOAM 官方文档", "OpenFOAM CFD toolbox"),
    )
    assert lookup.status == "ok"
    assert lookup.research_queries[0] == "OpenFOAM 简介"
    assert "OpenFOAM 官方文档" in lookup.research_queries
    assert "OpenFOAM 官方文档" in seen


def test_prompt_context_keeps_source_ids_aligned_with_page_bodies() -> None:
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
            page_url="https://nonebot.dev/docs/plugin",
            page_text="Create a plugin with a matcher.",
            page_urls=("https://nonebot.dev/docs/plugin",),
            page_texts=("Create a plugin with a matcher.",),
            research_queries=("NoneBot 文档", "NoneBot 官方文档"),
            research_rounds=1,
        )
    )

    assert "<fresh_research" in context
    assert "<sources>" in context
    assert "[S1 正文] https://nonebot.dev/docs/plugin" in context
    assert "检索词：" in context
    assert "NoneBot 官方文档" in context


def test_should_run_followup_research_round_caps_at_three() -> None:
    mismatch = (FreshItem("无关标题", "example.com", "", url="https://example.com/x"),)
    assert fresh_context._should_run_followup_research_round(
        query="OpenFOAM 简介",
        kind="web",
        items=mismatch,
        pages=(),
        hops=3,
        remaining_seconds=3.0,
        current_round=1,
    )
    assert fresh_context._should_run_followup_research_round(
        query="OpenFOAM 简介",
        kind="web",
        items=mismatch,
        pages=(),
        hops=3,
        remaining_seconds=3.0,
        current_round=2,
    )
    assert not fresh_context._should_run_followup_research_round(
        query="OpenFOAM 简介",
        kind="web",
        items=mismatch,
        pages=(),
        hops=3,
        remaining_seconds=3.0,
        current_round=3,
    )


def test_research_round3_queries_are_new_angles() -> None:
    used = ("OpenFOAM 简介", "OpenFOAM 官网", "OpenFOAM wikipedia")
    queries = fresh_context._research_followup_queries(
        "OpenFOAM 简介",
        kind="web",
        used_queries=used,
        round_index=3,
    )
    assert queries
    assert all(item not in used for item in queries)


@pytest.mark.anyio
async def test_lookup_third_round_runs_when_second_round_still_misses(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        seen.append(query)
        if "英文" in query or "site:wikipedia.org" in query:
            return "OpenFOAM is an open source CFD toolbox.", (
                FreshItem(
                    "OpenFOAM",
                    "openfoam.org",
                    "",
                    summary="Open source CFD toolbox",
                    url="https://www.openfoam.com",
                ),
            )
        return "无关摘要", (
            FreshItem("杂讯", "example.com", "", summary="noise", url="https://example.com/noise"),
        )

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=3,
        timeout_seconds=8,
    )
    lookup = await tool.lookup("OpenFOAM 简介", kind="web")
    assert lookup.status == "ok"
    assert lookup.research_rounds == 3
    assert any(item.endswith(":round3") for item in lookup.attempted_providers)
    assert any("英文" in query or "wikipedia.org" in query for query in seen)



@pytest.mark.anyio
async def test_lookup_drops_results_when_judge_says_useless(monkeypatch) -> None:
    async def fake_tavily(query: str, *, kind: str, api_key: str):
        return "今日股市大涨", (
            FreshItem("股市", "example.com", "", summary="noise", url="https://example.com/noise"),
        )

    class RejectJudge:
        async def judge_search_useful(self, *, query: str, evidence: str):
            return False, "jev_search_useful_0.10"

        async def should_followup_search(self, **kwargs):
            return False

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=3,
        timeout_seconds=8,
        research_judge=RejectJudge(),
    )
    lookup = await tool.lookup("OpenFOAM 简介", kind="web")
    assert lookup.status == "no_result"
    assert lookup.items == ()
    assert "jev_search_useful" in lookup.error


@pytest.mark.anyio
async def test_lookup_jev_can_request_second_round(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_tavily(query: str, *, kind: str, api_key: str):
        seen.append(query)
        if "官方" in query or "路透" in query or "英文" in query or "wikipedia" in query:
            return "OpenFOAM is an open source CFD toolbox.", (
                FreshItem(
                    "OpenFOAM",
                    "openfoam.org",
                    "",
                    summary="Open source CFD toolbox",
                    url="https://www.openfoam.com",
                ),
            )
        return "OpenFOAM 简介", (
            FreshItem("OpenFOAM 简介", "example.com", "", summary="简介页面", url="https://example.com/of"),
        )

    class HopJudge:
        async def judge_search_useful(self, *, query: str, evidence: str):
            return True, "jev_search_useful_0.90"

        async def should_followup_search(self, **kwargs):
            return kwargs["current_round"] < 2

    monkeypatch.setattr(fresh_context, "_fetch_tavily_lookup", fake_tavily)
    tool = FreshContextTool(
        provider="tavily",
        tavily_api_key="test-key",
        followup_page_max_tries=0,
        followup_search_hops=3,
        timeout_seconds=8,
        research_judge=HopJudge(),
    )
    lookup = await tool.lookup("OpenFOAM 简介", kind="web")
    assert lookup.status == "ok"
    assert lookup.research_rounds >= 2
    assert len(seen) > 1
