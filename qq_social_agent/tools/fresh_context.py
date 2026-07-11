from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import httpx


@dataclass(frozen=True)
class FreshItem:
    title: str
    source: str
    published_at: str
    summary: str = ""
    url: str = ""
    score: float | None = None

    def to_prompt_line(self) -> str:
        parts = [self.title]
        if self.source:
            parts.append(f"来源 {self.source}")
        if self.published_at:
            parts.append(f"时间 {self.published_at}")
        if self.summary:
            parts.append(f"摘要 {self.summary}")
        return "- " + "，".join(parts)


@dataclass(frozen=True)
class FreshLookup:
    query: str
    kind: str
    items: tuple[FreshItem, ...]
    status: str
    provider: str = "google_news"
    answer: str = ""
    cached: bool = False


@dataclass(frozen=True)
class FreshFactPack:
    topic: str
    kind: str
    provider: str
    status: str
    freshness: str
    facts: tuple[str, ...]
    uncertain: tuple[str, ...]
    sources: tuple[str, ...]
    cached: bool = False


@dataclass(frozen=True)
class FreshIntent:
    query: str
    kind: str


class FreshContextTool:
    def __init__(
        self,
        *,
        max_external_queries_per_minute: int = 2,
        cache_ttl_seconds: int = 10 * 60,
        failure_ttl_seconds: int = 2 * 60,
        provider: str | None = None,
        tavily_api_key: str | None = None,
    ):
        self.max_external_queries_per_minute = max_external_queries_per_minute
        self.cache_ttl_seconds = cache_ttl_seconds
        self.failure_ttl_seconds = failure_ttl_seconds
        self.provider = (provider or os.getenv("FRESH_SEARCH_PROVIDER") or "auto").strip().lower()
        self.tavily_api_key = (tavily_api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        self._cache: dict[tuple[str, str], tuple[float, FreshLookup]] = {}
        self._query_times: list[float] = []

    async def context_for(self, query: str, *, kind: str = "news") -> str:
        lookup = await self.lookup(query, kind=kind)
        return _prompt_context_from_fact_pack(fact_pack_from_lookup(lookup))

    async def lookup(self, query: str, *, kind: str = "news") -> FreshLookup:
        normalized_query = _normalize_query(query)
        normalized_kind = kind if kind in {"news", "sports", "web"} else "news"
        if not normalized_query:
            return FreshLookup(query, normalized_kind, (), "empty_query")

        key = (normalized_kind, normalized_query)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached:
            cached_at, lookup = cached
            ttl = self.cache_ttl_seconds if lookup.items else self.failure_ttl_seconds
            if now - cached_at <= ttl:
                return FreshLookup(
                    lookup.query,
                    lookup.kind,
                    lookup.items,
                    lookup.status,
                    provider=lookup.provider,
                    answer=lookup.answer,
                    cached=True,
                )

        if not self._allow_external_query(now):
            return FreshLookup(normalized_query, normalized_kind, (), "rate_limited")

        provider = self._resolved_provider()
        answer = ""
        if provider == "tavily":
            answer, items = await _fetch_tavily_lookup(
                normalized_query,
                kind=normalized_kind,
                api_key=self.tavily_api_key,
            )
            if not items and self.provider == "auto":
                provider = "google_news"
                items = await _fetch_google_news_items(normalized_query, kind=normalized_kind)
        else:
            items = await _fetch_google_news_items(normalized_query, kind=normalized_kind)
        status = "ok" if items or answer else "failed"
        lookup = FreshLookup(normalized_query, normalized_kind, items, status, provider=provider, answer=answer)
        self._cache[key] = (now, lookup)
        return lookup

    def _resolved_provider(self) -> str:
        if self.provider == "tavily":
            return "tavily"
        if self.provider == "google_news":
            return "google_news"
        if self.provider == "auto" and self.tavily_api_key:
            return "tavily"
        return "google_news"

    def _allow_external_query(self, now: float) -> bool:
        self._query_times = [t for t in self._query_times if now - t < 60]
        if len(self._query_times) >= self.max_external_queries_per_minute:
            return False
        self._query_times.append(now)
        return True


def _prompt_context_from_lookup(lookup: FreshLookup) -> str:
    return _prompt_context_from_fact_pack(fact_pack_from_lookup(lookup))


def fact_pack_from_lookup(lookup: FreshLookup) -> FreshFactPack:
    if lookup.status == "empty_query":
        return FreshFactPack(
            topic=lookup.query,
            kind=lookup.kind,
            provider=lookup.provider,
            status="empty_query",
            freshness="无查询词",
            facts=(),
            uncertain=("没有可用查询词。",),
            sources=(),
            cached=lookup.cached,
        )
    if lookup.status == "rate_limited":
        return FreshFactPack(
            topic=lookup.query,
            kind=lookup.kind,
            provider=lookup.provider,
            status="rate_limited",
            freshness="本分钟查询已达上限",
            facts=(),
            uncertain=("外部信息源限流；不要编造最新事实。",),
            sources=(),
            cached=lookup.cached,
        )
    facts: list[str] = []
    uncertain: list[str] = []
    sources: list[str] = []
    if lookup.answer:
        facts.append(f"快速摘要：{lookup.answer}")
    for item in lookup.items[:4]:
        fact_parts = [item.title]
        if item.published_at:
            fact_parts.append(f"时间 {item.published_at}")
        if item.summary:
            fact_parts.append(f"摘要 {item.summary}")
        facts.append("，".join(fact_parts))
        source = item.source or _source_from_url(item.url)
        if source:
            sources.append(source)
    if not facts:
        uncertain.append(f"查询“{lookup.query}”没有拿到可靠结果。")
    if len(set(sources)) <= 1 and facts:
        uncertain.append("来源较少，不能把单条摘要当成绝对事实。")
    return FreshFactPack(
        topic=lookup.query,
        kind=lookup.kind,
        provider=lookup.provider,
        status=lookup.status if facts else "no_result",
        freshness=_freshness_label(lookup),
        facts=tuple(facts[:5]),
        uncertain=tuple(_dedupe_strings(uncertain)[:4]),
        sources=tuple(_dedupe_strings(sources)[:5]),
        cached=lookup.cached,
    )


def _prompt_context_from_fact_pack(pack: FreshFactPack) -> str:
    if pack.status == "empty_query":
        return ""
    if pack.status == "rate_limited":
        return (
            "最新背景信息：本分钟外部信息源查询已达上限；这不是没有网络。"
            "回复时不要编造最新事实，不要说“没联网”；可以说这类刚发生的事需要等可靠消息。"
        )
    if pack.status in {"failed", "no_result"} or (not pack.facts and pack.uncertain):
        return (
            f"最新背景信息：信息源可用，但查询“{pack.topic}”没有拿到可靠结果。"
            "回复时不要编造最新事实，不要说“没联网”；可以承认没拿到可靠新消息。"
        )

    lines = [
        (
            "最新背景信息"
            f"（查询：{pack.topic}；类型：{pack.kind}；来源：{pack.provider}；"
            "只当背景，不要播报搜索过程）："
        ),
        f"状态：{pack.status}；时效：{pack.freshness}",
    ]
    if pack.sources:
        lines.append(f"来源：{'、'.join(pack.sources)}")
    if pack.facts:
        lines.append("事实背景：")
        lines.extend(f"- {fact}" for fact in pack.facts[:4])
    if pack.uncertain:
        lines.append("不确定点：")
        lines.extend(f"- {item}" for item in pack.uncertain[:3])
    lines.append("回复时基于这些背景做短评；优先相信多来源共同支持的信息；不要说“我搜索到/我查到”，不要把单条摘要当成绝对事实。")
    return "\n".join(lines)


def _freshness_label(lookup: FreshLookup) -> str:
    dates = [item.published_at for item in lookup.items if item.published_at]
    if dates:
        return "；".join(dates[:2])
    if lookup.answer:
        return "由信息源快速摘要提供，具体发布时间未知"
    if lookup.status == "rate_limited":
        return "限流"
    return "未知"


def _dedupe_strings(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = _clean_text(item)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


async def _fetch_tavily_lookup(
    query: str,
    *,
    kind: str,
    api_key: str,
) -> tuple[str, tuple[FreshItem, ...]]:
    if not api_key:
        return "", ()
    topic = "news" if kind in {"news", "sports"} else "general"
    payload: dict[str, object] = {
        "query": _tavily_query(query, kind=kind),
        "search_depth": "basic",
        "topic": topic,
        "max_results": 4,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    if kind in {"news", "sports"}:
        payload["time_range"] = "week"
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception:
        return "", ()
    return _parse_tavily_answer(data), _parse_tavily_results(data)


def _tavily_query(query: str, *, kind: str) -> str:
    if kind == "sports":
        return f"{query} 最新赛果 比分"
    if kind == "news":
        return f"{query} 最新消息"
    return query


def _parse_tavily_results(data: object) -> tuple[FreshItem, ...]:
    if not isinstance(data, dict):
        return ()
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return ()
    items: list[FreshItem] = []
    seen: set[str] = set()
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        url = str(raw.get("url") or "").strip()
        content = str(raw.get("content") or "").strip()
        if not title or _looks_like_low_quality_result(title, url):
            continue
        key = _fresh_result_key(title, url)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            FreshItem(
                title=title[:120],
                source=_source_from_url(url)[:40],
                published_at=str(raw.get("published_date") or "")[:40],
                summary=_clean_text(content)[:180],
                url=url[:240],
                score=_as_float(raw.get("score")),
            )
        )
    return tuple(sorted(items, key=_fresh_item_sort_key)[:5])


def _parse_tavily_answer(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    answer = str(data.get("answer") or "").strip()
    if not answer:
        return ""
    return _clean_text(answer)[:260]


async def _fetch_google_news_items(query: str, *, kind: str) -> tuple[FreshItem, ...]:
    search_query = query
    if kind == "sports":
        search_query = f"{query} 比赛 赛果"
    window = "30d" if kind == "sports" else "14d"
    if "when:" not in search_query:
        search_query = f"{search_query} when:{window}"
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(search_query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    )
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 qq-social-agent/0.1"},
            )
            response.raise_for_status()
    except Exception:
        return ()
    return _parse_google_news_rss(response.text)


def _parse_google_news_rss(xml_text: str) -> tuple[FreshItem, ...]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return ()

    items: list[FreshItem] = []
    for item in root.findall("./channel/item"):
        raw_title = _text(item.find("title"))
        if not raw_title:
            continue
        title, source_from_title = _split_title_source(raw_title)
        source = _text(item.find("source")) or source_from_title
        if _looks_like_low_quality_result(title, source):
            continue
        published_at = _format_pub_date(_text(item.find("pubDate")))
        summary = _clean_html(_text(item.find("description")))
        items.append(
            FreshItem(
                title=title[:120],
                source=source[:40],
                published_at=published_at,
                summary=summary[:160],
            )
        )
        if len(items) >= 5:
            break
    return tuple(items)


def _text(node: ElementTree.Element[str] | None) -> str:
    if node is None or node.text is None:
        return ""
    return html.unescape(node.text).strip()


def _split_title_source(title: str) -> tuple[str, str]:
    if " - " not in title:
        return title.strip(), ""
    article_title, source = title.rsplit(" - ", 1)
    return article_title.strip(), source.strip()


def _format_pub_date(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return value[:40]
    return parsed.strftime("%Y-%m-%d %H:%M")


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return _clean_text(text)


def _clean_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc
    except ValueError:
        return ""
    return host.removeprefix("www.")


def _fresh_result_key(title: str, url: str) -> str:
    host = _source_from_url(url).casefold()
    title_key = re.sub(r"\W+", "", title.casefold())[:80]
    return f"{host}:{title_key}"


def _fresh_item_sort_key(item: FreshItem) -> tuple[int, int, float]:
    has_date = 1 if item.published_at else 0
    has_summary = 1 if item.summary else 0
    score = item.score if item.score is not None else 0.0
    return (-has_date, -has_summary, -score)


def _looks_like_low_quality_result(title: str, source: str) -> bool:
    haystack = f"{title} {source}".lower()
    blocked = [
        "x.com",
        "网址",
        "直播地址",
        "results on x",
        "live posts & updates",
        "hg",
        "𝐡",
        "𝐠",
        "博彩",
        "下注",
        "赔率",
        "胜平负",
        "预测",
        "prediction",
        "odds",
    ]
    return any(token in haystack for token in blocked)


def fresh_kind_from_text(text: str) -> str | None:
    intent = detect_fresh_intent(text)
    return intent.kind if intent else None


def detect_fresh_intent(text: str) -> FreshIntent | None:
    normalized = text.lower()
    compact = re.sub(r"\s+", "", normalized)
    if not compact or _is_low_value_fresh_query(compact):
        return None

    latest_terms = (
        "最新",
        "刚刚",
        "发生",
        "新闻",
        "冲突",
        "战争",
        "伊朗",
        "美国",
        "以色列",
        "乌克兰",
        "俄罗斯",
        "政策",
        "发布会",
        "赛果",
        "比分",
        "世界杯",
        "msi",
        "nba",
        "欧冠",
        "英超",
    )
    explicit_search_terms = (
        "搜",
        "搜索",
        "查",
        "查一下",
        "查查",
        "现在",
        "今天",
        "最新",
        "刚刚",
        "新闻",
        "发生什么",
        "怎么了",
        "怎么样了",
        "比分",
        "赛果",
        "赛程",
        "结果",
    )
    if not any(term in normalized for term in latest_terms + explicit_search_terms):
        return None
    sports_terms = ("赛果", "比分", "世界杯", "msi", "nba", "欧冠", "英超", "比赛")
    kind = "sports" if any(term in normalized for term in sports_terms) else "news"
    query = _fresh_query_from_text(text)
    if _is_low_value_fresh_query(query):
        return None
    return FreshIntent(query=query, kind=kind)


def should_use_fresh_context(query: str, fallback_text: str = "") -> bool:
    query = _normalize_query(query)
    if not query or _is_low_value_fresh_query(query):
        return False
    return detect_fresh_intent(f"{query} {fallback_text}") is not None


def _normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    return query[:120]


def _fresh_query_from_text(text: str) -> str:
    query = _normalize_query(text)
    query = re.sub(
        r"^(帮我|你)?(搜一下|搜索一下|搜搜|搜|查一下|查查|查)(一下)?",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    query = re.sub(r"(现在|今天)?(怎么样了|怎么了|是什么情况|咋了|如何了)$", "", query).strip()
    query = re.sub(r"(最新消息|最新新闻|新闻|赛果|比分|结果)$", "", query).strip()
    return _normalize_query(query or text)


def _is_low_value_fresh_query(text: str) -> bool:
    compact = re.sub(r"[\s，。！？,.!?]+", "", text.lower())
    if not compact:
        return True
    low_value_tokens = (
        "你好",
        "美好",
        "测试",
        "周几",
        "星期几",
        "几点",
        "日期",
        "乱码",
        "随便搜搜",
        "你能搜什么",
    )
    if any(token in compact for token in low_value_tokens):
        return True
    return len(compact) <= 2
