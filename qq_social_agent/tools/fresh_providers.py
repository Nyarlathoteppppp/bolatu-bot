"""HTTP search provider adapters and response parsers."""
from __future__ import annotations

import html
import re
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree
import httpx

from .fresh_text import (
    _as_float,
    _clean_html,
    _clean_text,
    _httpx_timeout,
    _normalized_host,
    _source_from_url,
    _wikipedia_host,
)
from .fresh_types import (
    FreshItem,
    SearchProviderError,
)


async def _fetch_sadai_web_lookup(
    query: str,
    *,
    kind: str,
    api_key: str,
    model: str = "gpt-6.1-sol",
    timeout_seconds: float = 30.0,
    max_results: int = 5,
) -> tuple[str, tuple[FreshItem, ...]]:
    """VerySadai Responses with the built-in web_search tool.

    The answer is taken from the last message only: the provider prepends a
    Codex-style "我会先核实…" message before the search call.
    """
    focus = "最新新闻" if kind == "news" else "赛程比分" if kind == "sports" else "资料"
    payload = {
        "model": model,
        "store": False,
        "max_output_tokens": 1200,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "web_search"}],
        "input": [{"role": "user", "content": (
            f"联网检索{focus}并用中文回答：{query}\n"
            "给出关键事实、数字和日期，3 到 6 句；只写查到的内容，查不到就直说。"
        )}],
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://verysadai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"},
            json=payload,
        )
    if response.status_code == 429:
        raise SearchProviderError("rate_limited")
    if response.status_code >= 400:
        raise SearchProviderError(f"http_{response.status_code}")
    data = response.json()
    messages = [item for item in data.get("output") or () if isinstance(item, dict) and item.get("type") == "message"]
    searched = any(isinstance(item, dict) and item.get("type") == "web_search_call" for item in data.get("output") or ())
    if not messages or not searched:
        raise SearchProviderError("no_web_search")
    answer_parts: list[str] = []
    items: list[FreshItem] = []
    seen: set[str] = set()
    for part in messages[-1].get("content") or ():
        if not isinstance(part, dict) or part.get("type") != "output_text":
            continue
        answer_parts.append(str(part.get("text") or ""))
        for note in part.get("annotations") or ():
            url = str((note or {}).get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            items.append(FreshItem(
                title=str(note.get("title") or url)[:160],
                source=urlparse(url).netloc,
                published_at="",
                url=url,
            ))
    answer = _plain_cited_text("".join(answer_parts))
    if not answer:
        raise SearchProviderError("empty_answer")
    return answer, tuple(items[: max(1, max_results)])


def _plain_cited_text(text: str) -> str:
    """Markdown citations -> plain source names, so QQ replies never paste raw links."""
    text = re.sub(r"\(\[([^\]]+)\]\((?:https?://)[^)]+\)\)", r"（\1）", str(text or ""))
    text = re.sub(r"\[([^\]]+)\]\((?:https?://)[^)]+\)", r"\1", text)
    text = text.replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


async def _fetch_tavily_lookup(
    query: str,
    *,
    kind: str,
    api_key: str,
    timeout_seconds: float = 12.0,
    max_results: int = 4,
) -> tuple[str, tuple[FreshItem, ...]]:
    if not api_key:
        raise SearchProviderError("missing_api_key")
    topic = "news" if kind in {"news", "sports"} else "general"
    payload: dict[str, object] = {
        "query": _tavily_query(query, kind=kind),
        "search_depth": "basic",
        "topic": topic,
        "max_results": max(1, min(10, int(max_results))),
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    if kind in {"news", "sports"}:
        payload["time_range"] = "week"
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(timeout_seconds), follow_redirects=True) as client:
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
    except httpx.TimeoutException as exc:
        raise SearchProviderError("timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise SearchProviderError(f"http_{exc.response.status_code}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchProviderError(type(exc).__name__.lower()) from exc
    return _parse_tavily_answer(data), _parse_tavily_results(data)


async def _fetch_searxng_items(
    query: str,
    *,
    kind: str,
    base_url: str,
    timeout_seconds: float = 10.0,
    max_results: int = 5,
) -> tuple[FreshItem, ...]:
    """Use an operator-owned SearXNG JSON endpoint, never a public instance."""
    root = str(base_url or "").strip().rstrip("/")
    if not root.startswith(("http://", "https://")):
        raise SearchProviderError("invalid_base_url")
    params: dict[str, str] = {
        "q": _tavily_query(query, kind=kind),
        "format": "json",
        "language": "zh-CN",
        "safesearch": "0",
    }
    if kind in {"news", "sports"}:
        params["categories"] = "news"
        params["time_range"] = "month"
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(timeout_seconds), follow_redirects=True) as client:
            response = await client.get(
                f"{root}/search",
                params=params,
                headers={"Accept": "application/json", "User-Agent": "qq-social-agent/0.1"},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise SearchProviderError("timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise SearchProviderError(f"http_{exc.response.status_code}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchProviderError(type(exc).__name__.lower()) from exc
    return _parse_searxng_results(data)[:max_results]


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
    return tuple(sorted(items, key=_fresh_item_sort_key)[:10])


def _parse_searxng_results(data: object) -> tuple[FreshItem, ...]:
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
        title = _clean_text(str(raw.get("title") or ""))
        url = str(raw.get("url") or "").strip()
        if not title or not url or _looks_like_low_quality_result(title, url):
            continue
        key = _fresh_result_key(title, url)
        if key in seen:
            continue
        seen.add(key)
        source = _clean_text(str(raw.get("engine") or "")) or _source_from_url(url)
        published_at = _clean_text(str(raw.get("publishedDate") or raw.get("published_at") or ""))[:40]
        items.append(
            FreshItem(
                title=title[:120],
                source=source[:40],
                published_at=published_at,
                summary=_clean_html(str(raw.get("content") or raw.get("snippet") or ""))[:180],
                url=url[:500],
                score=_as_float(raw.get("score")),
            )
        )
    return tuple(sorted(items, key=_fresh_item_sort_key)[:10])


def _parse_tavily_answer(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    answer = str(data.get("answer") or "").strip()
    if not answer:
        return ""
    return _clean_text(answer)[:480]


async def _fetch_google_news_items(
    query: str,
    *,
    kind: str,
    timeout_seconds: float = 10.0,
    max_results: int = 5,
) -> tuple[FreshItem, ...]:
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
        async with httpx.AsyncClient(timeout=_httpx_timeout(timeout_seconds), follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 qq-social-agent/0.1"},
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SearchProviderError("timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise SearchProviderError(f"http_{exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise SearchProviderError(type(exc).__name__.lower()) from exc
    return _parse_google_news_rss(response.text)[:max_results]


async def _fetch_bing_web_items(
    query: str,
    *,
    timeout_seconds: float = 10.0,
    max_results: int = 5,
) -> tuple[FreshItem, ...]:
    url = f"https://www.bing.com/search?format=rss&q={quote_plus(query)}"
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(timeout_seconds), follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 qq-social-agent/0.1"},
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SearchProviderError("timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise SearchProviderError(f"http_{exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise SearchProviderError(type(exc).__name__.lower()) from exc
    return _parse_bing_rss(response.text)[:max_results]


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
        url = _text(item.find("link"))
        items.append(
            FreshItem(
                title=title[:120],
                source=source[:40],
                published_at=published_at,
                summary=summary[:160],
                url=url[:500],
            )
        )
        if len(items) >= 10:
            break
    return tuple(items)


def _parse_bing_rss(xml_text: str) -> tuple[FreshItem, ...]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return ()

    items: list[FreshItem] = []
    seen: set[str] = set()
    for item in root.findall(".//item"):
        title = _text(item.find("title"))
        url = _text(item.find("link"))
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
                published_at=_format_pub_date(_text(item.find("pubDate"))),
                summary=_clean_html(_text(item.find("description")))[:180],
                url=url[:500],
            )
        )
        if len(items) >= 10:
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


def _fresh_result_key(title: str, url: str) -> str:
    host = _source_from_url(url).casefold()
    title_key = re.sub(r"\W+", "", title.casefold())[:80]
    return f"{host}:{title_key}"


def _fresh_item_sort_key(item: FreshItem) -> tuple[int, int, int, int, float]:
    host_score = _host_priority(item.url)
    has_date = 1 if item.published_at else 0
    has_summary = 1 if item.summary else 0
    score = item.score if item.score is not None else 0.0
    return (-host_score, 0, -has_date, -has_summary, -score)


def _host_priority(url: str) -> int:
    host = _normalized_host(_source_from_url(url))
    if not host:
        return 0
    if _wikipedia_host(host):
        return 6
    if host == "github.com" or host.endswith(".github.io"):
        return 5
    if host.endswith(".gov.cn") or host.endswith(".edu.cn"):
        return 4
    if any(host == item or host.endswith("." + item) for item in _PREFERRED_NEWS_HOSTS):
        return 3
    if any(marker in host for marker in ("notes.", "zhihu.com", "bilibili.com", "hupu.com")):
        return 2
    return 0


def _looks_like_low_quality_result(title: str, source: str) -> bool:
    host = _normalized_host(_source_from_url(source) or source)
    title_blob = f"{title} {source}".casefold()
    if host in {"x.com", "twitter.com", "t.co"} or host.endswith(".x.com") or host.endswith(".twitter.com"):
        return True
    if "x.com/" in title_blob or "twitter.com/" in title_blob:
        return True
    blocked_title = (
        "网址",
        "直播地址",
        "results on x",
        "live posts & updates",
        "博彩",
        "下注",
        "赔率",
        "胜平负",
        "prediction",
        "odds",
    )
    if any(token in title_blob for token in blocked_title):
        return True
    return False


_PREFERRED_NEWS_HOSTS = (
    "thepaper.cn",
    "cls.cn",
    "stcn.com",
    "eastmoney.com",
    "sina.com.cn",
    "163.com",
    "qq.com",
    "people.com.cn",
    "xinhuanet.com",
    "gov.cn",
)
