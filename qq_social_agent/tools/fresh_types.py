"""Shared search records and provider error type."""
from __future__ import annotations
from dataclasses import dataclass


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
    attempted_providers: tuple[str, ...] = ()
    latency_ms: int = 0
    error: str = ""
    page_url: str = ""
    page_title: str = ""
    page_text: str = ""
    page_status: str = ""
    page_error: str = ""
    page_urls: tuple[str, ...] = ()
    page_texts: tuple[str, ...] = ()
    research_queries: tuple[str, ...] = ()
    research_rounds: int = 1


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
    source_refs: tuple[str, ...] = ()
    page_text: str = ""
    page_url: str = ""
    page_texts: tuple[str, ...] = ()
    page_urls: tuple[str, ...] = ()
    research_queries: tuple[str, ...] = ()
    research_rounds: int = 1


@dataclass(frozen=True)
class FreshIntent:
    query: str
    kind: str
    explicit: bool = False
    required: bool = False


class SearchProviderError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
