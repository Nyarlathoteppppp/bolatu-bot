from __future__ import annotations

import re
from dataclasses import dataclass

from .safe_url_reader import SafeUrlReader, SafeUrlReaderConfig, UrlReadResult


_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,!?;:)]}>，。！？；：）】》」』"


@dataclass(frozen=True)
class DeepContentResult:
    requested: bool
    url: str = ""
    read: UrlReadResult | None = None
    reason: str = ""

    @property
    def context(self) -> str:
        return self.read.to_context() if self.read is not None else ""


class DeepContentTool:
    def __init__(self, reader: SafeUrlReader):
        self.reader = reader

    @classmethod
    def from_config(cls, raw: object) -> "DeepContentTool":
        return cls(SafeUrlReader(SafeUrlReaderConfig.from_config(raw)))

    async def context_for_text(
        self,
        text: str,
        *,
        addressed_bot: bool,
        force: bool = False,
    ) -> DeepContentResult:
        url = first_http_url(text)
        if not url:
            return DeepContentResult(False, reason="no_url")
        if not (force or addressed_bot or explicit_deep_read_requested(text)):
            return DeepContentResult(False, url=url, reason="context_not_allowed")
        result = await self.reader.read(url)
        return DeepContentResult(True, url=url, read=result, reason=result.status)

    def status_snapshot(self) -> dict[str, object]:
        return self.reader.status_snapshot()

    async def aclose(self) -> None:
        await self.reader.aclose()


def first_http_url(text: str) -> str:
    match = _URL_RE.search(str(text or ""))
    if match is None:
        return ""
    return match.group(0).rstrip(_TRAILING_PUNCTUATION)


def explicit_deep_read_requested(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).casefold()
    return any(
        token in compact
        for token in (
            "看看这个链接",
            "看下这个链接",
            "读一下",
            "读下网页",
            "总结链接",
            "总结网页",
            "网页内容",
            "链接里说",
            "这篇文章",
        )
    )
