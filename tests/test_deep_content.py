import asyncio

from qq_social_agent.tools.deep_content import DeepContentTool, explicit_deep_read_requested, first_http_url
from qq_social_agent.tools.safe_url_reader import UrlReadResult


class FakeReader:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def read(self, url: str) -> UrlReadResult:
        self.urls.append(url)
        return UrlReadResult("ok", url, final_url="https://example.com/a", text="正文")

    def status_snapshot(self) -> dict[str, object]:
        return {"urls": len(self.urls)}

    async def aclose(self) -> None:
        return None


def test_url_extraction_strips_chinese_punctuation() -> None:
    assert first_http_url("看看 https://example.com/a?q=1。") == "https://example.com/a?q=1"


def test_unaddressed_bare_url_is_not_fetched() -> None:
    reader = FakeReader()
    tool = DeepContentTool(reader)  # type: ignore[arg-type]

    result = asyncio.run(tool.context_for_text("https://example.com/a", addressed_bot=False))

    assert not result.requested
    assert result.reason == "context_not_allowed"
    assert reader.urls == []


def test_addressed_or_explicit_url_is_fetched_once() -> None:
    reader = FakeReader()
    tool = DeepContentTool(reader)  # type: ignore[arg-type]

    addressed = asyncio.run(
        tool.context_for_text("风雪看 https://example.com/a", addressed_bot=True)
    )
    explicit = asyncio.run(
        tool.context_for_text("总结网页 https://example.com/b", addressed_bot=False)
    )

    assert addressed.context.endswith("正文")
    assert explicit.requested
    assert reader.urls == ["https://example.com/a", "https://example.com/b"]
    assert explicit_deep_read_requested("帮我总结网页")
