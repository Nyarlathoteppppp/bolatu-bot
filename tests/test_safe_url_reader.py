import asyncio

import httpx

from qq_social_agent.tools.safe_url_reader import (
    SafeUrlReader,
    SafeUrlReaderConfig,
    extract_html_text,
)


async def _public_resolver(host: str, port: int) -> tuple[str, ...]:
    return ("93.184.216.34",)


def _run_read(
    url: str,
    handler,
    *,
    config: SafeUrlReaderConfig | None = None,
    resolver=_public_resolver,
):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = SafeUrlReader(config, client=client, resolver=resolver)
            result = await reader.read(url)
            return result, reader.status_snapshot()

    return asyncio.run(scenario())


def test_reader_rejects_non_http_credentials_and_private_literal_ips() -> None:
    def unused_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked URLs must not reach transport")

    for url in (
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://user:pass@example.com/",
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://example.com:8080/admin",
    ):
        result, _ = _run_read(url, unused_handler)
        assert not result.ok, url
        assert result.status in {"blocked", "invalid_url"}


def test_reader_rejects_dns_with_any_private_or_link_local_address() -> None:
    async def mixed_resolver(host: str, port: int) -> tuple[str, ...]:
        return ("93.184.216.34", "10.0.0.2")

    def unused_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("mixed DNS answer must be blocked")

    result, _ = _run_read("https://example.com/", unused_handler, resolver=mixed_resolver)

    assert result.status == "blocked"
    assert result.error == "non_public_ip"


def test_reader_revalidates_redirect_and_blocks_private_target() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})

    result, _ = _run_read("https://example.com/start", handler)

    assert result.status == "blocked"
    assert result.error == "non_public_ip"
    assert result.redirects == 1
    assert len(calls) == 1


def test_reader_follows_public_relative_redirect_and_extracts_html() -> None:
    html = b"""
    <html><head><title> Test Page </title><style>.bad{display:none}</style></head>
    <body><main><h1>Hello</h1><p>QQ group content.</p></main>
    <script>steal_token()</script></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/article"})
        return httpx.Response(200, content=html, headers={"Content-Type": "text/html; charset=utf-8"})

    result, status = _run_read("https://example.com/start", handler)

    assert result.ok
    assert result.redirects == 1
    assert result.final_url == "https://example.com/article"
    assert result.title == "Test Page"
    assert "Hello" in result.text
    assert "QQ group content." in result.text
    assert "steal_token" not in result.text
    assert "display:none" not in result.text
    assert "不可信外部数据" in result.to_context()
    assert status["counters"]["successes"] == 1


def test_reader_enforces_content_type_and_declared_size() -> None:
    def binary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"png", headers={"Content-Type": "image/png"})

    result, _ = _run_read("https://example.com/a.png", binary_handler)
    assert result.status == "unsupported_content_type"

    def oversized_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"small",
            headers={"Content-Type": "text/plain", "Content-Length": "9999"},
        )

    config = SafeUrlReaderConfig(max_bytes=64)
    result, status = _run_read("https://example.com/a.txt", oversized_handler, config=config)
    assert result.status == "too_large"
    assert result.error == "content_length_exceeded"
    assert status["counters"]["too_large"] == 1


def test_reader_enforces_streamed_body_size_and_timeout() -> None:
    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 40
            yield b"y" * 40

    def large_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=LargeStream(), headers={"Content-Type": "text/plain"})

    result, _ = _run_read(
        "https://example.com/a.txt",
        large_handler,
        config=SafeUrlReaderConfig(max_bytes=64),
    )
    assert result.status == "too_large"
    assert result.error == "body_size_exceeded"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    result, _ = _run_read("https://example.com/slow", timeout_handler)
    assert result.status == "timeout"
    assert result.error == "request_timeout"


def test_reader_truncates_text_and_status_hides_url_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="a" * 500, headers={"Content-Type": "text/plain"})

    result, status = _run_read(
        "https://example.com/article?token=super-secret",
        handler,
        config=SafeUrlReaderConfig(max_text_chars=256),
    )

    assert result.ok
    assert result.truncated
    assert len(result.text) == 256
    assert "super-secret" not in str(status)
    assert status["last_request"]["url"] == "https://example.com/article"


def test_extract_html_text_has_readable_blocks() -> None:
    title, text = extract_html_text("<title>A &amp; B</title><p>first</p><p>second<br>line</p>")

    assert title == "A & B"
    assert text.splitlines() == ["first", "second", "line"]


def test_safe_url_reader_config_bounds_values() -> None:
    config = SafeUrlReaderConfig.from_config(
        {
            "timeout_seconds": 999,
            "max_bytes": 1,
            "max_redirects": 99,
            "allowed_ports": [443, "8443", -1],
            "allowed_content_types": ["TEXT/HTML; charset=utf-8", "text/plain"],
        }
    )

    assert config.timeout_seconds == 30.0
    assert config.max_bytes == 1_024
    assert config.max_redirects == 8
    assert config.allowed_ports == (443, 8443)
    assert config.allowed_content_types == ("text/html", "text/plain")
