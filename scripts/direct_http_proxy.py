#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
from urllib.parse import urlsplit


READ_LIMIT = 1024 * 1024


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _open_target(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(host, port)


def _parse_connect_target(target: str) -> tuple[str, int] | None:
    if ":" not in target:
        return None
    host, raw_port = target.rsplit(":", 1)
    host = host.strip("[]")
    try:
        port = int(raw_port)
    except ValueError:
        return None
    if not host or port <= 0 or port > 65535:
        return None
    return host, port


async def _handle_connect(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target: str,
    buffered: bytes,
) -> None:
    parsed = _parse_connect_target(target)
    if parsed is None:
        client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await client_writer.drain()
        return
    host, port = parsed
    try:
        remote_reader, remote_writer = await _open_target(host, port)
    except OSError:
        client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await client_writer.drain()
        return
    client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await client_writer.drain()
    if buffered:
        remote_writer.write(buffered)
        await remote_writer.drain()
    await asyncio.gather(
        _pipe(client_reader, remote_writer),
        _pipe(remote_reader, client_writer),
    )


async def _handle_http(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    method: str,
    target: str,
    version: str,
    headers: bytes,
    buffered: bytes,
) -> None:
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await client_writer.drain()
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    try:
        remote_reader, remote_writer = await _open_target(parsed.hostname, port)
    except OSError:
        client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await client_writer.drain()
        return
    request_line = f"{method} {path} {version}\r\n".encode()
    remote_writer.write(request_line + headers + b"\r\n" + buffered)
    await remote_writer.drain()
    await asyncio.gather(
        _pipe(client_reader, remote_writer),
        _pipe(remote_reader, client_writer),
    )


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        writer.close()
        await writer.wait_closed()
        return
    lines = head.split(b"\r\n")
    try:
        method, target, version = lines[0].decode("latin1").split(" ", 2)
    except ValueError:
        writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return
    headers = b"\r\n".join(lines[1:-2])
    buffered = b""
    if method.upper() == "CONNECT":
        await _handle_connect(reader, writer, target, buffered)
    else:
        await _handle_http(reader, writer, method, target, version, headers, buffered)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Small direct HTTP CONNECT proxy for remote Codex.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7897)
    args = parser.parse_args()
    server = await asyncio.start_server(
        _handle_client,
        args.host,
        args.port,
        limit=READ_LIMIT,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"direct_http_proxy listening on {sockets}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
