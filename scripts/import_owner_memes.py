#!/usr/bin/env python3
"""One-shot manual import for approved QQ images. This script is never run by the bot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from qq_social_agent.config import PROJECT_ROOT, load_config
from qq_social_agent.memory import MemoryStore
from qq_social_agent.siliconflow_ocr import SiliconFlowOcrClient


OWNER_ID = 1535071184
GROUP_ID = 1026813421
MAX_FILE_BYTES = 3 * 1024 * 1024
ANNOTATION_PROMPT = """你在为一个私人 QQ 表情包库做人工辅助标注。只判断这张图是否适合被聊天机器人作为表情包复用。
不要把文章截图、新闻截图、代码截图、长聊天记录、普通资料图算作表情包。
只输出 JSON：
{"is_meme":true,"description":"这张图在聊天里表达什么，不超过35字","tags":["情绪或场景标签，最多5个"]}
"""


def _clean_json(text: str) -> dict[str, object]:
    body = str(text or "").strip().strip("`")
    if body.startswith("json"):
        body = body[4:].strip()
    match = re.search(r"\{[\s\S]*\}", body)
    if not match:
        return {}
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


async def _download(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ValueError("image_too_large")
                chunks.append(chunk)
    return b"".join(chunks), content_type


def _extension(content_type: str, fallback_name: str = "") -> str:
    guessed = mimetypes.guess_extension(content_type) if content_type else None
    if guessed in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return guessed
    suffix = Path(fallback_name).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".png"


async def _import_one(
    *,
    memory: MemoryStore,
    vision: SiliconFlowOcrClient,
    asset_dir: Path,
    payload: bytes,
    content_type: str,
    source_message_id: str,
    source_name: str,
    manual_annotation: dict[str, object] | None = None,
) -> tuple[bool, str]:
    if not payload or len(payload) > MAX_FILE_BYTES:
        return False, "invalid_size"
    digest = hashlib.sha256(payload).hexdigest()
    suffix = _extension(content_type, source_name)
    target = asset_dir / f"{digest}{suffix}"
    if not target.exists():
        target.write_bytes(payload)
    if manual_annotation is not None:
        annotation = manual_annotation
    else:
        try:
            raw_annotation = await vision.recognize(str(target), prompt=ANNOTATION_PROMPT)
        except Exception as exc:
            return False, f"vision_failed:{type(exc).__name__}"
        annotation = _clean_json(raw_annotation)
    is_meme = bool(annotation.get("is_meme", False))
    description = str(annotation.get("description", "") or "").strip()
    raw_tags = annotation.get("tags", [])
    tags = tuple(str(tag).strip()[:24] for tag in raw_tags if str(tag).strip()) if isinstance(raw_tags, list) else ()
    memory.upsert_meme_asset(
        sha256=digest,
        source_group_id=GROUP_ID,
        source_user_id=OWNER_ID,
        source_message_id=source_message_id,
        file_path=str(target),
        mime_type=content_type or "image/png",
        byte_size=len(payload),
        description=description,
        tags=tags,
        enabled=is_meme and bool(description),
    )
    return is_meme and bool(description), description or "not_a_meme"


def _today_group_images(db_path: Path) -> list[tuple[str, str, str]]:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "select source_message_id, created_at, message_segments_json from messages where group_id=? and user_id=? order by id",
        (GROUP_ID, OWNER_ID),
    ).fetchall()
    items: list[tuple[str, str, str]] = []
    for row in rows:
        observed = datetime.fromtimestamp(float(row["created_at"]), ZoneInfo("Asia/Shanghai")).date()
        if observed != today:
            continue
        try:
            segments = json.loads(str(row["message_segments_json"] or "[]"))
        except json.JSONDecodeError:
            continue
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "image":
                continue
            data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
            url = str(data.get("url", "") or "").strip()
            if url:
                items.append((str(row["source_message_id"] or row["created_at"]), url, str(data.get("file", "") or "")))
    return items


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--today-group-images", action="store_true")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--labels-json")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    memory = MemoryStore(config.data_path)
    raw_library = config.raw.get("meme_library", {})
    asset_dir = config.data_path.parent / str(raw_library.get("asset_dir", "meme_library"))
    asset_dir.mkdir(parents=True, exist_ok=True)
    vision = SiliconFlowOcrClient.from_config(config.raw.get("image_ocr", {}))
    imported = rejected = failed = 0
    manual_labels: dict[str, object] = {}
    if args.labels_json:
        try:
            loaded = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
            manual_labels = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            print("invalid labels json")
    try:
        if args.today_group_images:
            for source_message_id, url, source_name in _today_group_images(config.data_path):
                try:
                    payload, content_type = await _download(url)
                    ok, detail = await _import_one(
                        memory=memory, vision=vision, asset_dir=asset_dir, payload=payload,
                        content_type=content_type, source_message_id=source_message_id, source_name=source_name,
                    )
                except Exception as exc:
                    failed += 1
                    print(f"failed group:{source_message_id}: {type(exc).__name__}")
                    continue
                imported += int(ok)
                rejected += int(not ok)
                print(f"{'imported' if ok else 'rejected'} group:{source_message_id}: {detail}")
        for raw_path in args.files:
            path = Path(raw_path)
            try:
                payload = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "image/png"
                ok, detail = await _import_one(
                    memory=memory, vision=vision, asset_dir=asset_dir, payload=payload,
                    content_type=content_type, source_message_id=f"manual:{path.name}", source_name=path.name,
                    manual_annotation=manual_labels.get(path.name)
                    if isinstance(manual_labels.get(path.name), dict)
                    else None,
                )
            except Exception as exc:
                failed += 1
                print(f"failed file:{path.name}: {type(exc).__name__}")
                continue
            imported += int(ok)
            rejected += int(not ok)
            print(f"{'imported' if ok else 'rejected'} file:{path.name}: {detail}")
    finally:
        await vision.aclose()
    print(json.dumps({"imported": imported, "rejected": rejected, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
