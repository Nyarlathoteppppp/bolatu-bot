from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from . import onebot_gateway
from .message_segments import (
    compact_spaces,
    file_metadata,
    is_marketface_segment,
    message_segments_from_payload,
    message_text_from_payload,
    segment_placeholder,
    segment_type_and_data,
    segments_to_text,
)
from .siliconflow_ocr import SiliconFlowOcrClient


@dataclass(frozen=True)
class ImageOcrResult:
    image_key: str
    text: str
    from_cache: bool = False


@dataclass(frozen=True)
class ImageOcrContext:
    text: str
    image_count: int
    ocr_count: int
    skipped_reason: str = ""


@dataclass(frozen=True)
class _OcrCacheEntry:
    text: str
    created_at: float


class ImageOcrService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        max_images_per_message: int = 2,
        max_text_chars_per_image: int = 220,
        max_calls_per_minute: int = 18,
        cache_ttl_seconds: int = 24 * 60 * 60,
        api_timeout_seconds: float = 8.0,
        napcat_ocr_enabled: bool = True,
        fallback_ocr: SiliconFlowOcrClient | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_images_per_message = max(0, int(max_images_per_message))
        self.max_text_chars_per_image = max(40, int(max_text_chars_per_image))
        self.max_calls_per_minute = max(0, int(max_calls_per_minute))
        self.cache_ttl_seconds = max(60, int(cache_ttl_seconds))
        self.api_timeout_seconds = max(0.05, float(api_timeout_seconds))
        self.napcat_ocr_enabled = bool(napcat_ocr_enabled)
        self.fallback_ocr = fallback_ocr
        self._cache: dict[str, _OcrCacheEntry] = {}
        self._call_times: deque[float] = deque()

    @classmethod
    def from_config(cls, raw: object) -> "ImageOcrService":
        cfg = raw if isinstance(raw, dict) else {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            max_images_per_message=int(cfg.get("max_images_per_message", 2)),
            max_text_chars_per_image=int(cfg.get("max_text_chars_per_image", 220)),
            max_calls_per_minute=int(cfg.get("max_calls_per_minute", 18)),
            cache_ttl_seconds=int(cfg.get("cache_ttl_seconds", 24 * 60 * 60)),
            api_timeout_seconds=float(cfg.get("api_timeout_seconds", 8.0)),
            napcat_ocr_enabled=bool(cfg.get("napcat_ocr_enabled", True)),
            fallback_ocr=SiliconFlowOcrClient.from_config(cfg),
        )

    async def context_for_event(self, bot: onebot_gateway.OneBotGateway, event: Any) -> ImageOcrContext:
        if not self.enabled:
            return ImageOcrContext("", 0, 0, "disabled")
        image_segments = ocr_image_segments_from_event(event)
        if not image_segments:
            return ImageOcrContext("", 0, 0)
        if self.max_images_per_message <= 0:
            return ImageOcrContext("", len(image_segments), 0, "message_limit")
        results: list[ImageOcrResult] = []
        for data in image_segments[: self.max_images_per_message]:
            result = await self.ocr_image_segment(bot, data)
            if result is not None and result.text:
                results.append(result)
        if not results:
            return ImageOcrContext("", len(image_segments), 0, "empty_ocr")
        lines = [
            f"第{index}张图：{_compact_ocr_text(result.text, self.max_text_chars_per_image)}"
            for index, result in enumerate(results, start=1)
        ]
        return ImageOcrContext(
            text="；".join(lines),
            image_count=len(image_segments),
            ocr_count=len(results),
        )

    async def ocr_image_segment(
        self,
        bot: onebot_gateway.OneBotGateway,
        data: dict[str, Any],
    ) -> ImageOcrResult | None:
        image_key = image_cache_key(data)
        if not image_key:
            return None
        cached = self._cached_text(image_key)
        if cached is not None:
            return ImageOcrResult(image_key=image_key, text=cached, from_cache=True)
        if not self._rate_limit_available(time.time()):
            return None
        targets = list(image_ocr_targets(data))
        file_id = str(data.get("file", "") or "").strip()
        if file_id:
            try:
                image_info = await onebot_gateway.get_image(
                    bot,
                    file_id,
                    timeout_seconds=self.api_timeout_seconds,
                )
            except Exception:
                image_info = {}
            targets.extend(image_ocr_targets(image_info))
        seen_targets: set[str] = set()
        for target in targets:
            if target in seen_targets:
                continue
            seen_targets.add(target)
            text = await self._ocr_target(bot, target)
            if text:
                self._cache[image_key] = _OcrCacheEntry(text=text, created_at=time.time())
                return ImageOcrResult(image_key=image_key, text=text, from_cache=False)
        self._cache[image_key] = _OcrCacheEntry(text="", created_at=time.time())
        return None
    async def _ocr_target(self, bot: onebot_gateway.OneBotGateway, target: str) -> str:
        if self.napcat_ocr_enabled:
            for enhanced in (False, True):
                if not self._rate_limit_available(time.time()):
                    return ""
                self._remember_call(time.time())
                try:
                    payload = await onebot_gateway.ocr_image(
                        bot,
                        target,
                        enhanced=enhanced,
                        timeout_seconds=self.api_timeout_seconds,
                    )
                except Exception:
                    continue
                text = parse_ocr_text(payload)
                if text:
                    return text
        if self.fallback_ocr is not None:
            if not self._rate_limit_available(time.time()):
                return ""
            self._remember_call(time.time())
            try:
                text = await self.fallback_ocr.recognize(target)
            except Exception:
                return ""
            if text:
                return _compact_ocr_text(text, 500)
        return ""

    def _cached_text(self, image_key: str) -> str | None:
        entry = self._cache.get(image_key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self.cache_ttl_seconds:
            self._cache.pop(image_key, None)
            return None
        return entry.text

    def _rate_limit_available(self, now: float) -> bool:
        if self.max_calls_per_minute <= 0:
            return False
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()
        return len(self._call_times) < self.max_calls_per_minute

    def _remember_call(self, now: float) -> None:
        self._call_times.append(now)

    async def aclose(self) -> None:
        closer = getattr(self.fallback_ocr, "aclose", None)
        if closer is not None:
            await closer()


async def file_metadata_context_for_event(
    bot: onebot_gateway.OneBotGateway,
    event: Any,
    *,
    max_files: int = 2,
    timeout_seconds: float = 6.0,
    language: str = "zh",
) -> str:
    """Complete missing file name/size metadata without downloading file content."""

    supplements: list[str] = []
    for segment in list(getattr(event, "message", []) or []):
        segment_type, data = segment_type_and_data(segment)
        if segment_type != "file":
            continue
        metadata = file_metadata(data)
        file_id = metadata["file_id"]
        if not file_id or (metadata["name"] and metadata["size"]):
            continue
        try:
            fetched = await onebot_gateway.get_file(
                bot,
                file_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            continue
        merged = dict(fetched)
        for key, value in data.items():
            if value not in (None, ""):
                merged[key] = value
        placeholder = segment_placeholder("file", merged, language=language)
        if placeholder and placeholder not in supplements:
            supplements.append(placeholder)
        if len(supplements) >= max(0, int(max_files)):
            break
    return compact_spaces(" ".join(supplements))


def image_segments_from_event(event: Any) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for segment in getattr(event, "message", []) or []:
        segment_type, data = segment_type_and_data(segment)
        if segment_type in {"image", "mface"}:
            images.append(data)
    return images


def ocr_image_segments_from_event(event: Any) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for segment in getattr(event, "message", []) or []:
        segment_type, data = segment_type_and_data(segment)
        if segment_type not in {"image", "mface"}:
            continue
        if is_marketface_segment(segment_type, data):
            continue
        images.append(data)
    return images


def image_cache_key(data: dict[str, Any]) -> str:
    for key in ("file_unique", "file_id", "url", "path", "md5", "file", "summary"):
        value = str(data.get(key, "") or "").strip()
        if key == "file" and value.casefold() == "marketface":
            continue
        if value:
            return f"{key}:{value}"
    return ""


def image_ocr_targets(data: dict[str, Any]) -> tuple[str, ...]:
    targets: list[str] = []
    for key in ("path", "url", "file"):
        value = str(data.get(key, "") or "").strip()
        if key == "file" and value.casefold() == "marketface":
            continue
        if value:
            targets.append(value)
    return tuple(targets)


def parse_ocr_text(payload: Any) -> str:
    data = onebot_gateway.unwrap_data(payload)
    texts = _collect_ocr_texts(data)
    return _compact_ocr_text(" ".join(texts), 500)


def _collect_ocr_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = compact_spaces(value)
        return [text] if text else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_collect_ocr_texts(item))
        return _dedupe_texts(texts)
    if not isinstance(value, dict):
        return []

    direct_texts: list[str] = []
    for key in ("text", "words", "content", "label", "detected_text"):
        raw = value.get(key)
        if isinstance(raw, str):
            text = compact_spaces(raw)
            if text:
                direct_texts.append(text)
    nested_texts: list[str] = []
    for key in ("texts", "results", "result", "ocr_results", "words_result", "items", "data"):
        if key in value:
            nested_texts.extend(_collect_ocr_texts(value.get(key)))
    return _dedupe_texts([*direct_texts, *nested_texts])


def _compact_ocr_text(text: str, limit: int) -> str:
    compact = compact_spaces(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _dedupe_texts(texts: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in texts:
        text = compact_spaces(item)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def sender_nickname(sender: Any, *, fallback_user_id: int | str | None = None) -> str:
    if isinstance(sender, dict):
        name = str(sender.get("card") or sender.get("nickname") or sender.get("name") or "").strip()
    else:
        name = str(
            getattr(sender, "card", "") or getattr(sender, "nickname", "") or getattr(sender, "name", "") or ""
        ).strip()
    if name:
        return name
    return str(fallback_user_id or "").strip()


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
