from __future__ import annotations

import json
import re
from typing import Any


CONTEXT_MEDIA_SEGMENT_TYPES = frozenset(
    {
        "image",
        "mface",
        "face",
        "record",
        "video",
        "forward",
        "json",
        "xml",
        "file",
        "music",
        "share",
        "lightapp",
        "location",
        "contact",
    }
)


def compact_spaces(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def message_segments_from_payload(payload: Any) -> list[Any]:
    if hasattr(payload, "message"):
        payload = getattr(payload, "message")
    if isinstance(payload, dict):
        if "type" in payload:
            return [payload]
        for key in ("message", "content"):
            if key not in payload or payload.get(key) is None:
                continue
            segments = message_segments_from_payload(payload.get(key))
            if segments:
                return segments
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        return [{"type": "text", "data": {"text": payload}}] if payload.strip() else []
    return []


def message_text_from_payload(payload: Any, *, language: str = "en") -> str:
    if hasattr(payload, "extract_plain_text"):
        try:
            text = compact_spaces(payload.extract_plain_text())
            if text:
                return text
        except Exception:
            pass
    if isinstance(payload, str):
        return compact_spaces(payload)
    return segments_to_text(message_segments_from_payload(payload), language=language)


def segments_to_text(segments: list[Any], *, language: str = "en") -> str:
    parts: list[str] = []
    for segment in segments:
        segment_type, data = segment_type_and_data(segment)
        if segment_type == "text":
            text = compact_spaces(data.get("text", ""))
            if text:
                parts.append(text)
            continue
        placeholder = segment_placeholder(segment_type, data, language=language)
        if placeholder:
            parts.append(placeholder)
    return compact_spaces(" ".join(parts))


def segment_type_and_data(segment: Any) -> tuple[str, dict[str, Any]]:
    segment_type = str(getattr(segment, "type", "") or "")
    raw_data = getattr(segment, "data", None)
    if isinstance(segment, dict):
        segment_type = str(segment.get("type", segment_type) or "")
        raw_data = segment.get("data", raw_data)
    data = raw_data if isinstance(raw_data, dict) else {}
    return segment_type.strip().casefold(), data


def is_marketface_segment(segment_type: str, data: dict[str, Any]) -> bool:
    normalized_type = str(segment_type or "").strip().casefold()
    if normalized_type == "mface":
        return True
    if normalized_type != "image":
        return False
    file_value = compact_spaces(data.get("file", "")).casefold()
    sub_type = compact_spaces(data.get("sub_type", "")).casefold()
    summary = compact_spaces(data.get("summary", "")).casefold().strip("[]【】")
    return bool(
        file_value == "marketface"
        or sub_type in {"marketface", "mface", "emoji", "sticker"}
        or "动画表情" in summary
        or "商城表情" in summary
        or data.get("emoji_id")
        or data.get("emoji_package_id")
    )


def format_file_size(value: object) -> str:
    raw = compact_spaces(value)
    if not raw or raw.casefold() in {"empty", "none", "null", "unknown"}:
        return ""
    try:
        size = max(0, int(float(raw)))
    except (TypeError, ValueError):
        return _short_value(raw, 24)
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(size)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "B":
        return f"{int(amount)} {unit}"
    rendered = f"{amount:.1f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}"


def file_metadata(data: dict[str, Any]) -> dict[str, str]:
    name = _first_text(data, "name", "file_name")
    raw_file = _clean_sentinel(data.get("file", ""))
    if not name and raw_file and raw_file.casefold() != "marketface":
        name = raw_file
    return {
        "name": _short_value(name, 100),
        "size": format_file_size(data.get("file_size", "")),
        "file_id": _short_value(_first_text(data, "file_id", "id"), 160),
    }


def segment_placeholder(
    segment_type: str,
    data: dict[str, Any],
    *,
    language: str = "en",
) -> str:
    segment_type = str(segment_type or "").strip().casefold()
    zh = str(language or "").casefold().startswith("zh")
    if segment_type == "at":
        qq = _short_value(data.get("qq", ""), 32)
        return f"[@{qq}]" if qq else ("[@群友]" if zh else "[@group_member]")
    if segment_type == "reply":
        message_id = _short_value(_first_text(data, "id", "message_id"), 64)
        if zh:
            return f"[回复消息:{message_id}]" if message_id else "[回复消息]"
        return f"[reply:{message_id}]" if message_id else "[reply]"
    if segment_type in {"image", "mface"}:
        marketface = is_marketface_segment(segment_type, data)
        summary = _short_value(_first_text(data, "summary", "name"), 100)
        label = "表情包" if zh and marketface else "图片" if zh else "mface" if marketface else "image"
        return _placeholder(label, summary)
    if segment_type == "face":
        face_id = _short_value(data.get("id", ""), 32)
        return _placeholder("表情" if zh else "face", face_id)
    if segment_type in {"record", "video", "file"}:
        metadata = file_metadata(data)
        details = [value for value in (metadata["name"], metadata["size"]) if value]
        if segment_type == "record":
            label = "语音" if zh else "voice"
        elif segment_type == "video":
            label = "视频" if zh else "video"
        else:
            label = "文件" if zh else "file"
        return _placeholder(label, "，".join(details))
    if segment_type == "forward":
        return "[转发消息]" if zh else "[forward]"
    if segment_type in {"json", "xml"}:
        payload = compact_spaces(data.get("data", ""))
        if _looks_like_forward(payload):
            return "[转发消息]" if zh else "[forward]"
        if segment_type == "json":
            return _json_card_placeholder(payload, zh=zh)
        return "[XML卡片]" if zh else "[xml]"
    if segment_type == "music":
        title = _first_text(data, "title", "name")
        singer = _first_text(data, "singer", "artist")
        detail = " - ".join(value for value in (_short_value(title, 80), _short_value(singer, 60)) if value)
        if not detail:
            detail = _short_value(_first_text(data, "id", "type"), 48)
        return _placeholder("音乐分享" if zh else "music", detail)
    if segment_type == "share":
        detail = _short_value(_first_text(data, "title", "content", "summary"), 100)
        return _placeholder("链接分享" if zh else "share", detail)
    if segment_type == "lightapp":
        payload = compact_spaces(data.get("data", "") or data.get("content", ""))
        detail = _json_card_summary(payload)
        return _placeholder("小程序" if zh else "miniapp", detail)
    if segment_type == "location":
        detail = _short_value(_first_text(data, "title", "name", "address", "content"), 100)
        return _placeholder("位置" if zh else "location", detail)
    if segment_type == "contact":
        contact_type = _short_value(data.get("type", ""), 20)
        contact_id = _short_value(data.get("id", ""), 32)
        detail = ":".join(value for value in (contact_type, contact_id) if value)
        return _placeholder("联系人" if zh else "contact", detail)
    if segment_type == "rps":
        return _placeholder("猜拳" if zh else "rps", _short_value(data.get("result", ""), 20))
    if segment_type == "dice":
        return _placeholder("骰子" if zh else "dice", _short_value(data.get("result", ""), 20))
    if segment_type == "markdown":
        detail = _short_value(_first_text(data, "content", "markdown"), 120)
        return _placeholder("Markdown", detail)
    if segment_type == "node":
        return "[转发节点]" if zh else "[forward_node]"
    return ""


def _json_card_placeholder(payload: str, *, zh: bool) -> str:
    if not payload:
        return "[JSON卡片]" if zh else "[json]"
    kind = _json_card_kind(payload)
    summary = _json_card_summary(payload)
    labels = {
        "music": ("音乐分享", "music"),
        "share": ("链接分享", "share"),
        "miniapp": ("小程序", "miniapp"),
        "location": ("位置", "location"),
        "contact": ("联系人", "contact"),
        "json": ("JSON卡片", "json"),
    }
    label = labels[kind][0 if zh else 1]
    return _placeholder(label, summary)


def _json_card_kind(payload: str) -> str:
    folded = payload.casefold()
    hints = (
        ("music", ("music", "音乐", "song")),
        ("miniapp", ("miniapp", "mini_app", "lightapp", "小程序")),
        ("location", ("location", "位置", "latitude", "longitude")),
        ("contact", ("contact", "联系人", "contact_user", "contact_group")),
        ("share", ("share", "链接分享", '"news"', '"url"')),
    )
    for kind, markers in hints:
        if any(marker in folded for marker in markers):
            return kind
    return "json"


def _json_card_summary(payload: str) -> str:
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    values: list[str] = []
    _collect_named_text(
        parsed,
        keys={"prompt", "title", "desc", "description", "summary", "tag", "name", "app_name"},
        output=values,
    )
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _short_value(value, 90)
        key = compact.casefold()
        if not compact or key in seen:
            continue
        seen.add(key)
        unique.append(compact)
        if len(unique) >= 2:
            break
    return " / ".join(unique)


def _collect_named_text(value: Any, *, keys: set[str], output: list[str]) -> None:
    if len(output) >= 8:
        return
    if isinstance(value, list):
        for item in value[:12]:
            _collect_named_text(item, keys=keys, output=output)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if str(key).casefold() in keys and isinstance(item, (str, int, float)):
            text = compact_spaces(item)
            if text:
                output.append(text)
        elif isinstance(item, (dict, list)):
            _collect_named_text(item, keys=keys, output=output)


def _looks_like_forward(payload: str) -> bool:
    folded = payload.casefold()
    return "forward" in folded or "聊天记录" in payload


def _first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _clean_sentinel(data.get(key, ""))
        if value:
            return value
    return ""


def _clean_sentinel(value: object) -> str:
    text = compact_spaces(value)
    return "" if text.casefold() in {"empty", "none", "null", "unknown"} else text


def _short_value(value: object, limit: int) -> str:
    text = compact_spaces(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _placeholder(label: str, detail: str) -> str:
    return f"[{label}:{detail}]" if detail else f"[{label}]"
