from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .media_context import coerce_int


@dataclass(frozen=True)
class NoticeSnapshot:
    notice_type: str
    sub_type: str
    group_id: int | None
    user_id: int | None
    operator_id: int | None
    target_id: int | None
    message_id: str
    duration_seconds: int
    file_name: str
    file_size: int
    old_card: str
    new_card: str

    def metric_metadata(self) -> dict[str, object]:
        values: dict[str, object] = {
            "notice_type": self.notice_type,
            "sub_type": self.sub_type,
            "operator_id": self.operator_id,
            "target_id": self.target_id,
            "message_id": self.message_id,
            "duration_seconds": self.duration_seconds,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "old_card": self.old_card,
            "new_card": self.new_card,
        }
        return {key: value for key, value in values.items() if value not in (None, "", 0)}


def notice_snapshot(event: Any) -> NoticeSnapshot:
    message_id = str(getattr(event, "message_id", "") or "").strip()
    file_payload = getattr(event, "file", None)
    if hasattr(file_payload, "model_dump"):
        file_payload = file_payload.model_dump()
    elif hasattr(file_payload, "dict"):
        file_payload = file_payload.dict()
    file_data = file_payload if isinstance(file_payload, dict) else {}
    return NoticeSnapshot(
        notice_type=str(getattr(event, "notice_type", "") or ""),
        sub_type=str(getattr(event, "sub_type", "") or ""),
        group_id=_optional_int(getattr(event, "group_id", None)),
        user_id=_optional_int(getattr(event, "user_id", None)),
        operator_id=_optional_int(getattr(event, "operator_id", None)),
        target_id=_optional_int(getattr(event, "target_id", None)),
        message_id=message_id,
        duration_seconds=coerce_int(getattr(event, "duration", 0), 0),
        file_name=str(file_data.get("name") or file_data.get("file_name") or "").strip()[:160],
        file_size=max(0, coerce_int(file_data.get("size") or file_data.get("file_size"), 0)),
        old_card=str(getattr(event, "card_old", "") or getattr(event, "old_card", "") or "").strip()[:120],
        new_card=str(getattr(event, "card_new", "") or getattr(event, "new_card", "") or "").strip()[:120],
    )


def _optional_int(value: Any) -> int | None:
    number = coerce_int(value, 0)
    return number or None
