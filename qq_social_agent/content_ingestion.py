from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from . import onebot_gateway
from .message_segments import file_metadata, segment_type_and_data
from .tools.file_content_reader import FileContentReader, FileContentReaderConfig
from .tools.voice_transcript import (
    SiliconFlowTranscriptProvider,
    VoiceTranscriptConfig,
    VoiceTranscriptContext,
    VoiceTranscriptRequest,
    VoiceTranscriptService,
)


_MIME_BY_EXTENSION = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass(frozen=True)
class ContentIngestionResult:
    file_context: str = ""
    voice_context: str = ""
    file_count: int = 0
    voice_count: int = 0
    file_status: str = ""
    voice_status: str = ""
    file_name: str = ""
    file_source_id: str = ""
    file_text: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(part for part in (self.file_context, self.voice_context) if part).strip()


class ContentIngestionService:
    def __init__(
        self,
        *,
        file_reader: FileContentReader | None = None,
        voice_service: VoiceTranscriptService | None = None,
        onebot_timeout_seconds: float = 10.0,
    ) -> None:
        self.file_reader = file_reader or FileContentReader()
        self.voice_service = voice_service or VoiceTranscriptService()
        self.onebot_timeout_seconds = max(1.0, min(30.0, float(onebot_timeout_seconds)))

    @classmethod
    def from_config(cls, raw: object) -> "ContentIngestionService":
        config = raw if isinstance(raw, dict) else {}
        file_config = FileContentReaderConfig.from_config(config.get("file_content"))
        voice_config = VoiceTranscriptConfig.from_config(config.get("voice_transcript"))
        provider = SiliconFlowTranscriptProvider.from_config(voice_config)
        return cls(
            file_reader=FileContentReader(file_config),
            voice_service=VoiceTranscriptService(voice_config, provider=provider),
            onebot_timeout_seconds=_bounded_float(config.get("onebot_timeout_seconds"), 10.0, 1.0, 30.0),
        )

    async def context_for_event(
        self,
        bot: onebot_gateway.OneBotGateway,
        event: Any,
        *,
        allow_file_content: bool,
        voice_context: VoiceTranscriptContext,
    ) -> ContentIngestionResult:
        segments = list(getattr(event, "message", []) or [])
        file_segments = []
        voice_segments = []
        for segment in segments:
            segment_type, data = segment_type_and_data(segment)
            if segment_type == "file":
                file_segments.append(data)
            elif segment_type == "record":
                voice_segments.append(data)

        file_text = ""
        file_status = ""
        file_name = ""
        file_source_id = ""
        file_body = ""
        if file_segments and allow_file_content:
            file_text, file_status, file_name, file_source_id, file_body = await self._read_file_segment(
                bot, file_segments[0]
            )
        elif file_segments:
            file_status = "context_not_allowed"

        voice_text = ""
        voice_status = ""
        if voice_segments:
            voice_text, voice_status = await self._transcribe_voice_segment(
                bot,
                voice_segments[0],
                context=voice_context,
            )

        return ContentIngestionResult(
            file_context=file_text,
            voice_context=voice_text,
            file_count=len(file_segments),
            voice_count=len(voice_segments),
            file_status=file_status,
            voice_status=voice_status,
            file_name=file_name,
            file_source_id=file_source_id,
            file_text=file_body,
        )

    async def _read_file_segment(
        self,
        bot: onebot_gateway.OneBotGateway,
        data: dict[str, Any],
    ) -> tuple[str, str, str, str, str]:
        metadata = file_metadata(data)
        filename = str(metadata.get("name") or data.get("name") or "").strip()
        file_id = str(metadata.get("file_id") or data.get("file_id") or "").strip()
        file_ref = str(data.get("file") or "").strip()
        if not filename or not (file_id or file_ref):
            return "", "missing_file_reference", filename, file_id or file_ref, ""
        try:
            fetched = await onebot_gateway.get_file(
                bot,
                file_id or None,
                file=file_ref or None,
                timeout_seconds=self.onebot_timeout_seconds,
            )
        except Exception as exc:
            return "", f"onebot_error:{type(exc).__name__}", filename, file_id or file_ref, ""
        payload = _binary_payload(fetched, max_bytes=self.file_reader.config.max_file_bytes)
        if payload is None:
            return "", "file_bytes_unavailable", filename, file_id or file_ref, ""
        content_type = _MIME_BY_EXTENSION.get(PurePath(filename).suffix.casefold(), "")
        result = self.file_reader.read(filename, payload, content_type=content_type)
        return result.to_context(), result.status, result.filename, file_id or file_ref, result.text

    async def _transcribe_voice_segment(
        self,
        bot: onebot_gateway.OneBotGateway,
        data: dict[str, Any],
        *,
        context: VoiceTranscriptContext,
    ) -> tuple[str, str]:
        policy = self.voice_service.config
        if not policy.enabled:
            return "", "disabled"
        file_id = str(data.get("file_id") or "").strip()
        file_ref = str(data.get("file") or "").strip()
        if not file_id and not file_ref:
            return "", "missing_voice_reference"
        try:
            converted = await onebot_gateway.get_record(
                bot,
                file_id=file_id or None,
                file=file_ref or None,
                out_format="mp3",
                timeout_seconds=self.onebot_timeout_seconds,
            )
        except Exception as exc:
            return "", f"onebot_error:{type(exc).__name__}"
        payload = _binary_payload(converted, max_bytes=policy.max_audio_bytes)
        if payload is None:
            converted_ref = str(converted.get("file") or converted.get("path") or "").strip()
            converted_id = str(converted.get("file_id") or "").strip()
            if converted_ref or converted_id:
                try:
                    fetched = await onebot_gateway.get_file(
                        bot,
                        converted_id or None,
                        file=converted_ref or None,
                        timeout_seconds=self.onebot_timeout_seconds,
                    )
                except Exception as exc:
                    return "", f"onebot_file_error:{type(exc).__name__}"
                payload = _binary_payload(fetched, max_bytes=policy.max_audio_bytes)
        if payload is None:
            return "", "voice_bytes_unavailable"
        result = await self.voice_service.transcribe(
            VoiceTranscriptRequest(
                audio=payload,
                filename="qq-voice.mp3",
                content_type="audio/mpeg",
                language="zh",
                context=context,
            )
        )
        return result.to_context(), result.status

    def status_snapshot(self) -> dict[str, object]:
        return {
            "file_content": self.file_reader.status_snapshot(),
            "voice_transcript": self.voice_service.status_snapshot(),
            "onebot_timeout_seconds": self.onebot_timeout_seconds,
        }

    async def aclose(self) -> None:
        await self.voice_service.aclose()


def _binary_payload(value: object, *, max_bytes: int) -> bytes | None:
    payload = onebot_gateway.unwrap_data(value)
    data = payload if isinstance(payload, dict) else {}
    candidates = [("base64", data.get("base64")), ("data", data.get("data")), ("file", data.get("file"))]
    for key, candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        encoded = ""
        if raw.startswith("base64://"):
            encoded = raw[len("base64://") :]
        elif raw.startswith("data:") and ";base64," in raw:
            encoded = raw.split(";base64,", 1)[1]
        elif key == "base64":
            encoded = raw
        if not encoded or len(encoded) > max_bytes * 2:
            continue
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            continue
        if 0 < len(decoded) <= max_bytes:
            return decoded
    return None


def explicit_file_read_requested(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).casefold()
    return any(
        token in compact
        for token in ("读文件", "看文件", "看看文件", "总结文件", "文件内容", "读一下", "看一下附件", "总结附件")
    )


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))
