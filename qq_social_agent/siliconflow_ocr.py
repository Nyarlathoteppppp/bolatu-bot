from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiliconFlowOcrConfig:
    enabled: bool = False
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key_env: str = "SILICONFLOW_API_KEY"
    model: str = "deepseek-ai/DeepSeek-OCR"
    timeout_seconds: float = 20.0
    max_tokens: int = 512
    detail: str = "high"
    max_file_bytes: int = 5 * 1024 * 1024
    prompt: str = "请进行 OCR，逐字转写图片中能看清的文字。只输出识别到的文字，不要描述图片，不要解释。"


class SiliconFlowOcrClient:
    def __init__(self, config: SiliconFlowOcrConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._warned_missing_key = False

    @classmethod
    def from_config(cls, raw: object) -> "SiliconFlowOcrClient":
        cfg = raw if isinstance(raw, dict) else {}
        return cls(
            SiliconFlowOcrConfig(
                enabled=bool(cfg.get("siliconflow_fallback_enabled", False)),
                base_url=str(cfg.get("siliconflow_base_url", "https://api.siliconflow.cn/v1")),
                api_key_env=str(cfg.get("siliconflow_api_key_env", "SILICONFLOW_API_KEY")),
                model=str(cfg.get("siliconflow_model", "deepseek-ai/DeepSeek-OCR")),
                timeout_seconds=float(cfg.get("siliconflow_timeout_seconds", 20.0)),
                max_tokens=int(cfg.get("siliconflow_max_tokens", 512)),
                detail=str(cfg.get("siliconflow_detail", "high")),
                max_file_bytes=int(cfg.get("siliconflow_max_file_bytes", 5 * 1024 * 1024)),
                prompt=str(
                    cfg.get(
                        "siliconflow_prompt",
                        "请进行 OCR，逐字转写图片中能看清的文字。只输出识别到的文字，不要描述图片，不要解释。",
                    )
                ),
            )
        )

    async def recognize(self, target: str) -> str:
        if not self.config.enabled:
            return ""
        client = self._get_client()
        if client is None:
            return ""
        image_url = self._image_url_payload(target)
        if not image_url:
            return ""
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": self.config.detail,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.config.prompt,
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=self.config.max_tokens,
        )
        return _clean_ocr_text(_extract_text(response))

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            if not self._warned_missing_key:
                logger.warning(
                    "qq_social_agent siliconflow ocr key missing: "
                    f"env={self.config.api_key_env}"
                )
                self._warned_missing_key = True
            return None
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        return self._client

    def _image_url_payload(self, target: str) -> str:
        text = str(target or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://", "data:image/")):
            return text
        path = Path(text)
        if not path.is_file():
            return ""
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        if size <= 0 or size > self.config.max_file_bytes:
            return ""
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return ""
        return f"data:{mime_type};base64,{encoded}"


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return " ".join(parts).strip()
    return ""


def _clean_ocr_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    match = re.fullmatch(
        r"(?:图片中(?:的)?文字|识别(?:到)?(?:的)?文字|文字|OCR\s*结果|识别结果)"
        r"\s*(?:为|是|如下)?\s*[:：]?\s*[“\"']?(.*?)[”\"']?\s*[。.]?",
        compact,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).strip():
        compact = match.group(1).strip()
    return compact.strip("“”\"'")
