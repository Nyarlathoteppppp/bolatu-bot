from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool = True
    api_key_env: str = "SILICONFLOW_API_KEY"
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-m3"
    dimensions: int = 0
    timeout_seconds: float = 8.0
    batch_size: int = 24

    @classmethod
    def from_mapping(cls, raw: object) -> "EmbeddingConfig":
        config = raw if isinstance(raw, dict) else {}
        return cls(
            enabled=bool(config.get("enabled", True)),
            api_key_env=str(config.get("api_key_env", "SILICONFLOW_API_KEY")),
            base_url=str(config.get("base_url", "https://api.siliconflow.cn/v1")),
            model=str(config.get("model", "BAAI/bge-m3")),
            dimensions=max(0, int(config.get("dimensions", 0) or 0)),
            timeout_seconds=max(1.0, float(config.get("timeout_seconds", 8.0))),
            batch_size=max(1, min(64, int(config.get("batch_size", 24)))),
        )


class SiliconFlowEmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env, "").strip()
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            follow_redirects=False,
        )
        self.calls = 0
        self.failures = 0
        self.last_error = ""
        self.last_success_at = 0.0

    @property
    def available(self) -> bool:
        return bool(self.config.enabled and self.api_key and self.config.model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        clean = [str(text).strip() for text in texts]
        if not clean or any(not text for text in clean):
            raise ValueError("embedding input must contain non-empty text")
        if not self.available:
            raise RuntimeError(f"embedding provider unavailable: missing {self.config.api_key_env}")
        payload: dict[str, object] = {
            "model": self.config.model,
            "input": clean,
            "encoding_format": "float",
        }
        if self.config.dimensions > 0:
            payload["dimensions"] = self.config.dimensions
        self.calls += 1
        try:
            response = await self._client.post(
                "/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            body = response.json()
            rows = body.get("data") if isinstance(body, dict) else None
            if not isinstance(rows, list):
                raise RuntimeError("embedding response has no data list")
            ordered = sorted(
                (row for row in rows if isinstance(row, dict)),
                key=lambda row: int(row.get("index", 0)),
            )
            vectors = [row.get("embedding") for row in ordered]
            if len(vectors) != len(clean) or any(not isinstance(vector, list) for vector in vectors):
                raise RuntimeError("embedding response count does not match input")
            result = [[float(value) for value in vector] for vector in vectors]
            if not result or not result[0] or any(len(vector) != len(result[0]) for vector in result):
                raise RuntimeError("embedding response contains invalid vector dimensions")
            self.last_success_at = time.time()
            self.last_error = ""
            return result
        except Exception as exc:
            self.failures += 1
            self.last_error = str(exc)[:240]
            raise

    async def aclose(self) -> None:
        await self._client.aclose()

    def status_snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "available": self.available,
            "provider": "siliconflow",
            "model": self.config.model,
            "configured_dimensions": self.config.dimensions or None,
            "api_key_env": self.config.api_key_env,
            "api_key_configured": bool(self.api_key),
            "calls": self.calls,
            "failures": self.failures,
            "last_success_at": self.last_success_at or None,
            "last_error": self.last_error,
        }
