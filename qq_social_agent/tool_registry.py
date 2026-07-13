from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

from .pipeline_types import ToolKind, ToolRequest, ToolResult


ToolHandler = Callable[[ToolRequest], ToolResult | Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    kind: ToolKind
    description: str
    handler: ToolHandler
    enabled: bool = True


class ToolRegistry:
    """Small execution boundary shared by search, market and future tools."""

    def __init__(self) -> None:
        self._specs: dict[ToolKind, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.kind] = spec

    def available(self, allowed: Iterable[ToolKind] | None = None) -> tuple[ToolSpec, ...]:
        allowed_set = set(allowed) if allowed is not None else None
        return tuple(
            spec
            for kind, spec in self._specs.items()
            if spec.enabled and (allowed_set is None or kind in allowed_set)
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        started = time.monotonic()
        spec = self._specs.get(request.kind)
        if spec is None or not spec.enabled:
            return ToolResult(request.kind, "unavailable", error="tool_not_registered")
        try:
            result = spec.handler(request)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, ToolResult):
                raise TypeError("tool handler must return ToolResult")
            if result.elapsed_ms:
                return result
            return ToolResult(
                kind=result.kind,
                status=result.status,
                context=result.context,
                evidence=result.evidence,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                error=result.error,
                metadata=result.metadata,
            )
        except Exception as exc:
            return ToolResult(
                request.kind,
                "error",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}"[:240],
            )

