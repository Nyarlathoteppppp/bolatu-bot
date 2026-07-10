from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import yaml

from .config import PROJECT_ROOT


DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "zhangfengxue.yaml"


class PromptRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_PROMPT_FILE
        with self.path.open("r", encoding="utf-8") as f:
            self.raw: dict[str, Any] = yaml.safe_load(f) or {}
        self.flows: dict[str, Any] = self.raw.get("flows", {})
        self.action_guides: dict[str, str] = {
            str(key): str(value)
            for key, value in (self.raw.get("action_guides", {}) or {}).items()
        }

    def render(self, flow: str, part: str, **values: object) -> str:
        raw_flow = self.flows.get(flow)
        if not isinstance(raw_flow, dict):
            raise KeyError(f"Unknown prompt flow: {flow}")
        text = str(raw_flow.get(part, "")).strip()
        if not text:
            raise KeyError(f"Unknown prompt part: {flow}.{part}")
        mapping = {key: str(value) for key, value in values.items()}
        return Template(text).safe_substitute(mapping).strip()

    def action_guide(self, action: str, default: str = "") -> str:
        return self.action_guides.get(action, default)
