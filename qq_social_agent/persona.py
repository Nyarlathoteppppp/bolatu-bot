from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    description: str
    prompt: str
    decision_prompt: str
    keywords: tuple[str, ...]
    max_reply_chars: int
    passive_reply_probability: float


class PersonaRegistry:
    def __init__(self, persona_dir: Path):
        self.persona_dir = persona_dir
        self._personas = self._load_all()

    def _load_all(self) -> dict[str, Persona]:
        personas: dict[str, Persona] = {}
        for path in sorted(self.persona_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
            interests = raw.get("interests", {})
            style = raw.get("style", {})
            persona = Persona(
                id=str(raw["id"]),
                name=str(raw.get("name", raw["id"])),
                description=str(raw.get("description", "")),
                prompt=str(raw.get("prompt", "")),
                decision_prompt=str(raw.get("decision_prompt", raw.get("prompt", ""))),
                keywords=tuple(str(item) for item in interests.get("keywords", [])),
                max_reply_chars=int(style.get("max_reply_chars", 180)),
                passive_reply_probability=float(style.get("passive_reply_probability", 0.18)),
            )
            personas[persona.id] = persona
        return personas

    def get(self, persona_id: str) -> Persona:
        if persona_id not in self._personas:
            available = ", ".join(sorted(self._personas))
            raise KeyError(f"Unknown persona {persona_id!r}. Available: {available}")
        return self._personas[persona_id]

    def has(self, persona_id: str) -> bool:
        return persona_id in self._personas

    def ids(self) -> list[str]:
        return sorted(self._personas)
