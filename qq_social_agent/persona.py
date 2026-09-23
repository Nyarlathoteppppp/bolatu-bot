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
    max_reply_chars: int
    passive_reply_probability: float


_MISSING = object()


class PersonaRegistry:
    def __init__(self, persona_dir: Path):
        self.persona_dir = persona_dir
        self._personas = self._load_all()

    def _load_all(self) -> dict[str, Persona]:
        personas: dict[str, Persona] = {}
        for path in sorted(self.persona_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
            raw = raw.get("persona", raw)
            if "id" not in raw:
                continue
            style = raw.get("style", {})
            persona = Persona(
                id=str(raw["id"]),
                name=str(raw.get("name", raw["id"])),
                description=str(raw.get("description", "")),
                prompt=str(raw.get("prompt", "")),
                decision_prompt=str(raw.get("decision_prompt", raw.get("prompt", ""))),
                max_reply_chars=int(style.get("max_reply_chars", 180)),
                passive_reply_probability=float(style.get("passive_reply_probability", 0.18)),
            )
            personas[persona.id] = persona
        return personas

    def get(self, persona_id: str, default: Persona | None | object = _MISSING) -> Persona | None:
        persona = self.resolve(persona_id)
        if persona is not None:
            return persona
        if default is not _MISSING:
            return default  # type: ignore[return-value]
        available = ", ".join(sorted(self._personas))
        raise KeyError(f"Unknown persona {persona_id!r}. Available: {available}")

    def resolve(self, persona_id: str | None) -> Persona | None:
        key = str(persona_id or "").strip()
        if not key:
            return None
        if key in self._personas:
            return self._personas[key]
        for persona in self._personas.values():
            if persona.name == key or persona.id == key:
                return persona
        return None

    def has(self, persona_id: str) -> bool:
        return self.resolve(persona_id) is not None

    def ids(self) -> list[str]:
        return sorted(self._personas)
