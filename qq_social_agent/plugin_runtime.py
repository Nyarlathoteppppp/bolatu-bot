from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CAPABILITY_KEYS = ("commands", "tools", "scheduled_tasks", "web_routes", "event_handlers")
_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class PluginManifestError(ValueError):
    """Raised when a local plugin manifest is malformed."""


@dataclass(frozen=True)
class LocalPluginCapability:
    name: str
    description: str = ""
    permission: str = ""
    target: str = ""
    metadata: dict[str, object] | None = None

    def to_summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "target": self.target,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class LocalPluginManifest:
    plugin_id: str
    name: str
    version: str
    enabled: bool
    description: str
    entrypoint: str
    capabilities: dict[str, tuple[LocalPluginCapability, ...]]
    permissions: tuple[str, ...]
    settings: dict[str, object]
    path: str
    loaded_at: float

    @property
    def capability_count(self) -> int:
        return sum(len(items) for items in self.capabilities.values())

    def capability_counts(self) -> dict[str, int]:
        return {key: len(self.capabilities.get(key, ())) for key in CAPABILITY_KEYS}

    def to_summary(self) -> dict[str, object]:
        return {
            "id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "capabilities": {
                key: [item.to_summary() for item in self.capabilities.get(key, ())]
                for key in CAPABILITY_KEYS
            },
            "capability_counts": self.capability_counts(),
            "permissions": list(self.permissions),
            "settings": dict(self.settings),
            "path": self.path,
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True)
class LocalPluginRuntimeCapability:
    plugin_id: str
    plugin_name: str
    kind: str
    name: str
    description: str = ""
    permission: str = ""
    target: str = ""
    metadata: dict[str, object] | None = None

    def to_summary(self) -> dict[str, object]:
        return {
            "plugin_id": self.plugin_id,
            "plugin_name": self.plugin_name,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "target": self.target,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PluginLoadError:
    path: str
    error: str

    def to_summary(self) -> dict[str, str]:
        return {"path": self.path, "error": self.error}


class LocalPluginRegistry:
    """Loads local plugin manifests and builds an explicit execution plan.

    Runtime execution stays allowlisted by the host application: manifests declare
    commands/tools/tasks/routes/events, and qq_social_agent.plugin binds known
    targets to local implementations. This avoids arbitrary dynamic imports while
    letting plugins own capability switches, permissions, and UI metadata.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._plugins: dict[str, LocalPluginManifest] = {}
        self._errors: tuple[PluginLoadError, ...] = ()
        self._loaded_at = 0.0

    @property
    def loaded_at(self) -> float:
        return self._loaded_at

    @property
    def errors(self) -> tuple[PluginLoadError, ...]:
        return self._errors

    def reload(self) -> None:
        loaded: dict[str, LocalPluginManifest] = {}
        errors: list[PluginLoadError] = []
        now = time.time()
        for path in self._manifest_paths():
            try:
                manifest = load_local_plugin_manifest(path, loaded_at=now)
            except Exception as exc:
                errors.append(PluginLoadError(str(path), str(exc)))
                continue
            if manifest.plugin_id in loaded:
                errors.append(PluginLoadError(str(path), f"duplicate plugin id: {manifest.plugin_id}"))
                continue
            loaded[manifest.plugin_id] = manifest
        self._plugins = loaded
        self._errors = tuple(errors)
        self._loaded_at = now

    def plugins(self, *, include_disabled: bool = True) -> tuple[LocalPluginManifest, ...]:
        items = sorted(self._plugins.values(), key=lambda item: item.plugin_id)
        if include_disabled:
            return tuple(items)
        return tuple(item for item in items if item.enabled)

    def enabled_plugins(self) -> tuple[LocalPluginManifest, ...]:
        return self.plugins(include_disabled=False)

    def summary(self, *, include_disabled: bool = True) -> list[dict[str, object]]:
        return [plugin.to_summary() for plugin in self.plugins(include_disabled=include_disabled)]

    def status_payload(self) -> dict[str, object]:
        enabled = self.enabled_plugins()
        return {
            "root": str(self.root),
            "loaded_at": self._loaded_at,
            "total": len(self._plugins),
            "enabled": len(enabled),
            "disabled": len(self._plugins) - len(enabled),
            "errors": [error.to_summary() for error in self._errors],
            "capability_counts": _merge_capability_counts(enabled),
            "execution_plan": self.execution_plan(),
            "plugins": [plugin.to_summary() for plugin in enabled],
        }

    def capabilities(
        self,
        kind: str | None = None,
        *,
        enabled_only: bool = True,
    ) -> tuple[LocalPluginRuntimeCapability, ...]:
        normalized_kind = str(kind or "").strip()
        records: list[LocalPluginRuntimeCapability] = []
        for plugin in self.plugins(include_disabled=not enabled_only):
            if enabled_only and not plugin.enabled:
                continue
            for capability_kind in CAPABILITY_KEYS:
                if normalized_kind and capability_kind != normalized_kind:
                    continue
                for capability in plugin.capabilities.get(capability_kind, ()):
                    records.append(
                        LocalPluginRuntimeCapability(
                            plugin_id=plugin.plugin_id,
                            plugin_name=plugin.name,
                            kind=capability_kind,
                            name=capability.name,
                            description=capability.description,
                            permission=capability.permission,
                            target=capability.target,
                            metadata=capability.metadata,
                        )
                    )
        return tuple(records)

    def capability_enabled(
        self,
        kind: str,
        *,
        plugin_id: str = "",
        name: str = "",
        target: str = "",
        permission: str = "",
    ) -> bool:
        return any(
            _capability_matches(
                capability,
                plugin_id=plugin_id,
                name=name,
                target=target,
                permission=permission,
            )
            for capability in self.capabilities(kind, enabled_only=True)
        )

    def execution_plan(self) -> dict[str, list[dict[str, object]]]:
        return {
            kind: [capability.to_summary() for capability in self.capabilities(kind, enabled_only=True)]
            for kind in CAPABILITY_KEYS
        }

    def _manifest_paths(self) -> tuple[Path, ...]:
        if not self.root.exists():
            return ()
        paths = list(self.root.glob("*/manifest.yaml"))
        paths.extend(self.root.glob("*/manifest.yml"))
        paths.extend(self.root.glob("*.plugin.yaml"))
        paths.extend(self.root.glob("*.plugin.yml"))
        return tuple(sorted(path for path in paths if path.is_file()))


def load_local_plugin_manifest(path: Path | str, *, loaded_at: float | None = None) -> LocalPluginManifest:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise PluginManifestError(f"cannot read yaml: {exc}") from exc
    if not isinstance(raw, dict):
        raise PluginManifestError("manifest root must be a mapping")

    plugin_id = _required_string(raw, "id")
    if not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise PluginManifestError("id must match ^[a-z][a-z0-9_-]{1,63}$")
    name = _string(raw.get("name"), plugin_id)
    version = _string(raw.get("version"), "0.1.0")
    enabled = bool(raw.get("enabled", True))
    description = _string(raw.get("description"), "")
    entrypoint = _string(raw.get("entrypoint"), "")
    permissions = _string_tuple(raw.get("permissions"))
    settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}
    capabilities_raw = raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else {}
    capabilities = {
        key: _parse_capabilities(capabilities_raw.get(key), key=key)
        for key in CAPABILITY_KEYS
    }

    return LocalPluginManifest(
        plugin_id=plugin_id,
        name=name,
        version=version,
        enabled=enabled,
        description=description,
        entrypoint=entrypoint,
        capabilities=capabilities,
        permissions=permissions,
        settings=dict(settings),
        path=str(manifest_path),
        loaded_at=float(loaded_at if loaded_at is not None else time.time()),
    )


def _parse_capabilities(raw: object, *, key: str) -> tuple[LocalPluginCapability, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PluginManifestError(f"capabilities.{key} must be a list")
    parsed: list[LocalPluginCapability] = []
    for item in raw:
        if isinstance(item, str):
            parsed.append(LocalPluginCapability(name=item, target=item))
            continue
        if not isinstance(item, dict):
            raise PluginManifestError(f"capabilities.{key} items must be mappings or strings")
        name = _capability_name(item)
        if not name:
            raise PluginManifestError(f"capabilities.{key} item missing name")
        parsed.append(
            LocalPluginCapability(
                name=name,
                description=_string(item.get("description"), ""),
                permission=_string(item.get("permission"), ""),
                target=_capability_target(item, fallback=name),
                metadata={
                    str(k): v
                    for k, v in item.items()
                    if k not in {"name", "description", "permission"}
                    and isinstance(v, (str, int, float, bool, type(None), list, dict))
                },
            )
        )
    return tuple(parsed)


def _capability_name(item: dict[object, object]) -> str:
    for key in ("name", "command", "tool", "task", "route", "event", "handler"):
        value = item.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return ""


def _capability_target(item: dict[object, object], *, fallback: str) -> str:
    for key in ("handler", "route", "command", "tool", "task", "event", "target"):
        value = item.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return fallback


def _required_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise PluginManifestError(f"missing required field: {key}")
    return str(value).strip()


def _string(value: object, default: str) -> str:
    if isinstance(value, (str, int, float)) and str(value).strip():
        return str(value).strip()
    return default


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _capability_matches(
    capability: LocalPluginRuntimeCapability,
    *,
    plugin_id: str = "",
    name: str = "",
    target: str = "",
    permission: str = "",
) -> bool:
    if plugin_id and capability.plugin_id != plugin_id:
        return False
    if name and _normalized_key(capability.name) != _normalized_key(name):
        return False
    if target and _normalized_key(capability.target) != _normalized_key(target):
        return False
    if permission and _normalized_key(capability.permission) != _normalized_key(permission):
        return False
    return True


def _normalized_key(value: object) -> str:
    return str(value or "").strip().casefold()


def _merge_capability_counts(plugins: tuple[LocalPluginManifest, ...]) -> dict[str, int]:
    counts = {key: 0 for key in CAPABILITY_KEYS}
    for plugin in plugins:
        for key, value in plugin.capability_counts().items():
            counts[key] = counts.get(key, 0) + value
    return counts
