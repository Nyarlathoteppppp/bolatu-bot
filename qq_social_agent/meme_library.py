from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .memory import MemeAsset, MemoryStore


@dataclass(frozen=True)
class MemeGateResult:
    allowed: bool
    reason: str
    messages_since_last_meme: int


class PrivateMemeLibrary:
    """A manually curated image pack. It never observes or imports live QQ images."""

    def __init__(self, memory: MemoryStore, raw: object, *, data_dir: Path) -> None:
        cfg = raw if isinstance(raw, dict) else {}
        self.memory = memory
        self.enabled = bool(cfg.get("enabled", True))
        self.allowed_private_users = _int_set(cfg.get("allowed_private_users", (1903297906,)))
        self.allowed_group_ids = _int_set(cfg.get("allowed_group_ids", ()))
        self.candidate_limit = max(1, min(10, int(cfg.get("candidate_limit", 5))))
        self.same_meme_cooldown_seconds = max(60, int(cfg.get("same_meme_cooldown_seconds", 6 * 60 * 60)))
        self.group_cooldown_seconds = max(60, int(cfg.get("group_cooldown_seconds", 15 * 60)))
        self.asset_dir = (data_dir / str(cfg.get("asset_dir", "meme_library"))).resolve()
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def turn_gate(self, user_id: int, *, received_messages: int) -> MemeGateResult:
        if not self.enabled or int(user_id) not in self.allowed_private_users:
            return MemeGateResult(False, "user_not_enabled", 0)
        state = self._state(user_id)
        count = max(0, int(state.get("messages_since_last_meme", 99))) + max(1, int(received_messages))
        state["messages_since_last_meme"] = count
        self._save_state(user_id, state)
        roll = random.random()
        if count <= 3 and roll < 0.70:
            return MemeGateResult(False, "early_turn_70_percent_skip", count)
        if count == 4 and roll < 0.20:
            return MemeGateResult(False, "fourth_turn_20_percent_skip", count)
        return MemeGateResult(True, "eligible", count)

    def candidates(self, user_id: int, *, query: str) -> list[MemeAsset]:
        if not self.enabled or int(user_id) not in self.allowed_private_users:
            return []
        return self._scope_candidates(self._state_key(user_id), query=query)

    def group_gate(self, group_id: int) -> MemeGateResult:
        if not self.enabled or int(group_id) not in self.allowed_group_ids:
            return MemeGateResult(False, "group_not_enabled", 0)
        state = self._state_for_key(self._group_state_key(group_id))
        last_sent_at = float(state.get("last_sent_at", 0) or 0)
        if last_sent_at and time.time() - last_sent_at < self.group_cooldown_seconds:
            return MemeGateResult(False, "group_cooldown", 0)
        return MemeGateResult(True, "eligible", 0)

    def group_candidates(self, group_id: int, *, query: str) -> list[MemeAsset]:
        if not self.enabled or int(group_id) not in self.allowed_group_ids:
            return []
        return self._scope_candidates(self._group_state_key(group_id), query=query)

    def candidate_text(self, assets: list[MemeAsset]) -> str:
        return "\n".join(
            f"- ID {asset.id}: {asset.description or '未写说明'}；标签：{'、'.join(asset.tags) or '无'}"
            for asset in assets
        )

    def image_base64_ref(self, meme_id: int) -> str:
        asset = self.memory.meme_asset(meme_id)
        if asset is None or not asset.enabled:
            return ""
        path = Path(asset.file_path).resolve()
        try:
            path.relative_to(self.asset_dir)
            payload = path.read_bytes()
        except (OSError, ValueError):
            return ""
        if not payload:
            return ""
        return "base64://" + base64.b64encode(payload).decode("ascii")

    def mark_sent(self, user_id: int, meme_id: int) -> bool:
        if not self.memory.mark_meme_asset_used(meme_id):
            return False
        state = self._state(user_id)
        state["messages_since_last_meme"] = 0
        self._record_scope_meme_sent(state, meme_id)
        self._save_state(user_id, state)
        return True

    def mark_group_sent(self, group_id: int, meme_id: int) -> bool:
        if not self.memory.mark_meme_asset_used(meme_id):
            return False
        state = self._state_for_key(self._group_state_key(group_id))
        state["last_sent_at"] = time.time()
        self._record_scope_meme_sent(state, meme_id)
        self._save_state_for_key(self._group_state_key(group_id), state)
        return True

    def _scope_candidates(self, state_key: str, *, query: str) -> list[MemeAsset]:
        """Keep meme reuse independent for each private chat and group.

        ``meme_assets.last_used_at`` remains global usage telemetry, but must not
        prevent an image used in one conversation from ever being offered in
        another.  The conversational cooldown lives in the matching app_kv
        state instead.
        """
        assets = self.memory.meme_assets_for_private(
            query=query,
            limit=80,
            same_meme_cooldown_seconds=0,
        )
        if not assets:
            return []
        state = self._state_for_key(state_key)
        recent = state.get("meme_sent_at", {})
        recent_by_id = recent if isinstance(recent, dict) else {}
        cutoff = time.time() - self.same_meme_cooldown_seconds
        available = [
            asset
            for asset in assets
            if _as_timestamp(recent_by_id.get(str(asset.id))) <= cutoff
        ]
        return available[: self.candidate_limit]

    @staticmethod
    def _record_scope_meme_sent(state: dict[str, object], meme_id: int) -> None:
        raw = state.get("meme_sent_at")
        sent_at = raw if isinstance(raw, dict) else {}
        sent_at[str(int(meme_id))] = time.time()
        state["meme_sent_at"] = sent_at

    def _state_key(self, user_id: int) -> str:
        return f"private_meme_library:{int(user_id)}"

    def _state(self, user_id: int) -> dict[str, object]:
        return self._state_for_key(self._state_key(user_id))

    def _state_for_key(self, key: str) -> dict[str, object]:
        raw = self.memory.app_kv_get(key)
        try:
            state = json.loads(raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            state = {}
        return state if isinstance(state, dict) else {}

    def _save_state(self, user_id: int, state: dict[str, object]) -> None:
        self._save_state_for_key(self._state_key(user_id), state)

    def _group_state_key(self, group_id: int) -> str:
        return f"group_meme_library:{int(group_id)}"

    def _save_state_for_key(self, key: str, state: dict[str, object]) -> None:
        self.memory.app_kv_set(key, json.dumps(state, ensure_ascii=False))


def _int_set(value: object) -> set[int]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    result: set[int] = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            result.add(parsed)
    return result


def _as_timestamp(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
