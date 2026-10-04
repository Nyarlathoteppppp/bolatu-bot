"""Style rule learning, merging and retrieval over the shared SQLite connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time

from .memory_models import ChatMessage, StyleRule
from .memory_repository_utils import (
    _loads_int_list,
    _message_from_row,
    _text_relevance_score,
    _unique_recent_ints,
)
from .memory_text import _compact_text

_STYLE_RULE_CANONICAL_REPLACEMENTS = (
    ("面对", "群友"),
    ("遇到", "群友"),
    ("当群友", "群友"),
    ("可以", ""),
    ("适合", ""),
    ("建议", ""),
    ("表达方式", ""),
    ("表达", ""),
    ("回应", "回复"),
    ("接话", "回复"),
    ("夸张比喻", "夸张"),
    ("夸张词", "夸张"),
    ("夸张方式", "夸张"),
    ("离谱内容", "荒诞"),
    ("荒谬观点", "荒诞"),
    ("荒诞疑问", "荒诞"),
    ("荒诞话题", "荒诞"),
    ("明显玩梗内容", "玩梗"),
    ("网络梗", "玩梗"),
    ("接梗", "玩梗"),
    ("用简短", "用短"),
    ("一句", "短句"),
    ("短促", "短"),
    ("调侃", "吐槽"),
    ("吐槽强化", "吐槽"),
    ("放大槽点", "放大荒诞"),
    ("突出反差", "放大荒诞"),
    ("强化喜剧效果", "放大荒诞"),
    ("放大荒诞感", "放大荒诞"),
    ("反讽短句", "反讽"),
    ("简短反讽句", "反讽"),
    ("黑色幽默", "反讽"),
    ("谐音梗", "谐音"),
    ("双关语", "双关"),
    ("制造幽默", "制造笑点"),
    ("制造双关效果", "制造笑点"),
    ("可爱语气词", "可爱语气"),
    ("拟声词", "可爱语气"),
    ("简短感叹句", "简短感叹"),
    ("简短感叹", "短句感叹"),
)


def _style_rule_fingerprint(situation: str, style: str) -> str:
    left = _normalize_style_rule_text(situation)
    right = _normalize_style_rule_text(style)
    if not left or not right:
        return ""
    return f"{left}=>{right}"[:160]


def _normalize_style_rule_text(text: str) -> str:
    normalized = _compact_text(text).casefold()
    for old, new in _STYLE_RULE_CANONICAL_REPLACEMENTS:
        normalized = normalized.replace(old.casefold(), new.casefold())
    normalized = re.sub(r"(的时候|时可以|时用|时要)", "时", normalized)
    normalized = re.sub(r"[，。、；：:,.!?！？\s]+", "", normalized)
    low_info = ("方式", "语气", "内容", "观点", "话题")
    for token in low_info:
        if len(normalized) > 12:
            normalized = normalized.replace(token, "")
    return normalized[:80]


_STYLE_RULE_SEMANTIC_MARKERS = (
    "夸张",
    "荒诞",
    "玩梗",
    "吐槽",
    "反讽",
    "短句",
    "短",
    "谐音",
    "双关",
    "互损",
    "自嘲",
    "安慰",
    "难受",
    "情绪",
    "认真",
    "技术",
    "代码",
    "搜索",
    "政治",
    "创造者",
    "小鸟",
    "可爱",
)


def _style_rule_semantic_markers(text: str) -> set[str]:
    normalized = _normalize_style_rule_text(text)
    return {marker for marker in _STYLE_RULE_SEMANTIC_MARKERS if marker in normalized}


def _style_rules_are_semantically_mergeable(
    existing_situation: str,
    existing_style: str,
    *,
    incoming_situation: str,
    incoming_style: str,
) -> bool:
    """Merge only clearly equivalent group-wide rules without an LLM call.

    A shared generic setting alone is not enough: both the triggering scene and
    the proposed expression must overlap on a known semantic marker. Personal
    style keeps using exact fingerprints so one member's habits never bleed into
    another member's profile.
    """
    existing_scene = _style_rule_semantic_markers(existing_situation)
    incoming_scene = _style_rule_semantic_markers(incoming_situation)
    existing_expression = _style_rule_semantic_markers(existing_style)
    incoming_expression = _style_rule_semantic_markers(incoming_style)
    return bool(existing_scene & incoming_scene) and bool(existing_expression & incoming_expression)


def _style_rule_confidence(
    *,
    support_user_count: int,
    evidence_count: int,
    merged_count: int,
    has_user_ids: bool,
) -> float:
    if not has_user_ids:
        base = 0.65
    elif support_user_count >= 2:
        base = 0.78
    else:
        base = 0.56
    base += min(0.14, max(0, evidence_count - 1) * 0.015)
    base += min(0.08, max(0, merged_count) * 0.01)
    base += min(0.08, max(0, support_user_count - 1) * 0.02)
    return round(max(0.1, min(0.94, base)), 3)


def _prefer_specific_style_text(old: str, new: str, *, limit: int) -> str:
    old = str(old or "").strip()
    new = str(new or "").strip()
    if not old:
        return new[:limit]
    if not new:
        return old[:limit]
    return (new if _style_text_specificity(new) > _style_text_specificity(old) else old)[:limit]


def _style_text_specificity(text: str) -> float:
    compact = _compact_text(text)
    score = min(4.0, len(compact) / 12)
    specific_markers = ("技术", "代码", "难受", "情绪", "创造者", "小鸟", "学术", "搜索", "认真", "安慰", "自嘲", "互损")
    score += sum(0.35 for marker in specific_markers if marker in compact)
    generic_markers = ("群友", "内容", "话题", "事情", "表达", "回应", "接话")
    score -= sum(0.18 for marker in generic_markers if marker in compact)
    return score


def _style_rule_value(row: sqlite3.Row, *, now: float) -> float:
    situation = str(row["situation"] or "")
    style = str(row["style"] or "")
    evidence = max(1, int(row["evidence_count"] or 1))
    support = max(1, int(row["support_user_count"] or 1))
    confidence = max(0.1, min(1.0, float(row["confidence"] or 0.6)))
    merged = max(0, int(row["merged_count"] or 0))
    last_seen = float(row["last_seen_at"] or row["created_at"] or 0.0)
    age_days = max(0.0, (now - last_seen) / (24 * 60 * 60))
    recency = 1.5 / (1.0 + age_days / 14.0)
    value = confidence * 3.0
    value += min(evidence, 20) * 0.18
    value += min(support, 8) * 0.42
    value += min(merged, 20) * 0.08
    value += recency
    value += _style_text_specificity(f"{situation}{style}") * 0.35
    if str(row["scope"] or "") == "personal":
        value -= 0.35
    generic_patterns = (
        "群友回复",
        "群友表达",
        "群友讨论",
        "群友提到",
        "用短句",
        "简短评价",
        "用调侃",
        "用玩梗",
    )
    compact = _compact_text(f"{situation}{style}")
    value -= sum(0.22 for marker in generic_patterns if marker in compact)
    if support <= 1:
        value -= 0.35
    if evidence <= 1:
        value -= 0.25
    return value


class StyleRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def messages_for_style_learning(
        self,
        group_id: int,
        *,
        limit: int,
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ? and is_bot = 0 and length(trim(text)) >= 2
            order by id desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def last_style_rule_at(self, group_id: int) -> float:
        row = self.conn.execute(
            "select max(created_at) as ts from style_rules where group_id = ?",
            (group_id,),
        ).fetchone()
        return float(row["ts"] or 0.0) if row else 0.0

    def add_style_rules(
        self,
        group_id: int,
        rules: list[tuple],
        *,
        keep: int = 45,
    ) -> dict[str, int]:
        now = time.time()
        clean_rules: list[tuple[str, str, str, tuple[int, ...], tuple[int, ...], str, str]] = []
        for raw_rule in rules:
            if len(raw_rule) < 3:
                continue
            situation, style, source_text = (str(raw_rule[0]), str(raw_rule[1]), str(raw_rule[2]))
            source_user_ids = tuple(int(value) for value in (raw_rule[3] if len(raw_rule) > 3 else ()) if int(value) > 0)
            source_message_ids = tuple(int(value) for value in (raw_rule[4] if len(raw_rule) > 4 else ()) if int(value) > 0)
            situation = situation.strip()
            style = style.strip()
            if situation and style:
                fingerprint = _style_rule_fingerprint(situation, style)
                if not fingerprint:
                    continue
                scope = "group" if not source_user_ids or len(set(source_user_ids)) >= 2 else "personal"
                clean_rules.append((situation, style, source_text.strip(), source_user_ids, source_message_ids, scope, fingerprint))
        if not clean_rules:
            return {"new": 0, "merged": 0, "expired": 0, "skipped": 0}
        stats = {"new": 0, "merged": 0, "expired": 0, "skipped": 0}
        for situation, style, source_text, user_ids, message_ids, scope, fingerprint in clean_rules:
            existing = self._find_mergeable_style_rule(
                group_id,
                fingerprint,
                situation=situation,
                style=style,
                scope=scope,
                source_user_ids=user_ids,
                now=now,
            )
            if existing is not None:
                self._merge_style_rule(
                    existing,
                    situation=situation,
                    style=style,
                    source_text=source_text,
                    source_user_ids=user_ids,
                    source_message_ids=message_ids,
                    now=now,
                )
                stats["merged"] += 1
                continue
            support_user_count = max(1, len(set(user_ids)))
            evidence_count = max(1, len(set(message_ids)))
            confidence = _style_rule_confidence(
                support_user_count=support_user_count,
                evidence_count=evidence_count,
                merged_count=0,
                has_user_ids=bool(user_ids),
            )
            self.conn.execute(
                """
                insert into style_rules(
                  group_id, situation, style, source_text, created_at, scope,
                  source_user_ids_json, source_message_ids_json, support_user_count,
                  evidence_count, confidence, status, valid_to,
                  rule_fingerprint, last_seen_at, merged_count
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, 0)
                """,
                (
                    group_id, situation[:60], style[:80], source_text[:200], now, scope,
                    json.dumps(_unique_recent_ints(user_ids), ensure_ascii=False),
                    json.dumps(_unique_recent_ints(message_ids, limit=20), ensure_ascii=False),
                    support_user_count, evidence_count, confidence,
                    now + (90 if scope == "group" else 30) * 24 * 60 * 60,
                    fingerprint, now,
                ),
            )
            stats["new"] += 1
        stats["expired"] = self._expire_low_value_style_rules(group_id, keep=keep, now=now)
        self.conn.commit()
        return stats

    def _find_mergeable_style_rule(
        self,
        group_id: int,
        fingerprint: str,
        *,
        situation: str,
        style: str,
        scope: str,
        source_user_ids: tuple[int, ...],
        now: float,
    ) -> sqlite3.Row | None:
        rows = self.conn.execute(
            """
            select id, group_id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to,
                   rule_fingerprint, last_seen_at, merged_count
            from style_rules
            where group_id = ?
              and status = 'active'
              and scope = ?
              and (valid_to is null or valid_to > ?)
            order by support_user_count desc, evidence_count desc, confidence desc, last_seen_at desc, id desc
            limit 80
            """,
            (group_id, scope, now),
        ).fetchall()
        for row in rows:
            existing_fingerprint = str(row["rule_fingerprint"] or "")
            if existing_fingerprint == fingerprint:
                if scope != "personal":
                    return row
                incoming_users = set(source_user_ids)
                existing_users = set(_loads_int_list(row["source_user_ids_json"]))
                if not incoming_users or not existing_users or incoming_users & existing_users:
                    return row
        if scope != "group":
            return None
        for row in rows:
            if _style_rules_are_semantically_mergeable(
                str(row["situation"] or ""),
                str(row["style"] or ""),
                incoming_situation=situation,
                incoming_style=style,
            ):
                return row
        return None

    def _merge_style_rule(
        self,
        row: sqlite3.Row,
        *,
        situation: str,
        style: str,
        source_text: str,
        source_user_ids: tuple[int, ...],
        source_message_ids: tuple[int, ...],
        now: float,
    ) -> None:
        old_user_ids = _loads_int_list(row["source_user_ids_json"])
        old_message_ids = _loads_int_list(row["source_message_ids_json"])
        merged_user_ids = _unique_recent_ints([*old_user_ids, *source_user_ids])
        merged_message_ids = _unique_recent_ints([*old_message_ids, *source_message_ids], limit=20)
        old_evidence_count = max(1, int(row["evidence_count"] or 1))
        incoming_evidence = max(1, len(set(source_message_ids)))
        evidence_count = old_evidence_count + incoming_evidence
        support_user_count = max(1, len(set(merged_user_ids)))
        merged_count = max(0, int(row["merged_count"] or 0)) + 1
        confidence = _style_rule_confidence(
            support_user_count=support_user_count,
            evidence_count=evidence_count,
            merged_count=merged_count,
            has_user_ids=bool(merged_user_ids),
        )
        scope = str(row["scope"] or "group")
        self.conn.execute(
            """
            update style_rules
            set situation = ?, style = ?, source_text = ?,
                source_user_ids_json = ?, source_message_ids_json = ?,
                support_user_count = ?, evidence_count = ?, confidence = ?,
                valid_to = ?, last_seen_at = ?, merged_count = ?
            where id = ?
            """,
            (
                _prefer_specific_style_text(str(row["situation"] or ""), situation, limit=60),
                _prefer_specific_style_text(str(row["style"] or ""), style, limit=80),
                (source_text or str(row["source_text"] or ""))[:200],
                json.dumps(merged_user_ids, ensure_ascii=False),
                json.dumps(merged_message_ids, ensure_ascii=False),
                support_user_count,
                evidence_count,
                confidence,
                now + (90 if scope == "group" else 30) * 24 * 60 * 60,
                now,
                merged_count,
                int(row["id"]),
            ),
        )

    def _expire_low_value_style_rules(self, group_id: int, *, keep: int, now: float) -> int:
        if keep <= 0:
            return 0
        rows = self.conn.execute(
            """
            select id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to,
                   rule_fingerprint, last_seen_at, merged_count
            from style_rules
            where group_id = ?
              and status = 'active'
              and (valid_to is null or valid_to > ?)
            """,
            (group_id, now),
        ).fetchall()
        if len(rows) <= keep:
            return 0
        ranked = sorted(
            rows,
            key=lambda row: (
                _style_rule_value(row, now=now),
                float(row["last_seen_at"] or row["created_at"] or 0.0),
                int(row["id"]),
            ),
            reverse=True,
        )
        expire_ids = [int(row["id"]) for row in ranked[keep:]]
        if not expire_ids:
            return 0
        self.conn.execute(
            f"""
            update style_rules
            set status = 'expired', valid_to = ?
            where id in ({','.join('?' for _ in expire_ids)})
            """,
            [now, *expire_ids],
        )
        return len(expire_ids)

    def recent_style_rules(self, group_id: int, limit: int) -> list[StyleRule]:
        rows = self.conn.execute(
            """
            select group_id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to,
                   rule_fingerprint, last_seen_at, merged_count
            from style_rules
            where group_id = ? and status = 'active' and (valid_to is null or valid_to > ?)
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, time.time(), limit),
        ).fetchall()
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
                scope=str(row["scope"]),
                source_user_ids=tuple(_loads_int_list(row["source_user_ids_json"])),
                source_message_ids=tuple(_loads_int_list(row["source_message_ids_json"])),
                support_user_count=int(row["support_user_count"]),
                evidence_count=int(row["evidence_count"]),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
                valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
                rule_fingerprint=str(row["rule_fingerprint"] or ""),
                last_seen_at=float(row["last_seen_at"] or row["created_at"] or 0.0),
                merged_count=int(row["merged_count"] or 0),
            )
            for row in reversed(rows)
        ]

    def relevant_style_rules(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        candidate_limit: int = 80,
        speaker_user_id: int | None = None,
    ) -> list[StyleRule]:
        rows = self.conn.execute(
            """
            select group_id, situation, style, source_text, created_at, scope,
                   source_user_ids_json, source_message_ids_json, support_user_count,
                   evidence_count, confidence, status, valid_to,
                   rule_fingerprint, last_seen_at, merged_count
            from style_rules
            where group_id = ? and status = 'active' and (valid_to is null or valid_to > ?)
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, time.time(), candidate_limit),
        ).fetchall()
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            source_user_ids = tuple(_loads_int_list(row["source_user_ids_json"]))
            if str(row["scope"]) == "personal" and speaker_user_id not in source_user_ids:
                continue
            haystack = f"{row['situation']} {row['style']} {row['source_text']}"
            score = _text_relevance_score(query, haystack)
            if score > 0:
                confidence = max(0.1, min(1.0, float(row["confidence"])))
                score = score * (0.5 + 0.5 * confidence) + min(2, max(0, int(row["support_user_count"]) - 1)) * 0.5
                last_seen = float(row["last_seen_at"] or row["created_at"] or 0.0)
                scored.append((score, last_seen, row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            StyleRule(
                group_id=int(row["group_id"]),
                situation=str(row["situation"]),
                style=str(row["style"]),
                source_text=str(row["source_text"]),
                created_at=float(row["created_at"]),
                scope=str(row["scope"]),
                source_user_ids=tuple(_loads_int_list(row["source_user_ids_json"])),
                source_message_ids=tuple(_loads_int_list(row["source_message_ids_json"])),
                support_user_count=int(row["support_user_count"]),
                evidence_count=int(row["evidence_count"]),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
                valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
                rule_fingerprint=str(row["rule_fingerprint"] or ""),
                last_seen_at=float(row["last_seen_at"] or row["created_at"] or 0.0),
                merged_count=int(row["merged_count"] or 0),
            )
            for _, _, row in scored[:limit]
        ]

    def migrate_focused_style_rules(self, group_id: int, focused_user_id: int) -> dict[str, int]:
        """Conservatively scope legacy rules from one prolific speaker without deleting useful group style."""
        rows = self.conn.execute(
            """
            select id, situation, style, source_text
            from style_rules
            where group_id = ? and scope = 'legacy' and status = 'active'
            order by id desc
            """,
            (group_id,),
        ).fetchall()
        personal = kept_group = expired_duplicates = expired_literal = 0
        seen_focused: set[tuple[str, str]] = set()
        marker = f"[#{str(focused_user_id)[-5:]}]"
        for row in rows:
            source_text = str(row["source_text"])
            exact = self.conn.execute(
                """
                select id, user_id from messages
                where group_id = ? and is_bot = 0 and text = ?
                order by id desc limit 1
                """,
                (group_id, source_text),
            ).fetchone()
            is_focused = bool(exact and int(exact["user_id"]) == focused_user_id) or source_text.startswith(marker) or marker in source_text[:80]
            if not is_focused:
                continue
            message_ids = [int(exact["id"])] if exact else []
            key = (_compact_text(str(row["situation"])), _compact_text(str(row["style"])))
            preserve_as_group = "目移" in f"{row['situation']} {row['style']} {source_text}"
            if preserve_as_group:
                self.conn.execute(
                    """
                    update style_rules
                    set scope = 'group', source_user_ids_json = ?, source_message_ids_json = ?,
                        confidence = 0.78, support_user_count = 1, evidence_count = 1
                    where id = ?
                    """,
                    (json.dumps([focused_user_id]), json.dumps(message_ids), int(row["id"])),
                )
                kept_group += 1
                continue
            literal_personal = any(token in str(row["style"]).casefold() for token in ("xhn", "扶她"))
            if literal_personal:
                self.conn.execute(
                    "update style_rules set status = 'expired', valid_to = ? where id = ?",
                    (time.time(), int(row["id"])),
                )
                expired_literal += 1
                continue
            if key in seen_focused:
                self.conn.execute(
                    "update style_rules set status = 'expired', valid_to = ? where id = ?",
                    (time.time(), int(row["id"])),
                )
                expired_duplicates += 1
                continue
            seen_focused.add(key)
            self.conn.execute(
                """
                    update style_rules
                    set scope = 'personal', source_user_ids_json = ?, source_message_ids_json = ?,
                    confidence = 0.55, support_user_count = 1, evidence_count = 1,
                    valid_to = ?
                    where id = ?
                """,
                (
                    json.dumps([focused_user_id]), json.dumps(message_ids),
                    time.time() + 60 * 24 * 60 * 60, int(row["id"]),
                ),
            )
            personal += 1
        self.conn.commit()
        return {
            "personal": personal,
            "kept_group": kept_group,
            "expired_duplicates": expired_duplicates,
            "expired_literal": expired_literal,
        }
