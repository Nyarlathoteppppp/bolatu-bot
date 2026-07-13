from __future__ import annotations

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .rag_store import RAGStore


RAG_TIMEZONE = ZoneInfo("Asia/Shanghai")


class RAGIndexer:
    def __init__(
        self,
        store: RAGStore,
        *,
        episode_gap_seconds: int = 600,
        max_chunk_chars: int = 700,
        max_chunk_messages: int = 6,
    ):
        self.store = store
        self.episode_gap_seconds = max(30, int(episode_gap_seconds))
        self.max_chunk_chars = max(120, int(max_chunk_chars))
        self.max_chunk_messages = max(1, min(12, int(max_chunk_messages)))

    def sync_all(self, *, message_batch_limit: int = 12000) -> dict[str, int]:
        stats = {
            "conversation": self._sync_messages(message_batch_limit),
            "summary": self._sync_summaries(),
            "memory_atom": self._sync_memory_atoms(),
            "member": self._sync_member_profiles(),
            "jargon": self._sync_jargon(),
            "feedback": self._sync_feedback(),
        }
        self.store.commit()
        return stats

    def _sync_messages(self, batch_limit: int) -> int:
        groups = self.store.conn.execute(
            "select distinct group_id from messages where is_bot = 0"
        ).fetchall()
        indexed = 0
        for group_row in groups:
            group_id = int(group_row["group_id"])
            cursor_text = self.store.get_index_cursor("messages", str(group_id))
            cursor = int(cursor_text or 0)
            rows = self.store.conn.execute(
                """
                select id, user_id, nickname, text, created_at, source_message_id
                from messages
                where group_id = ? and is_bot = 0 and id > ? and length(trim(text)) >= 2
                order by id asc limit ?
                """,
                (group_id, cursor, batch_limit),
            ).fetchall()
            for chunk in self._message_chunks(rows):
                first, last = chunk[0], chunk[-1]
                lines = []
                message_ids: list[str] = []
                speakers: set[int] = set()
                for row in chunk:
                    stamp = datetime.fromtimestamp(
                        float(row["created_at"]), RAG_TIMEZONE
                    ).strftime("%Y-%m-%d %H:%M")
                    user_id = int(row["user_id"])
                    speakers.add(user_id)
                    lines.append(f"[{stamp}] {row['nickname']}[{str(user_id)[-5:]}]：{row['text']}")
                    message_ids.append(str(row["source_message_id"] or row["id"]))
                self.store.upsert_document(
                    stable_key=f"messages:{group_id}:{first['id']}:{last['id']}",
                    group_id=group_id,
                    doc_type="conversation",
                    content="\n".join(lines),
                    source_name="messages",
                    source_row_id=f"{first['id']}:{last['id']}",
                    source_message_ids=message_ids,
                    speaker_user_id=next(iter(speakers)) if len(speakers) == 1 else None,
                    asserted_by_user_id=next(iter(speakers)) if len(speakers) == 1 else None,
                    created_at=float(last["created_at"]),
                    importance=0.45,
                    confidence=0.65,
                )
                indexed += 1
            if rows:
                self.store.set_index_cursor("messages", str(group_id), int(rows[-1]["id"]))
        return indexed

    def _message_chunks(self, rows: list[object]) -> list[list[object]]:
        chunks: list[list[object]] = []
        current: list[object] = []
        current_chars = 0
        previous_at = 0.0
        for row in rows:
            text = str(row["text"]).strip()
            if not text:
                continue
            created_at = float(row["created_at"])
            boundary = bool(
                current
                and (
                    created_at - previous_at > self.episode_gap_seconds
                    or len(current) >= self.max_chunk_messages
                    or current_chars + len(text) > self.max_chunk_chars
                )
            )
            if boundary:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(row)
            current_chars += len(text)
            previous_at = created_at
        if current:
            chunks.append(current)
        return chunks

    def _sync_summaries(self) -> int:
        if not _table_exists(self.store, "memory_summaries"):
            return 0
        rows = self.store.conn.execute(
            """
            select id, group_id, start_message_id, end_message_id, summary,
                   recall_cues_json, start_at, end_at, created_at
            from memory_summaries
            """
        ).fetchall()
        for row in rows:
            cues = _json_list(row["recall_cues_json"])
            content = str(row["summary"])
            if cues:
                content += "\n回想线索：" + "；".join(cues[:8])
            self.store.upsert_document(
                stable_key=f"summary:{row['id']}",
                group_id=int(row["group_id"]),
                doc_type="summary",
                content=content,
                source_name="memory_summaries",
                source_row_id=row["id"],
                source_message_ids=[row["start_message_id"], row["end_message_id"]],
                created_at=float(row["created_at"]),
                valid_from=float(row["start_at"]),
                importance=0.65,
                confidence=0.65,
            )
        return len(rows)

    def _sync_memory_atoms(self) -> int:
        if not _table_exists(self.store, "memory_atoms"):
            return 0
        rows = self.store.conn.execute(
            """
            select id, atom_type, group_id, subject_user_id, object_user_id, content,
                   source, evidence_type, source_message_id, observed_at, valid_from,
                   valid_to, confidence, importance, status, created_at, updated_at
            from memory_atoms
            """
        ).fetchall()
        for row in rows:
            evidence = str(row["evidence_type"] or "manual")
            source = str(row["source"] or "")
            content = f"[{row['atom_type']}] {row['content']}\n证据类型：{evidence}；来源：{source}"
            self.store.upsert_document(
                stable_key=f"atom:{row['id']}",
                group_id=int(row["group_id"]),
                doc_type="memory_atom",
                content=content,
                source_name="memory_atoms",
                source_row_id=row["id"],
                source_message_ids=[row["source_message_id"]] if row["source_message_id"] else [],
                subject_user_id=int(row["subject_user_id"]) if row["subject_user_id"] is not None else None,
                asserted_by_user_id=int(row["subject_user_id"]) if evidence == "message" and row["subject_user_id"] is not None else None,
                created_at=float(row["observed_at"] or row["created_at"]),
                valid_from=float(row["valid_from"]) if row["valid_from"] is not None else None,
                valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
                importance=float(row["importance"] or 0.5),
                confidence=float(row["confidence"] or 0.6),
                status=str(row["status"] or "active"),
            )
        return len(rows)

    def _sync_member_profiles(self) -> int:
        total = 0
        if _table_exists(self.store, "member_profiles"):
            rows = self.store.conn.execute(
                "select group_id, user_id, display_name, aliases_json, last_seen_at from member_profiles"
            ).fetchall()
            for row in rows:
                aliases = _json_list(row["aliases_json"])
                self.store.upsert_document(
                    stable_key=f"member:{row['group_id']}:{row['user_id']}",
                    group_id=int(row["group_id"]),
                    doc_type="member",
                    content=(
                        f"群友 {row['display_name']}，QQ {row['user_id']}。"
                        f"历史昵称/别名：{'、'.join(aliases) if aliases else '无'}。"
                    ),
                    source_name="member_profiles",
                    source_row_id=row["user_id"],
                    subject_user_id=int(row["user_id"]),
                    created_at=float(row["last_seen_at"]),
                    importance=0.7,
                    confidence=0.9,
                )
            total += len(rows)
        if _table_exists(self.store, "member_profile_summaries"):
            rows = self.store.conn.execute(
                """
                select id, group_id, user_id, profile_summary, interests_json,
                       speaking_style, representative_texts_json, created_at
                from member_profile_summaries
                """
            ).fetchall()
            for row in rows:
                interests = _json_list(row["interests_json"])
                content = f"群友 QQ {row['user_id']} 的阶段画像：{row['profile_summary']}"
                if interests:
                    content += "\n兴趣：" + "、".join(interests[:10])
                if row["speaking_style"]:
                    content += "\n说话习惯（仅作画像，不作为事实）：" + str(row["speaking_style"])
                self.store.upsert_document(
                    stable_key=f"member_summary:{row['id']}",
                    group_id=int(row["group_id"]),
                    doc_type="member",
                    content=content,
                    source_name="member_profile_summaries",
                    source_row_id=row["id"],
                    subject_user_id=int(row["user_id"]),
                    created_at=float(row["created_at"]),
                    importance=0.62,
                    confidence=0.6,
                )
            total += len(rows)
        return total

    def _sync_jargon(self) -> int:
        if not _table_exists(self.store, "custom_jargon_entries"):
            return 0
        rows = self.store.conn.execute(
            "select id, group_id, term, explanation, created_by, created_at from custom_jargon_entries"
        ).fetchall()
        for row in rows:
            self.store.upsert_document(
                stable_key=f"jargon:{row['id']}",
                group_id=int(row["group_id"]),
                doc_type="jargon",
                content=f"群内黑话“{row['term']}”：{row['explanation']}",
                source_name="custom_jargon_entries",
                source_row_id=row["id"],
                asserted_by_user_id=int(row["created_by"]),
                created_at=float(row["created_at"]),
                importance=0.85,
                confidence=0.9,
            )
        return len(rows)

    def _sync_feedback(self) -> int:
        total = 0
        if _table_exists(self.store, "recalled_reply_feedback"):
            rows = self.store.conn.execute(
                """
                select id, group_id, trigger_user_id, trigger_nickname, trigger_text,
                       bot_reply, owner_reason, avoid_rule, better_direction, reason_at
                from recalled_reply_feedback
                """
            ).fetchall()
            for row in rows:
                self.store.upsert_document(
                    stable_key=f"negative_feedback:{row['id']}",
                    group_id=int(row["group_id"]),
                    doc_type="feedback",
                    content=(
                        f"审批负反馈场景：{row['trigger_nickname']}说“{row['trigger_text']}”。"
                        f"机器人候选“{row['bot_reply']}”的问题：{row['owner_reason']}。"
                        f"避免：{row['avoid_rule']}；更好方向：{row['better_direction']}"
                    ),
                    source_name="recalled_reply_feedback",
                    source_row_id=row["id"],
                    subject_user_id=int(row["trigger_user_id"]),
                    created_at=float(row["reason_at"]),
                    importance=0.8,
                    confidence=0.95,
                )
            total += len(rows)
        if _table_exists(self.store, "approved_reply_feedback"):
            rows = self.store.conn.execute(
                """
                select id, group_id, trigger_user_id, trigger_nickname, trigger_text,
                       candidate_text, style, created_at
                from approved_reply_feedback
                """
            ).fetchall()
            for row in rows:
                self.store.upsert_document(
                    stable_key=f"positive_feedback:{row['id']}",
                    group_id=int(row["group_id"]),
                    doc_type="feedback",
                    content=(
                        f"审批认可场景：{row['trigger_nickname']}说“{row['trigger_text']}”。"
                        f"认可的回复方向：{row['style']}。候选仅作历史证据：“{row['candidate_text']}”"
                    ),
                    source_name="approved_reply_feedback",
                    source_row_id=row["id"],
                    subject_user_id=int(row["trigger_user_id"]),
                    created_at=float(row["created_at"]),
                    importance=0.72,
                    confidence=0.9,
                )
            total += len(rows)
        return total


def _table_exists(store: RAGStore, table: str) -> bool:
    return store.conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone() is not None


def _json_list(value: object) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []
