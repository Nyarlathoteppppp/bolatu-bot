"""Memory atom lifecycle, audit and retrieval over the shared SQLite connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time

from .memory_identity import PRIVATE_CHAT_ID_OFFSET, expand_linked_account_ids, linked_account_ids
from .memory_models import MemoryAtom, MemoryAtomAuditEvent
from .memory_repository_utils import _clamp_float, _relevance_terms, _source_message_key, _text_relevance_score


MEMORY_ATOM_EVIDENCE_TYPES = frozenset({"message", "event", "manual"})

MEMORY_ATOM_STATUSES = frozenset({"active", "superseded", "disputed", "expired"})

_MEMORY_ATOM_SELECT_COLUMNS = """
    id, atom_type, group_id, subject_user_id, object_user_id, content,
    source, evidence_type, source_message_id, observed_at,
    valid_from, valid_to, confidence, importance, status,
    supersedes_id, expires_at, created_at, updated_at
"""


class MemoryAtomRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def admin_recent_memory_atoms(
        self,
        *,
        group_id: int | None = None,
        status: str = "active",
        limit: int = 80,
        user_id: int | None = None,
        atom_type: str = "",
        query: str = "",
    ) -> list[MemoryAtom]:
        bounded = max(1, min(500, int(limit)))
        clauses: list[str] = []
        params: list[object] = []
        clean_status = status.strip().casefold()
        if group_id is not None:
            clauses.append("group_id = ?")
            params.append(int(group_id))
        if clean_status and clean_status != "all":
            clauses.append("status = ?")
            params.append(_normalize_memory_atom_status(clean_status))
        if user_id is not None:
            clauses.append("(subject_user_id = ? or object_user_id = ?)")
            params.extend((int(user_id), int(user_id)))
        clean_type = atom_type.strip()
        if clean_type and clean_type.casefold() != "all":
            clauses.append("atom_type = ?")
            params.append(clean_type[:32])
        clean_query = re.sub(r"\s+", " ", query).strip()
        if clean_query:
            like = f"%{clean_query[:120]}%"
            clauses.append(
                "(content like ? or source like ? or coalesce(source_message_id, '') like ? or cast(id as text) = ?)"
            )
            params.extend((like, like, like, clean_query))
        where = "where " + " and ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            {where}
            order by status = 'active' desc, importance desc, updated_at desc, id desc
            limit ?
            """,
            (*params, bounded),
        ).fetchall()
        return [_memory_atom_from_row(row) for row in rows]

    def admin_review_memory_atom(
        self,
        atom_id: int,
        *,
        action: str,
        actor_user_id: int | None = None,
        note: str = "",
    ) -> bool:
        atom = self.memory_atom(atom_id)
        if atom is None:
            return False
        clean_action = action.strip().casefold()
        now = time.time()
        detail = note.strip()[:420]
        try:
            if clean_action in {"expire", "expired", "delete"}:
                return self.expire_memory_atom(
                    atom_id,
                    reason=detail or "admin memory audit expired",
                    source="admin_ui",
                    actor_user_id=actor_user_id,
                )
            if clean_action in {"wrong_person", "dispute", "disputed"}:
                self.conn.execute(
                    "update memory_atoms set status = 'disputed', updated_at = ? where id = ?",
                    (now, atom_id),
                )
                audit_action = "marked_wrong_person" if clean_action == "wrong_person" else "disputed"
            elif clean_action in {"keep", "preserve", "active"}:
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'active', expires_at = null, valid_to = null, updated_at = ?
                    where id = ?
                    """,
                    (now, atom_id),
                )
                audit_action = "review_preserved"
            elif clean_action in {"boost", "important", "high"}:
                self.conn.execute(
                    """
                    update memory_atoms
                    set importance = min(1.0, importance + 0.15),
                        confidence = max(confidence, 0.82),
                        updated_at = ?
                    where id = ?
                    """,
                    (now, atom_id),
                )
                audit_action = "importance_boosted"
            elif clean_action in {"freeze", "pin"}:
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'active', expires_at = null, valid_to = null,
                        importance = max(importance, 0.86), confidence = max(confidence, 0.84),
                        updated_at = ?
                    where id = ?
                    """,
                    (now, atom_id),
                )
                audit_action = "pinned"
            else:
                return False
            self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action=audit_action,
                evidence_type="manual",
                source="admin_ui",
                source_message_id=None,
                actor_user_id=actor_user_id,
                detail=detail or audit_action,
                observed_at=now,
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def admin_merge_memory_atoms(
        self,
        source_atom_id: int,
        target_atom_id: int,
        *,
        actor_user_id: int | None = None,
        note: str = "",
    ) -> bool:
        source = self.memory_atom(source_atom_id)
        target = self.memory_atom(target_atom_id)
        if source is None or target is None or source.id == target.id:
            return False
        if source.group_id != target.group_id:
            return False
        if source.status not in {"active", "disputed"} or target.status not in {"active", "disputed"}:
            return False
        now = time.time()
        detail = note.strip()[:420] or f"merged #{source.id} into #{target.id}"
        try:
            self.conn.execute(
                """
                update memory_atoms
                set status = 'superseded', valid_to = ?, expires_at = ?, supersedes_id = ?, updated_at = ?
                where id = ?
                """,
                (now, now, target.id, now, source.id),
            )
            self.conn.execute(
                """
                update memory_atoms
                set importance = max(importance, ?), confidence = max(confidence, ?), updated_at = ?
                where id = ?
                """,
                (min(1.0, max(target.importance, source.importance)), min(1.0, max(target.confidence, source.confidence)), now, target.id),
            )
            self._insert_memory_atom_audit_event(
                atom_id=source.id,
                action="merged_into",
                evidence_type="manual",
                source="admin_ui",
                source_message_id=None,
                actor_user_id=actor_user_id,
                detail=detail,
                observed_at=now,
                metadata={"target_atom_id": target.id},
            )
            self._insert_memory_atom_audit_event(
                atom_id=target.id,
                action="merged_from",
                evidence_type="manual",
                source="admin_ui",
                source_message_id=None,
                actor_user_id=actor_user_id,
                detail=detail,
                observed_at=now,
                metadata={"source_atom_id": source.id},
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def upsert_memory_atom(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        expires_at: float | None = None,
        evidence_type: str | None = None,
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        status: str = "active",
        supersedes_id: int | None = None,
    ) -> int:
        clean_content = re.sub(r"\s+", " ", content).strip()
        if not clean_content:
            return 0
        clean_type = atom_type.strip()[:32] or "note"
        clean_source = source.strip()[:80] or "manual"
        source_key = _source_message_key(source_message_id)
        clean_evidence_type = _normalize_memory_evidence_type(
            evidence_type or _infer_memory_evidence_type(clean_source, source_key)
        )
        clean_status = _normalize_memory_atom_status(status)
        effective_valid_to = valid_to if valid_to is not None else expires_at
        now = time.time()
        existing = None
        if supersedes_id is None:
            existing = self.conn.execute(
                """
                select id, source, confidence, importance, expires_at,
                       evidence_type, source_message_id, observed_at,
                       valid_from, valid_to, status, supersedes_id
                from memory_atoms
                where group_id = ?
                  and atom_type = ?
                  and coalesce(subject_user_id, -1) = coalesce(?, -1)
                  and coalesce(object_user_id, -1) = coalesce(?, -1)
                  and content = ?
                  and status = 'active'
                order by updated_at desc
                limit 1
                """,
                (group_id, clean_type, subject_user_id, object_user_id, clean_content[:420]),
            ).fetchone()
        if existing:
            atom_id = int(existing["id"])
            before = {
                "source": str(existing["source"]),
                "confidence": float(existing["confidence"]),
                "importance": float(existing["importance"]),
                "expires_at": existing["expires_at"],
                "evidence_type": str(existing["evidence_type"]),
                "source_message_id": existing["source_message_id"],
                "observed_at": existing["observed_at"],
                "valid_from": existing["valid_from"],
                "valid_to": existing["valid_to"],
                "status": str(existing["status"]),
                "supersedes_id": existing["supersedes_id"],
            }
            after = {
                "source": clean_source,
                "confidence": _clamp_float(confidence, 0.0, 1.0),
                "importance": _clamp_float(importance, 0.0, 1.0),
                "expires_at": effective_valid_to,
                "evidence_type": clean_evidence_type,
                "source_message_id": source_key,
                "observed_at": observed_at if observed_at is not None else existing["observed_at"],
                "valid_from": valid_from if valid_from is not None else existing["valid_from"],
                "valid_to": effective_valid_to,
                "status": clean_status,
                "supersedes_id": supersedes_id if supersedes_id is not None else existing["supersedes_id"],
            }
            resulting_valid_from = after["valid_from"]
            resulting_valid_to = after["valid_to"]
            if (
                resulting_valid_from is not None
                and resulting_valid_to is not None
                and float(resulting_valid_to) < float(resulting_valid_from)
            ):
                raise ValueError("memory atom valid_to cannot be earlier than valid_from")
            if before == after:
                return atom_id
            try:
                self.conn.execute(
                    """
                    update memory_atoms
                    set source = ?, confidence = ?, importance = ?,
                        expires_at = ?, evidence_type = ?, source_message_id = ?,
                        observed_at = coalesce(?, observed_at),
                        valid_from = coalesce(?, valid_from), valid_to = ?,
                        status = ?, supersedes_id = coalesce(?, supersedes_id),
                        updated_at = ?
                    where id = ?
                    """,
                    (
                        clean_source,
                        _clamp_float(confidence, 0.0, 1.0),
                        _clamp_float(importance, 0.0, 1.0),
                        effective_valid_to,
                        clean_evidence_type,
                        source_key,
                        observed_at,
                        valid_from,
                        effective_valid_to,
                        clean_status,
                        supersedes_id,
                        now,
                        atom_id,
                    ),
                )
                self._insert_memory_atom_audit_event(
                    atom_id=atom_id,
                    action="refreshed",
                    evidence_type=clean_evidence_type,
                    source=clean_source,
                    source_message_id=source_key,
                    actor_user_id=None,
                    detail="legacy upsert refreshed existing atom",
                    observed_at=observed_at if observed_at is not None else now,
                    metadata={"before": before, "after": after},
                )
                self.conn.commit()
                return atom_id
            except Exception:
                self.conn.rollback()
                raise
        return self.add_memory_atom(
            atom_type=clean_type,
            group_id=group_id,
            content=clean_content,
            source=clean_source,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            evidence_type=clean_evidence_type,
            source_message_id=source_key,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=effective_valid_to,
            confidence=confidence,
            importance=importance,
            status=clean_status,
            supersedes_id=supersedes_id,
        )

    def add_memory_atom(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str = "manual",
        evidence_type: str | None = None,
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        status: str = "active",
        supersedes_id: int | None = None,
        actor_user_id: int | None = None,
        audit_detail: str = "",
    ) -> int:
        clean_content = re.sub(r"\s+", " ", content).strip()
        if not clean_content:
            return 0
        source_key = _source_message_key(source_message_id)
        clean_evidence_type = _normalize_memory_evidence_type(
            evidence_type or _infer_memory_evidence_type(source, source_key)
        )
        clean_status = _normalize_memory_atom_status(status)
        now = time.time()
        observed = float(observed_at) if observed_at is not None else now
        starts = float(valid_from) if valid_from is not None else observed
        ends = float(valid_to) if valid_to is not None else None
        if clean_status == "expired" and ends is None:
            ends = observed
        if ends is not None and ends < starts:
            raise ValueError("memory atom valid_to cannot be earlier than valid_from")
        try:
            superseded_event_id = 0
            if supersedes_id is not None:
                target = self.memory_atom(supersedes_id)
                if target is None:
                    raise ValueError(f"superseded memory atom does not exist: {supersedes_id}")
                if target.group_id != group_id:
                    raise ValueError("superseded memory atom must belong to the same group")
                if target.status not in {"active", "disputed"}:
                    raise ValueError(f"memory atom {supersedes_id} is already {target.status}")
                if clean_status != "active":
                    raise ValueError("a replacement memory atom must start as active")
                if target.valid_from is not None and observed < target.valid_from:
                    raise ValueError("replacement cannot predate the superseded atom's valid_from")
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'superseded', valid_to = ?, expires_at = ?, updated_at = ?
                    where id = ?
                    """,
                    (observed, observed, now, supersedes_id),
                )
                superseded_event_id = self._insert_memory_atom_audit_event(
                    atom_id=supersedes_id,
                    action="superseded",
                    evidence_type=clean_evidence_type,
                    source=source,
                    source_message_id=source_key,
                    actor_user_id=actor_user_id,
                    detail=audit_detail or "superseded by replacement atom",
                    observed_at=observed,
                )
            atom_id = self._insert_memory_atom_record(
                atom_type=atom_type,
                group_id=group_id,
                content=clean_content,
                source=source,
                evidence_type=clean_evidence_type,
                source_message_id=source_key,
                observed_at=observed,
                valid_from=starts,
                valid_to=ends,
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                confidence=confidence,
                importance=importance,
                status=clean_status,
                supersedes_id=supersedes_id,
                actor_user_id=actor_user_id,
                audit_action="created",
                audit_detail=audit_detail or "memory atom created",
                now=now,
            )
            if superseded_event_id:
                self.conn.execute(
                    "update memory_atom_audit_events set metadata_json = ? where id = ?",
                    (json.dumps({"replacement_atom_id": atom_id}), superseded_event_id),
                )
            self.conn.commit()
            return atom_id
        except Exception:
            self.conn.rollback()
            raise

    def _insert_memory_atom_record(
        self,
        *,
        atom_type: str,
        group_id: int,
        content: str,
        source: str,
        evidence_type: str,
        source_message_id: str | None,
        observed_at: float,
        valid_from: float,
        valid_to: float | None,
        subject_user_id: int | None,
        object_user_id: int | None,
        confidence: float,
        importance: float,
        status: str,
        supersedes_id: int | None,
        actor_user_id: int | None,
        audit_action: str,
        audit_detail: str,
        now: float,
    ) -> int:
        clean_type = atom_type.strip()[:32] or "note"
        clean_source = source.strip()[:80] or evidence_type
        cursor = self.conn.execute(
            """
            insert into memory_atoms(
              atom_type, group_id, subject_user_id, object_user_id, content,
              source, evidence_type, source_message_id, observed_at,
              valid_from, valid_to, confidence, importance, status,
              supersedes_id, expires_at, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_type,
                group_id,
                subject_user_id,
                object_user_id,
                content[:420],
                clean_source,
                evidence_type,
                source_message_id,
                observed_at,
                valid_from,
                valid_to,
                _clamp_float(confidence, 0.0, 1.0),
                _clamp_float(importance, 0.0, 1.0),
                status,
                supersedes_id,
                valid_to,
                now,
                now,
            ),
        )
        atom_id = int(cursor.lastrowid or 0)
        self._insert_memory_atom_audit_event(
            atom_id=atom_id,
            action=audit_action,
            evidence_type=evidence_type,
            source=clean_source,
            source_message_id=source_message_id,
            actor_user_id=actor_user_id,
            detail=audit_detail,
            observed_at=observed_at,
        )
        return atom_id

    def add_memory_counter_evidence(
        self,
        atom_id: int,
        *,
        content: str,
        source: str,
        evidence_type: str = "message",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        confidence: float = 0.8,
        mark_disputed: bool = True,
    ) -> int:
        atom = self.memory_atom(atom_id)
        clean_content = re.sub(r"\s+", " ", content).strip()
        if atom is None or not clean_content:
            return 0
        clean_evidence_type = _normalize_memory_evidence_type(evidence_type)
        observed = float(observed_at) if observed_at is not None else time.time()
        try:
            if mark_disputed and atom.status == "active":
                self.conn.execute(
                    "update memory_atoms set status = 'disputed', updated_at = ? where id = ?",
                    (time.time(), atom_id),
                )
            event_id = self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="counter_evidence",
                evidence_type=clean_evidence_type,
                source=source,
                source_message_id=_source_message_key(source_message_id),
                actor_user_id=actor_user_id,
                detail=clean_content[:420],
                observed_at=observed,
                metadata={"confidence": _clamp_float(confidence, 0.0, 1.0)},
            )
            self.conn.commit()
            return event_id
        except Exception:
            self.conn.rollback()
            raise

    def dispute_memory_atom(
        self,
        atom_id: int,
        *,
        content: str,
        source: str,
        evidence_type: str = "message",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        confidence: float = 0.8,
    ) -> bool:
        return bool(
            self.add_memory_counter_evidence(
                atom_id,
                content=content,
                source=source,
                evidence_type=evidence_type,
                source_message_id=source_message_id,
                observed_at=observed_at,
                actor_user_id=actor_user_id,
                confidence=confidence,
                mark_disputed=True,
            )
        )

    def expire_memory_atom(
        self,
        atom_id: int,
        *,
        reason: str = "",
        source: str = "manual",
        observed_at: float | None = None,
        actor_user_id: int | None = None,
    ) -> bool:
        atom = self.memory_atom(atom_id)
        if atom is None or atom.status not in {"active", "disputed"}:
            return False
        observed = float(observed_at) if observed_at is not None else time.time()
        if atom.valid_from is not None and observed < atom.valid_from:
            raise ValueError("expiry cannot predate the memory atom's valid_from")
        try:
            self.conn.execute(
                """
                update memory_atoms
                set status = 'expired', valid_to = ?, expires_at = ?, updated_at = ?
                where id = ?
                """,
                (observed, observed, time.time(), atom_id),
            )
            self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="expired",
                evidence_type="manual",
                source=source,
                source_message_id=None,
                actor_user_id=actor_user_id,
                detail=(reason.strip() or "memory atom expired")[:420],
                observed_at=observed,
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def correct_memory_atom(
        self,
        atom_id: int,
        *,
        content: str,
        source: str = "manual_correction",
        source_message_id: int | str | None = None,
        observed_at: float | None = None,
        actor_user_id: int | None = None,
        reason: str = "",
        confidence: float | None = None,
        importance: float | None = None,
        valid_to: float | None = None,
        atom_type: str | None = None,
        subject_user_id: int | None = None,
        object_user_id: int | None = None,
    ) -> int:
        old = self.memory_atom(atom_id)
        clean_content = re.sub(r"\s+", " ", content).strip()
        if old is None or old.status not in {"active", "disputed"} or not clean_content:
            return 0
        observed = float(observed_at) if observed_at is not None else time.time()
        if old.valid_from is not None and observed < old.valid_from:
            raise ValueError("correction cannot predate the memory atom's valid_from")
        if valid_to is not None and float(valid_to) < observed:
            raise ValueError("corrected memory valid_to cannot be earlier than observed_at")
        now = time.time()
        try:
            self.conn.execute(
                """
                update memory_atoms
                set status = 'superseded', valid_to = ?, expires_at = ?, updated_at = ?
                where id = ?
                """,
                (observed, observed, now, atom_id),
            )
            superseded_event_id = self._insert_memory_atom_audit_event(
                atom_id=atom_id,
                action="superseded",
                evidence_type="manual",
                source=source,
                source_message_id=_source_message_key(source_message_id),
                actor_user_id=actor_user_id,
                detail=(reason.strip() or "replaced by manual correction")[:420],
                observed_at=observed,
            )
            new_atom_id = self._insert_memory_atom_record(
                atom_type=atom_type or old.atom_type,
                group_id=old.group_id,
                content=clean_content,
                source=source,
                evidence_type="manual",
                source_message_id=_source_message_key(source_message_id),
                observed_at=observed,
                valid_from=observed,
                valid_to=valid_to,
                subject_user_id=old.subject_user_id if subject_user_id is None else subject_user_id,
                object_user_id=old.object_user_id if object_user_id is None else object_user_id,
                confidence=old.confidence if confidence is None else confidence,
                importance=old.importance if importance is None else importance,
                status="active",
                supersedes_id=old.id,
                actor_user_id=actor_user_id,
                audit_action="manual_correction",
                audit_detail=(reason.strip() or f"manual correction of atom {old.id}")[:420],
                now=now,
            )
            self.conn.execute(
                "update memory_atom_audit_events set metadata_json = ? where id = ?",
                (json.dumps({"replacement_atom_id": new_atom_id}), superseded_event_id),
            )
            self.conn.commit()
            return new_atom_id
        except Exception:
            self.conn.rollback()
            raise

    def memory_atom(self, atom_id: int) -> MemoryAtom | None:
        row = self.conn.execute(
            f"select {_MEMORY_ATOM_SELECT_COLUMNS} from memory_atoms where id = ?",
            (atom_id,),
        ).fetchone()
        return _memory_atom_from_row(row) if row is not None else None

    def memory_atom_audit_trail(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        rows = self.conn.execute(
            """
            select * from (
              select id, atom_id, action, evidence_type, source, source_message_id,
                     actor_user_id, detail, observed_at, created_at, metadata_json
              from memory_atom_audit_events
              where atom_id = ?
              order by created_at desc, id desc
              limit ?
            )
            order by created_at asc, id asc
            """,
            (atom_id, max(1, int(limit))),
        ).fetchall()
        return [_memory_atom_audit_event_from_row(row) for row in rows]

    def memory_atom_events(self, atom_id: int, *, limit: int = 100) -> list[MemoryAtomAuditEvent]:
        return self.memory_atom_audit_trail(atom_id, limit=limit)

    def _insert_memory_atom_audit_event(
        self,
        *,
        atom_id: int,
        action: str,
        evidence_type: str,
        source: str,
        source_message_id: str | None,
        actor_user_id: int | None,
        detail: str,
        observed_at: float,
        metadata: dict[str, object] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            insert into memory_atom_audit_events(
              atom_id, action, evidence_type, source, source_message_id,
              actor_user_id, detail, observed_at, created_at, metadata_json
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom_id,
                action.strip()[:32] or "updated",
                _normalize_memory_evidence_type(evidence_type),
                source.strip()[:80] or evidence_type,
                source_message_id,
                actor_user_id,
                detail.strip()[:420],
                observed_at,
                time.time(),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid or 0)

    def delete_memory_atom(self, atom_id: int) -> bool:
        return self.expire_memory_atom(atom_id, reason="legacy delete_memory_atom")

    def recent_memory_atoms(self, group_id: int, limit: int) -> list[MemoryAtom]:
        now = time.time()
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            where group_id = ?
              and status = 'active'
              and (valid_from is null or valid_from <= ?)
              and (valid_to is null or valid_to > ?)
              and (expires_at is null or expires_at > ?)
            order by importance desc, updated_at desc, id desc
            limit ?
            """,
            (group_id, now, now, now, limit),
        ).fetchall()
        return [_memory_atom_from_row(row) for row in rows]

    def active_memory_atoms_for_subject(
        self,
        group_id: int,
        subject_user_id: int | None,
        *,
        atom_types: tuple[str, ...] | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        if subject_user_id is None:
            return []
        subject_ids = sorted(linked_account_ids(subject_user_id))
        if not subject_ids:
            return []
        placeholders = ",".join("?" for _ in subject_ids)
        clauses = ["group_id = ?", "status = 'active'", f"subject_user_id in ({placeholders})"]
        params: list[object] = [int(group_id), *subject_ids]
        if atom_types:
            placeholders = ",".join("?" for _ in atom_types)
            clauses.append(f"atom_type in ({placeholders})")
            params.extend(atom_types)
        params.append(max(1, min(100, int(limit))))
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            where {" and ".join(clauses)}
            order by importance desc, updated_at desc, id desc
            limit ?
            """,
            tuple(params),
        ).fetchall()
        return [_memory_atom_from_row(row) for row in rows]

    def relevant_memory_atoms(
        self,
        group_id: int,
        query: str,
        *,
        subject_user_ids: list[int] | None = None,
        speaker_user_id: int | None = None,
        relationship_user_ids: list[int] | None = None,
        limit: int = 6,
        candidate_limit: int = 120,
        now: float | None = None,
    ) -> list[MemoryAtom]:
        current = time.time() if now is None else float(now)
        subject_set = expand_linked_account_ids(
            (
                *(subject_user_ids or ()),
                *(relationship_user_ids or ()),
                speaker_user_id,
            )
        )
        speaker_set = linked_account_ids(speaker_user_id)
        has_query_terms = bool(_relevance_terms(query))
        clauses = [
            "group_id = ?",
            "status = 'active'",
            "atom_type != 'jargon_candidate'",
            "(valid_from is null or valid_from <= ?)",
            "(valid_to is null or valid_to > ?)",
            "(expires_at is null or expires_at > ?)",
        ]
        params: list[object] = [group_id, current, current, current]
        if subject_set and not has_query_terms:
            placeholders = ",".join("?" for _ in subject_set)
            clauses.append(
                f"(subject_user_id in ({placeholders}) or object_user_id in ({placeholders}) or subject_user_id is null)"
            )
            params.extend(subject_set)
            params.extend(subject_set)
        rows = self.conn.execute(
            f"""
            select {_MEMORY_ATOM_SELECT_COLUMNS}
            from memory_atoms
            where {" and ".join(clauses)}
            """,
            tuple(params),
        ).fetchall()
        is_private_chat = group_id >= PRIVATE_CHAT_ID_OFFSET
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            content = str(row["content"])
            lexical_score = float(_text_relevance_score(query, content))
            score = lexical_score
            subject = row["subject_user_id"]
            obj = row["object_user_id"]
            person_match = bool(subject_set and (subject in subject_set or obj in subject_set))
            atom_type = str(row["atom_type"]).casefold()
            # A private conversation has one speaker, so person-match alone is
            # almost always true.  Do not let unrelated snack/anime preferences
            # crowd out the active subject just because they belong to this user.
            if (
                is_private_chat
                and lexical_score <= 0
                and person_match
                and atom_type not in {"identity", "relation", "fact"}
                and float(row["importance"] or 0.0) < 0.85
            ):
                continue
            if has_query_terms and lexical_score <= 0 and not person_match:
                continue
            if subject_set and subject in subject_set:
                score += 3.0
            if subject_set and obj in subject_set:
                score += 2.25
            if speaker_set and subject in speaker_set:
                score += 1.5
            elif speaker_set and obj in speaker_set:
                score += 0.75
            if atom_type == "relation" and subject_set:
                if subject in subject_set or obj in subject_set:
                    score += 0.75
            score += float(row["importance"] or 0.0) * 2.0
            score += float(row["confidence"] or 0.0) * 0.75
            score += _memory_atom_recency_score(row, now=current)
            score += _memory_atom_feedback_score(row)
            if score <= 0:
                continue
            scored.append((score, float(row["updated_at"]), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if candidate_limit > 0:
            scored = scored[:candidate_limit]
        return [_memory_atom_from_row(row) for _, _, row in scored[:limit]]

    def expire_due_memory_atoms(
        self,
        *,
        now: float | None = None,
        group_id: int | None = None,
    ) -> int:
        current = time.time() if now is None else float(now)
        return self._expire_due_memory_atoms(current, group_id=group_id)

    def _expire_due_memory_atoms(self, now: float, *, group_id: int | None = None) -> int:
        group_clause = " and group_id = ?" if group_id is not None else ""
        params: tuple[object, ...] = (now, group_id) if group_id is not None else (now,)
        rows = self.conn.execute(
            f"""
            select id, evidence_type, source, source_message_id
            from memory_atoms
            where status = 'active'
              and coalesce(valid_to, expires_at) is not null
              and coalesce(valid_to, expires_at) <= ?
              {group_clause}
            """,
            params,
        ).fetchall()
        try:
            for row in rows:
                atom_id = int(row["id"])
                self.conn.execute(
                    """
                    update memory_atoms
                    set status = 'expired',
                        valid_to = coalesce(valid_to, expires_at, ?),
                        expires_at = coalesce(expires_at, valid_to, ?),
                        updated_at = ?
                    where id = ?
                    """,
                    (now, now, now, atom_id),
                )
                self._insert_memory_atom_audit_event(
                    atom_id=atom_id,
                    action="expired",
                    evidence_type=str(row["evidence_type"] or "event"),
                    source="validity_window",
                    source_message_id=_source_message_key(row["source_message_id"]),
                    actor_user_id=None,
                    detail="validity window elapsed",
                    observed_at=now,
                )
            if rows:
                self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise


def _memory_atom_from_row(row: sqlite3.Row) -> MemoryAtom:
    columns = set(row.keys())
    subject = row["subject_user_id"]
    obj = row["object_user_id"]
    expires_at = row["expires_at"]
    source_message_id = row["source_message_id"] if "source_message_id" in columns else None
    observed_at = row["observed_at"] if "observed_at" in columns else row["created_at"]
    valid_from = row["valid_from"] if "valid_from" in columns else row["created_at"]
    valid_to = row["valid_to"] if "valid_to" in columns else expires_at
    supersedes_id = row["supersedes_id"] if "supersedes_id" in columns else None
    return MemoryAtom(
        id=int(row["id"]),
        atom_type=str(row["atom_type"]),
        group_id=int(row["group_id"]),
        subject_user_id=int(subject) if subject is not None else None,
        object_user_id=int(obj) if obj is not None else None,
        content=str(row["content"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        expires_at=float(expires_at) if expires_at is not None else None,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        evidence_type=str(row["evidence_type"] or "manual") if "evidence_type" in columns else "manual",
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        observed_at=float(observed_at) if observed_at is not None else float(row["created_at"]),
        valid_from=float(valid_from) if valid_from is not None else None,
        valid_to=float(valid_to) if valid_to is not None else None,
        status=str(row["status"] or "active") if "status" in columns else "active",
        supersedes_id=int(supersedes_id) if supersedes_id is not None else None,
    )


def _memory_atom_audit_event_from_row(row: sqlite3.Row) -> MemoryAtomAuditEvent:
    try:
        raw_metadata = json.loads(str(row["metadata_json"] or "{}"))
    except (TypeError, json.JSONDecodeError):
        raw_metadata = {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    source_message_id = row["source_message_id"]
    actor_user_id = row["actor_user_id"]
    return MemoryAtomAuditEvent(
        id=int(row["id"]),
        atom_id=int(row["atom_id"]),
        action=str(row["action"]),
        evidence_type=str(row["evidence_type"]),
        source=str(row["source"]),
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        actor_user_id=int(actor_user_id) if actor_user_id is not None else None,
        detail=str(row["detail"]),
        observed_at=float(row["observed_at"]),
        created_at=float(row["created_at"]),
        metadata=metadata,
    )


def _normalize_memory_evidence_type(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in MEMORY_ATOM_EVIDENCE_TYPES:
        raise ValueError(
            f"unsupported memory evidence type: {value!r}; "
            f"expected one of {sorted(MEMORY_ATOM_EVIDENCE_TYPES)}"
        )
    return normalized


def _infer_memory_evidence_type(source: str, source_message_id: str | None) -> str:
    if source_message_id:
        return "message"
    normalized = str(source or "").strip().casefold()
    if normalized.startswith("message:"):
        return "message"
    if normalized == "manual" or normalized.startswith(("manual:", "manual_", "builtin")):
        return "manual"
    return "event"


def _normalize_memory_atom_status(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in MEMORY_ATOM_STATUSES:
        raise ValueError(
            f"unsupported memory atom status: {value!r}; "
            f"expected one of {sorted(MEMORY_ATOM_STATUSES)}"
        )
    return normalized


def _memory_atom_recency_score(row: sqlite3.Row, *, now: float) -> float:
    observed_at = row["observed_at"]
    timestamp = float(observed_at) if observed_at is not None else float(row["updated_at"] or 0.0)
    age_seconds = max(0.0, now - timestamp)
    return 1.5 / (1.0 + age_seconds / (7 * 24 * 60 * 60))


def _memory_atom_feedback_score(row: sqlite3.Row) -> float:
    atom_type = str(row["atom_type"] or "").casefold()
    source = str(row["source"] or "").casefold()
    content = str(row["content"] or "")
    score = 0.0
    if atom_type == "feedback":
        score += 1.5
    if source.startswith(("approval_", "recall_", "owner_feedback")):
        score += 0.75
    if "不准奏反馈" in content or "优质反馈" in content:
        score += 0.5
    return score
