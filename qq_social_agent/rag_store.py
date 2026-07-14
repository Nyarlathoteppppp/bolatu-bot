from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from .temporal_evidence import default_evidence_kind


@dataclass(frozen=True)
class RAGDocument:
    id: int
    stable_key: str
    group_id: int
    doc_type: str
    content: str
    speaker_user_id: int | None
    subject_user_id: int | None
    source_name: str
    source_row_id: str
    source_message_ids: tuple[str, ...]
    participant_user_ids: tuple[int, ...]
    asserted_by_user_id: int | None
    created_at: float
    valid_from: float | None
    valid_to: float | None
    importance: float
    confidence: float
    status: str
    evidence_kind: str


@dataclass(frozen=True)
class RankedDocument:
    document: RAGDocument
    score: float


@dataclass(frozen=True)
class ResolvedMember:
    user_id: int
    matched_name: str


@dataclass(frozen=True)
class RAGEvaluationCase:
    id: int
    group_id: int
    query: str
    expected_terms: tuple[str, ...]
    expected_user_ids: tuple[int, ...]
    created_by: int


@dataclass(frozen=True)
class RAGKnowledgeSource:
    id: int
    group_id: int
    kind: str
    source_identity: str
    title: str
    content_hash: str
    stable_prefix: str
    version: int
    chunk_count: int
    source_message_id: str
    created_by: int
    status: str
    created_at: float
    updated_at: float


class RAGStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("pragma busy_timeout = 5000")
        self.conn.execute("pragma journal_mode = WAL")
        self.conn.execute("pragma synchronous = NORMAL")
        self.fts_available = False
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists rag_documents (
              id integer primary key autoincrement,
              stable_key text not null unique,
              group_id integer not null,
              doc_type text not null,
              content text not null,
              speaker_user_id integer,
              subject_user_id integer,
              source_name text not null,
              source_row_id text not null,
              source_message_ids_json text not null default '[]',
              participant_user_ids_json text not null default '[]',
              asserted_by_user_id integer,
              created_at real not null,
              valid_from real,
              valid_to real,
              importance real not null default 0.5,
              confidence real not null default 0.6,
              status text not null default 'active',
              evidence_kind text not null default 'unknown',
              content_hash text not null,
              embedding_status text not null default 'pending',
              embedding_model text,
              embedding_dim integer,
              indexed_at real not null,
              updated_at real not null
            );

            create index if not exists idx_rag_documents_group_type_time
              on rag_documents(group_id, doc_type, created_at desc);
            create index if not exists idx_rag_documents_embedding
              on rag_documents(embedding_status, status, id);
            create index if not exists idx_rag_documents_speaker
              on rag_documents(group_id, speaker_user_id, created_at desc);

            create table if not exists rag_embeddings (
              document_id integer primary key,
              model text not null,
              dimensions integer not null,
              vector_blob blob not null,
              norm real not null,
              created_at real not null,
              foreign key(document_id) references rag_documents(id) on delete cascade
            );

            create table if not exists rag_index_state (
              source_name text not null,
              scope_key text not null,
              cursor_value text not null,
              updated_at real not null,
              primary key(source_name, scope_key)
            );

            create table if not exists rag_retrieval_events (
              id integer primary key autoincrement,
              group_id integer not null,
              query_hash text not null,
              route text not null,
              lexical_count integer not null,
              semantic_count integer not null,
              injected_count integer not null,
              elapsed_ms integer not null,
              cache_hit integer not null default 0,
              source_ids_json text not null default '[]',
              details_json text not null default '{}',
              error text not null default '',
              created_at real not null
            );
            create index if not exists idx_rag_retrieval_events_time
              on rag_retrieval_events(created_at desc);

            create table if not exists rag_retrieval_feedback (
              id integer primary key autoincrement,
              group_id integer not null,
              retrieval_event_id integer not null,
              document_id integer not null,
              label text not null,
              note text not null default '',
              operator_id integer not null default 0,
              created_at real not null,
              foreign key(retrieval_event_id) references rag_retrieval_events(id),
              foreign key(document_id) references rag_documents(id)
            );
            create index if not exists idx_rag_feedback_document
              on rag_retrieval_feedback(document_id, created_at desc);

            create table if not exists rag_evaluation_cases (
              id integer primary key autoincrement,
              group_id integer not null,
              query text not null,
              expected_terms_json text not null default '[]',
              expected_user_ids_json text not null default '[]',
              created_by integer not null default 0,
              enabled integer not null default 1,
              created_at real not null,
              updated_at real not null
            );
            create unique index if not exists idx_rag_eval_group_query
              on rag_evaluation_cases(group_id, query);

            create table if not exists rag_knowledge_sources (
              id integer primary key autoincrement,
              group_id integer not null,
              kind text not null,
              source_identity text not null,
              identity_hash text not null,
              title text not null,
              content_hash text not null,
              stable_prefix text not null,
              version integer not null default 1,
              chunk_count integer not null default 0,
              source_message_id text not null default '',
              created_by integer not null default 0,
              status text not null default 'active',
              created_at real not null,
              updated_at real not null,
              unique(group_id, kind, identity_hash)
            );
            create index if not exists idx_rag_knowledge_sources_group_status
              on rag_knowledge_sources(group_id, status, updated_at desc);
            """
        )
        self._ensure_rag_v2_columns()
        try:
            self.conn.execute(
                """
                create virtual table if not exists rag_documents_fts using fts5(
                  content,
                  group_id unindexed,
                  doc_type unindexed,
                  tokenize='trigram'
                )
                """
            )
            self.fts_available = True
        except sqlite3.OperationalError:
            self.fts_available = False
        self.conn.commit()

    def _ensure_rag_v2_columns(self) -> None:
        document_columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(rag_documents)").fetchall()
        }
        if "participant_user_ids_json" not in document_columns:
            self.conn.execute(
                "alter table rag_documents add column participant_user_ids_json text not null default '[]'"
            )
        if "evidence_kind" not in document_columns:
            self.conn.execute(
                "alter table rag_documents add column evidence_kind text not null default 'unknown'"
            )
        self.conn.execute(
            """
            update rag_documents
            set evidence_kind = case doc_type
              when 'conversation' then 'reported_claim'
              when 'summary' then 'summary'
              when 'memory_atom' then 'structured_fact'
              when 'member' then 'profile_summary'
              when 'jargon' then 'curated_definition'
              when 'feedback' then 'approval_feedback'
              when 'file_knowledge' then 'reference'
              when 'web_knowledge' then 'reference'
              else 'unknown'
            end
            where evidence_kind is null or evidence_kind = '' or evidence_kind = 'unknown'
            """
        )
        self.conn.execute(
            """
            update rag_documents
            set evidence_kind = 'directory_fact'
            where doc_type = 'member' and source_name = 'member_profiles'
              and evidence_kind = 'profile_summary'
            """
        )
        retrieval_columns = {
            str(row["name"])
            for row in self.conn.execute("pragma table_info(rag_retrieval_events)").fetchall()
        }
        if "details_json" not in retrieval_columns:
            self.conn.execute(
                "alter table rag_retrieval_events add column details_json text not null default '{}'"
            )

    def upsert_document(
        self,
        *,
        stable_key: str,
        group_id: int,
        doc_type: str,
        content: str,
        source_name: str,
        source_row_id: int | str,
        source_message_ids: list[int | str] | tuple[int | str, ...] = (),
        participant_user_ids: list[int] | tuple[int, ...] = (),
        speaker_user_id: int | None = None,
        subject_user_id: int | None = None,
        asserted_by_user_id: int | None = None,
        created_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        importance: float = 0.5,
        confidence: float = 0.6,
        status: str = "active",
        evidence_kind: str | None = None,
    ) -> int:
        clean_content = re.sub(r"[ \t]+", " ", str(content)).strip()
        if not clean_content:
            return 0
        now = time.time()
        digest = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
        row = self.conn.execute(
            "select id, content_hash from rag_documents where stable_key = ?",
            (stable_key,),
        ).fetchone()
        changed = row is None or str(row["content_hash"]) != digest
        self.conn.execute(
            """
            insert into rag_documents(
              stable_key, group_id, doc_type, content, speaker_user_id, subject_user_id,
              source_name, source_row_id, source_message_ids_json, participant_user_ids_json,
              asserted_by_user_id,
              created_at, valid_from, valid_to, importance, confidence, status,
              evidence_kind, content_hash, embedding_status, indexed_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            on conflict(stable_key) do update set
              group_id=excluded.group_id,
              doc_type=excluded.doc_type,
              content=excluded.content,
              speaker_user_id=excluded.speaker_user_id,
              subject_user_id=excluded.subject_user_id,
              source_name=excluded.source_name,
              source_row_id=excluded.source_row_id,
              source_message_ids_json=excluded.source_message_ids_json,
              participant_user_ids_json=excluded.participant_user_ids_json,
              asserted_by_user_id=excluded.asserted_by_user_id,
              created_at=excluded.created_at,
              valid_from=excluded.valid_from,
              valid_to=excluded.valid_to,
              importance=excluded.importance,
              confidence=excluded.confidence,
              status=excluded.status,
              evidence_kind=excluded.evidence_kind,
              content_hash=excluded.content_hash,
              embedding_status=case
                when rag_documents.content_hash != excluded.content_hash then 'pending'
                else rag_documents.embedding_status
              end,
              indexed_at=excluded.indexed_at,
              updated_at=excluded.updated_at
            """,
            (
                stable_key,
                int(group_id),
                str(doc_type)[:32],
                clean_content,
                speaker_user_id,
                subject_user_id,
                str(source_name)[:48],
                str(source_row_id)[:80],
                json.dumps([str(value) for value in source_message_ids], ensure_ascii=False),
                json.dumps(sorted({int(value) for value in participant_user_ids if int(value) > 0})),
                asserted_by_user_id,
                float(created_at or now),
                valid_from,
                valid_to,
                max(0.0, min(1.0, float(importance))),
                max(0.0, min(1.0, float(confidence))),
                str(status)[:24] or "active",
                str(evidence_kind or default_evidence_kind(doc_type))[:32],
                digest,
                now,
                now,
            ),
        )
        doc_row = self.conn.execute(
            "select id from rag_documents where stable_key = ?", (stable_key,)
        ).fetchone()
        document_id = int(doc_row["id"])
        if changed:
            self.conn.execute("delete from rag_embeddings where document_id = ?", (document_id,))
        if self.fts_available:
            self.conn.execute("delete from rag_documents_fts where rowid = ?", (document_id,))
            if status == "active":
                self.conn.execute(
                    "insert into rag_documents_fts(rowid, content, group_id, doc_type) values (?, ?, ?, ?)",
                    (document_id, clean_content, str(group_id), str(doc_type)[:32]),
                )
        return document_id

    def resolve_query_members(
        self,
        group_id: int,
        query: str,
        *,
        limit: int = 8,
    ) -> list[ResolvedMember]:
        clean_query = re.sub(r"\s+", "", str(query)).casefold()
        if not clean_query:
            return []
        rows = self.conn.execute(
            """
            select
              p.user_id,
              p.display_name,
              p.aliases_json,
              coalesce(g.nickname, '') as nickname,
              coalesce(g.card, '') as card
            from member_profiles p
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
            where p.group_id = ?
            order by p.last_seen_at desc
            """,
            (int(group_id),),
        ).fetchall()
        # Context labels use five-digit QQ tails such as [#07496]. They are
        # attribution hints, not complete QQ numbers, and must never create
        # synthetic people like user 7496.
        explicit_ids = {
            int(value)
            for value in re.findall(r"(?<![#\d])(\d{5,12})(?!\d)", clean_query)
        }
        resolved: list[ResolvedMember] = []
        seen: set[int] = set()
        for row in rows:
            user_id = int(row["user_id"])
            names = [str(row["card"]), str(row["nickname"]), str(row["display_name"])]
            try:
                aliases = json.loads(str(row["aliases_json"] or "[]"))
            except json.JSONDecodeError:
                aliases = []
            if isinstance(aliases, list):
                names.extend(str(value) for value in aliases)
            matched_name = ""
            if user_id in explicit_ids:
                matched_name = str(user_id)
            else:
                candidates = sorted(
                    {
                        re.sub(r"\s+", "", name).strip().casefold()
                        for name in names
                        if len(re.sub(r"\s+", "", name).strip()) >= 2
                    },
                    key=len,
                    reverse=True,
                )
                matched_name = next((name for name in candidates if name in clean_query), "")
            if not matched_name or user_id in seen:
                continue
            seen.add(user_id)
            resolved.append(ResolvedMember(user_id=user_id, matched_name=matched_name))
            if len(resolved) >= max(1, limit):
                break
        for user_id in sorted(explicit_ids):
            if user_id not in seen and len(resolved) < max(1, limit):
                resolved.append(ResolvedMember(user_id=user_id, matched_name=str(user_id)))
        return resolved

    def commit(self) -> None:
        self.conn.commit()

    def get_index_cursor(self, source_name: str, scope_key: str) -> str:
        row = self.conn.execute(
            "select cursor_value from rag_index_state where source_name = ? and scope_key = ?",
            (source_name, scope_key),
        ).fetchone()
        return str(row["cursor_value"]) if row else ""

    def set_index_cursor(self, source_name: str, scope_key: str, value: int | str) -> None:
        self.conn.execute(
            """
            insert into rag_index_state(source_name, scope_key, cursor_value, updated_at)
            values (?, ?, ?, ?)
            on conflict(source_name, scope_key) do update set
              cursor_value=excluded.cursor_value, updated_at=excluded.updated_at
            """,
            (source_name, scope_key, str(value), time.time()),
        )

    def lexical_search(
        self,
        group_id: int,
        query: str,
        *,
        limit: int,
        doc_types: tuple[str, ...],
        exclude_recent_after: float | None = None,
    ) -> list[RankedDocument]:
        if not query.strip():
            return []
        type_clause = ",".join("?" for _ in doc_types)
        base_params: list[object] = [int(group_id), *doc_types]
        recent_sql = ""
        recent_params: list[object] = []
        if exclude_recent_after is not None:
            recent_sql = "and not (d.doc_type = 'conversation' and d.created_at >= ?)"
            recent_params.append(float(exclude_recent_after))
        rows: list[sqlite3.Row] = []
        fts_query = _fts_query(query)
        if self.fts_available and fts_query:
            try:
                rows = self.conn.execute(
                    f"""
                    select d.*, bm25(rag_documents_fts) as rank_score
                    from rag_documents_fts
                    join rag_documents d on d.id = rag_documents_fts.rowid
                    where rag_documents_fts match ?
                      and d.group_id = ?
                      and d.doc_type in ({type_clause})
                      and d.status = 'active'
                      and (d.valid_from is null or d.valid_from <= ?)
                      and (d.valid_to is null or d.valid_to > ?)
                      {recent_sql}
                    order by rank_score asc, d.importance desc, d.created_at desc
                    limit ?
                    """,
                    (
                        fts_query,
                        *base_params,
                        time.time(),
                        time.time(),
                        *recent_params,
                        limit,
                    ),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            terms = _like_terms(query)
            if not terms:
                return []
            like_clause = " or ".join("d.content like ?" for _ in terms)
            like_params = [f"%{term}%" for term in terms]
            rows = self.conn.execute(
                f"""
                select d.*, 0.0 as rank_score
                from rag_documents d
                where d.group_id = ?
                  and d.doc_type in ({type_clause})
                  and d.status = 'active'
                  and ({like_clause})
                  and (d.valid_from is null or d.valid_from <= ?)
                  and (d.valid_to is null or d.valid_to > ?)
                  {recent_sql}
                order by d.importance desc, d.created_at desc
                limit ?
                """,
                (
                    *base_params,
                    *like_params,
                    time.time(),
                    time.time(),
                    *recent_params,
                    limit,
                ),
            ).fetchall()
        return [
            RankedDocument(_document_from_row(row), 1.0 / (1.0 + index * 0.25))
            for index, row in enumerate(rows)
        ]

    def pending_embedding_documents(
        self,
        *,
        limit: int,
        min_chars: int = 4,
    ) -> list[RAGDocument]:
        rows = self.conn.execute(
            """
            select * from rag_documents
            where embedding_status = 'pending'
              and status = 'active'
              and length(trim(content)) >= ?
            order by importance desc, id asc
            limit ?
            """,
            (min_chars, limit),
        ).fetchall()
        return [_document_from_row(row) for row in rows]

    def save_embeddings(self, documents: list[RAGDocument], vectors: list[list[float]], model: str) -> None:
        now = time.time()
        for document, vector in zip(documents, vectors):
            if not vector:
                continue
            norm = math.sqrt(sum(float(value) * float(value) for value in vector))
            blob = struct.pack(f"<{len(vector)}f", *vector)
            self.conn.execute(
                """
                insert into rag_embeddings(document_id, model, dimensions, vector_blob, norm, created_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                  model=excluded.model, dimensions=excluded.dimensions,
                  vector_blob=excluded.vector_blob, norm=excluded.norm, created_at=excluded.created_at
                """,
                (document.id, model, len(vector), blob, norm, now),
            )
            self.conn.execute(
                """
                update rag_documents
                set embedding_status='ready', embedding_model=?, embedding_dim=?, updated_at=?
                where id=?
                """,
                (model, len(vector), now, document.id),
            )
        self.conn.commit()

    def mark_embedding_failed(self, document_ids: list[int]) -> None:
        if not document_ids:
            return
        placeholders = ",".join("?" for _ in document_ids)
        self.conn.execute(
            f"update rag_documents set embedding_status='failed', updated_at=? where id in ({placeholders})",
            (time.time(), *document_ids),
        )
        self.conn.commit()

    def semantic_search(
        self,
        group_id: int,
        query_vector: list[float],
        *,
        model: str,
        limit: int,
        doc_types: tuple[str, ...],
        exclude_recent_after: float | None = None,
    ) -> list[RankedDocument]:
        if not query_vector:
            return []
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm <= 0:
            return []
        type_clause = ",".join("?" for _ in doc_types)
        recent_sql = ""
        params: list[object] = [int(group_id), *doc_types, model, len(query_vector), time.time(), time.time()]
        if exclude_recent_after is not None:
            recent_sql = "and not (d.doc_type = 'conversation' and d.created_at >= ?)"
            params.append(float(exclude_recent_after))
        rows = self.conn.execute(
            f"""
            select d.*, e.vector_blob, e.norm
            from rag_documents d
            join rag_embeddings e on e.document_id = d.id
            where d.group_id = ?
              and d.doc_type in ({type_clause})
              and e.model = ? and e.dimensions = ?
              and d.status = 'active'
              and (d.valid_from is null or d.valid_from <= ?)
              and (d.valid_to is null or d.valid_to > ?)
              {recent_sql}
            """,
            params,
        ).fetchall()
        scored: list[RankedDocument] = []
        for row in rows:
            vector = struct.unpack(f"<{len(query_vector)}f", row["vector_blob"])
            denominator = query_norm * float(row["norm"] or 0.0)
            if denominator <= 0:
                continue
            cosine = sum(a * b for a, b in zip(query_vector, vector)) / denominator
            if cosine > 0:
                scored.append(RankedDocument(_document_from_row(row), float(cosine)))
        scored.sort(key=lambda item: (item.score, item.document.importance, item.document.created_at), reverse=True)
        return scored[:limit]

    def person_documents(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        doc_types: tuple[str, ...],
        limit: int = 20,
    ) -> list[RankedDocument]:
        targets = sorted({int(value) for value in user_ids if int(value) > 0})
        if not targets:
            return []
        type_clause = ",".join("?" for _ in doc_types)
        user_clause = ",".join("?" for _ in targets)
        rows = self.conn.execute(
            f"""
            select d.* from rag_documents d
            where d.group_id = ? and d.doc_type in ({type_clause}) and d.status = 'active'
              and (
                d.speaker_user_id in ({user_clause})
                or d.subject_user_id in ({user_clause})
                or exists (
                  select 1 from json_each(d.participant_user_ids_json) p
                  where cast(p.value as integer) in ({user_clause})
                )
              )
              and (d.valid_from is null or d.valid_from <= ?)
              and (d.valid_to is null or d.valid_to > ?)
            order by d.importance desc, d.confidence desc, d.created_at desc
            limit ?
            """,
            (
                int(group_id), *doc_types, *targets, *targets, *targets,
                time.time(), time.time(), max(1, int(limit)),
            ),
        ).fetchall()
        return [
            RankedDocument(_document_from_row(row), max(0.35, 0.62 - index * 0.025))
            for index, row in enumerate(rows)
        ]

    def record_retrieval(
        self,
        *,
        group_id: int,
        query: str,
        route: str,
        lexical_count: int,
        semantic_count: int,
        injected_ids: list[int],
        details: dict[str, object] | None = None,
        elapsed_ms: int,
        cache_hit: bool,
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            insert into rag_retrieval_events(
              group_id, query_hash, route, lexical_count, semantic_count,
              injected_count, elapsed_ms, cache_hit, source_ids_json, details_json,
              error, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(group_id),
                hashlib.sha256(query.encode("utf-8")).hexdigest()[:20],
                route[:32],
                lexical_count,
                semantic_count,
                len(injected_ids),
                elapsed_ms,
                int(cache_hit),
                json.dumps(injected_ids),
                json.dumps(details or {}, ensure_ascii=False),
                error[:240],
                time.time(),
            ),
        )
        self.conn.commit()

    def delete_documents_by_stable_prefix_except(
        self,
        prefix: str,
        keep_keys: list[str] | tuple[str, ...],
    ) -> int:
        rows = self.conn.execute(
            "select id, stable_key from rag_documents where stable_key like ?",
            (f"{prefix}%",),
        ).fetchall()
        keep = set(keep_keys)
        removed = 0
        for row in rows:
            if str(row["stable_key"]) in keep:
                continue
            document_id = int(row["id"])
            if self.fts_available:
                self.conn.execute("delete from rag_documents_fts where rowid = ?", (document_id,))
            self.conn.execute("delete from rag_embeddings where document_id = ?", (document_id,))
            self.conn.execute("delete from rag_documents where id = ?", (document_id,))
            removed += 1
        return removed

    def register_knowledge_source(
        self,
        *,
        group_id: int,
        kind: str,
        source_identity: str,
        title: str,
        content_hash: str,
        source_message_id: str = "",
        created_by: int = 0,
    ) -> tuple[RAGKnowledgeSource, bool]:
        clean_kind = "file" if kind == "file" else "web"
        clean_identity = str(source_identity).strip()[:500]
        identity_hash = hashlib.sha256(clean_identity.encode("utf-8")).hexdigest()
        existing_duplicate = self.conn.execute(
            """
            select * from rag_knowledge_sources
            where group_id = ? and kind = ? and content_hash = ? and status = 'active'
            order by updated_at desc limit 1
            """,
            (int(group_id), clean_kind, content_hash),
        ).fetchone()
        if existing_duplicate is not None:
            return _knowledge_source_from_row(existing_duplicate), True
        existing = self.conn.execute(
            """
            select * from rag_knowledge_sources
            where group_id = ? and kind = ? and identity_hash = ?
            """,
            (int(group_id), clean_kind, identity_hash),
        ).fetchone()
        now = time.time()
        if existing is None:
            cursor = self.conn.execute(
                """
                insert into rag_knowledge_sources(
                  group_id, kind, source_identity, identity_hash, title, content_hash,
                  stable_prefix, version, chunk_count, source_message_id, created_by,
                  status, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, '', 1, 0, ?, ?, 'active', ?, ?)
                """,
                (
                    int(group_id), clean_kind, clean_identity, identity_hash,
                    str(title).strip()[:180], content_hash, str(source_message_id)[:80],
                    int(created_by), now, now,
                ),
            )
            source_id = int(cursor.lastrowid)
            prefix = f"knowledge:{clean_kind}:{int(group_id)}:{source_id}:"
            self.conn.execute(
                "update rag_knowledge_sources set stable_prefix = ? where id = ?",
                (prefix, source_id),
            )
        else:
            source_id = int(existing["id"])
            version = int(existing["version"] or 0) + 1
            self.conn.execute(
                """
                update rag_knowledge_sources
                set source_identity=?, title=?, content_hash=?, version=?,
                    source_message_id=?, created_by=?, status='active', updated_at=?
                where id=?
                """,
                (
                    clean_identity, str(title).strip()[:180], content_hash, version,
                    str(source_message_id)[:80], int(created_by), now, source_id,
                ),
            )
        row = self.conn.execute(
            "select * from rag_knowledge_sources where id = ?", (source_id,)
        ).fetchone()
        return _knowledge_source_from_row(row), False

    def update_knowledge_source_chunks(self, source_id: int, chunk_count: int) -> None:
        self.conn.execute(
            "update rag_knowledge_sources set chunk_count=?, updated_at=? where id=?",
            (max(0, int(chunk_count)), time.time(), int(source_id)),
        )

    def knowledge_sources(
        self,
        group_id: int,
        *,
        include_deleted: bool = False,
        limit: int = 50,
    ) -> list[RAGKnowledgeSource]:
        status_sql = "" if include_deleted else "and status = 'active'"
        rows = self.conn.execute(
            f"""
            select * from rag_knowledge_sources
            where group_id = ? {status_sql}
            order by updated_at desc limit ?
            """,
            (int(group_id), max(1, min(200, int(limit)))),
        ).fetchall()
        return [_knowledge_source_from_row(row) for row in rows]

    def delete_knowledge_source(self, group_id: int, source_id: int) -> bool:
        source = self.conn.execute(
            "select * from rag_knowledge_sources where id=? and group_id=? and status='active'",
            (int(source_id), int(group_id)),
        ).fetchone()
        if source is None:
            return False
        prefix = str(source["stable_prefix"])
        rows = self.conn.execute(
            "select id from rag_documents where stable_key like ? and status='active'",
            (f"{prefix}%",),
        ).fetchall()
        for row in rows:
            document_id = int(row["id"])
            if self.fts_available:
                self.conn.execute("delete from rag_documents_fts where rowid=?", (document_id,))
        self.conn.execute(
            "update rag_documents set status='inactive', valid_to=?, updated_at=? where stable_key like ?",
            (time.time(), time.time(), f"{prefix}%"),
        )
        self.conn.execute(
            "update rag_knowledge_sources set status='deleted', updated_at=? where id=?",
            (time.time(), int(source_id)),
        )
        self.conn.commit()
        return True

    def reindex_knowledge_source(self, group_id: int, source_id: int) -> int:
        source = self.conn.execute(
            "select * from rag_knowledge_sources where id=? and group_id=? and status='active'",
            (int(source_id), int(group_id)),
        ).fetchone()
        if source is None:
            return 0
        prefix = str(source["stable_prefix"])
        rows = self.conn.execute(
            "select id, content, doc_type from rag_documents where stable_key like ? and status='active'",
            (f"{prefix}%",),
        ).fetchall()
        now = time.time()
        for row in rows:
            document_id = int(row["id"])
            self.conn.execute("delete from rag_embeddings where document_id=?", (document_id,))
            self.conn.execute(
                "update rag_documents set embedding_status='pending', updated_at=? where id=?",
                (now, document_id),
            )
            if self.fts_available:
                self.conn.execute("delete from rag_documents_fts where rowid=?", (document_id,))
                self.conn.execute(
                    "insert into rag_documents_fts(rowid,content,group_id,doc_type) values(?,?,?,?)",
                    (document_id, str(row["content"]), str(group_id), str(row["doc_type"])),
                )
        self.conn.commit()
        return len(rows)

    def feedback_adjustments(self, document_ids: list[int]) -> dict[int, float]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        rows = self.conn.execute(
            f"""
            select document_id, label, count(*) as count
            from rag_retrieval_feedback
            where document_id in ({placeholders})
            group by document_id, label
            """,
            document_ids,
        ).fetchall()
        weights = {"relevant": 0.10, "irrelevant": -0.16, "wrong_person": -0.28, "stale": -0.34}
        result: dict[int, float] = {}
        for row in rows:
            document_id = int(row["document_id"])
            result[document_id] = result.get(document_id, 0.0) + weights.get(
                str(row["label"]), 0.0
            ) * min(3, int(row["count"]))
        return {key: max(-0.55, min(0.30, value)) for key, value in result.items()}

    def add_feedback_for_latest(
        self,
        *,
        group_id: int,
        position: int,
        label: str,
        operator_id: int,
        note: str = "",
    ) -> tuple[int, str]:
        allowed = {"relevant", "irrelevant", "wrong_person", "stale"}
        if label not in allowed:
            raise ValueError("unknown feedback label")
        row = self.conn.execute(
            """
            select id, source_ids_json from rag_retrieval_events
            where group_id = ? and injected_count > 0
            order by id desc limit 1
            """,
            (int(group_id),),
        ).fetchone()
        if row is None:
            raise ValueError("这个群还没有可反馈的 RAG 命中")
        try:
            document_ids = [int(value) for value in json.loads(str(row["source_ids_json"]))]
        except (json.JSONDecodeError, TypeError, ValueError):
            document_ids = []
        if position < 1 or position > len(document_ids):
            raise ValueError(f"序号应在 1-{len(document_ids)} 之间")
        document_id = document_ids[position - 1]
        self.conn.execute(
            """
            insert into rag_retrieval_feedback(
              group_id, retrieval_event_id, document_id, label, note, operator_id, created_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (int(group_id), int(row["id"]), document_id, label, note[:300], int(operator_id), time.time()),
        )
        self.conn.commit()
        return document_id, label

    def recent_feedback(self, group_id: int, *, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            select f.*, d.doc_type, substr(d.content, 1, 100) as content_preview
            from rag_retrieval_feedback f
            join rag_documents d on d.id = f.document_id
            where f.group_id = ? order by f.id desc limit ?
            """,
            (int(group_id), max(1, min(50, int(limit)))),
        ).fetchall()

    def add_evaluation_case(
        self,
        *,
        group_id: int,
        query: str,
        expected_terms: list[str] | tuple[str, ...],
        expected_user_ids: list[int] | tuple[int, ...] = (),
        created_by: int = 0,
    ) -> int:
        clean_query = re.sub(r"\s+", " ", query).strip()
        terms = [str(value).strip()[:80] for value in expected_terms if str(value).strip()]
        users = sorted({int(value) for value in expected_user_ids if int(value) > 0})
        if len(clean_query) < 2 or not terms:
            raise ValueError("评测问题和期望关键词不能为空")
        now = time.time()
        self.conn.execute(
            """
            insert into rag_evaluation_cases(
              group_id, query, expected_terms_json, expected_user_ids_json,
              created_by, enabled, created_at, updated_at
            ) values (?, ?, ?, ?, ?, 1, ?, ?)
            on conflict(group_id, query) do update set
              expected_terms_json=excluded.expected_terms_json,
              expected_user_ids_json=excluded.expected_user_ids_json,
              created_by=excluded.created_by, enabled=1, updated_at=excluded.updated_at
            """,
            (int(group_id), clean_query[:300], json.dumps(terms, ensure_ascii=False), json.dumps(users), int(created_by), now, now),
        )
        self.conn.commit()
        row = self.conn.execute(
            "select id from rag_evaluation_cases where group_id = ? and query = ?",
            (int(group_id), clean_query[:300]),
        ).fetchone()
        return int(row["id"])

    def evaluation_cases(self, group_id: int, *, limit: int = 50) -> list[RAGEvaluationCase]:
        rows = self.conn.execute(
            """
            select * from rag_evaluation_cases
            where group_id = ? and enabled = 1 order by id asc limit ?
            """,
            (int(group_id), max(1, min(200, int(limit)))),
        ).fetchall()
        result: list[RAGEvaluationCase] = []
        for row in rows:
            try:
                terms = tuple(str(value) for value in json.loads(str(row["expected_terms_json"])))
                users = tuple(int(value) for value in json.loads(str(row["expected_user_ids_json"])))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            result.append(RAGEvaluationCase(int(row["id"]), int(row["group_id"]), str(row["query"]), terms, users, int(row["created_by"])))
        return result

    def delete_evaluation_case(self, group_id: int, case_id: int) -> bool:
        cursor = self.conn.execute(
            "update rag_evaluation_cases set enabled=0, updated_at=? where id=? and group_id=?",
            (time.time(), int(case_id), int(group_id)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def status_snapshot(self) -> dict[str, object]:
        counts = {
            str(row["doc_type"]): int(row["count"])
            for row in self.conn.execute(
                "select doc_type, count(*) as count from rag_documents group by doc_type"
            ).fetchall()
        }
        embedding = {
            str(row["embedding_status"]): int(row["count"])
            for row in self.conn.execute(
                "select embedding_status, count(*) as count from rag_documents group by embedding_status"
            ).fetchall()
        }
        recent = self.conn.execute(
            """
            select count(*) as count, avg(elapsed_ms) as avg_ms, max(created_at) as last_at
            from rag_retrieval_events where created_at >= ?
            """,
            (time.time() - 3600,),
        ).fetchone()
        feedback_count = int(
            self.conn.execute("select count(*) from rag_retrieval_feedback").fetchone()[0]
        )
        evaluation_count = int(
            self.conn.execute("select count(*) from rag_evaluation_cases where enabled = 1").fetchone()[0]
        )
        active_knowledge_sources = int(
            self.conn.execute(
                "select count(*) from rag_knowledge_sources where status='active'"
            ).fetchone()[0]
        )
        return {
            "fts5": self.fts_available,
            "documents": sum(counts.values()),
            "document_types": counts,
            "embedding_status": embedding,
            "retrievals_1h": int(recent["count"] or 0),
            "average_retrieval_ms_1h": round(float(recent["avg_ms"] or 0.0), 1),
            "last_retrieval_at": float(recent["last_at"] or 0.0) or None,
            "feedback_count": feedback_count,
            "evaluation_case_count": evaluation_count,
            "active_knowledge_sources": active_knowledge_sources,
        }

    def close(self) -> None:
        self.conn.close()


def _document_from_row(row: sqlite3.Row) -> RAGDocument:
    try:
        source_ids = tuple(str(value) for value in json.loads(str(row["source_message_ids_json"])))
    except (json.JSONDecodeError, TypeError):
        source_ids = ()
    try:
        participant_ids = tuple(
            int(value)
            for value in json.loads(str(row["participant_user_ids_json"] or "[]"))
            if int(value) > 0
        )
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        participant_ids = ()
    return RAGDocument(
        id=int(row["id"]),
        stable_key=str(row["stable_key"]),
        group_id=int(row["group_id"]),
        doc_type=str(row["doc_type"]),
        content=str(row["content"]),
        speaker_user_id=int(row["speaker_user_id"]) if row["speaker_user_id"] is not None else None,
        subject_user_id=int(row["subject_user_id"]) if row["subject_user_id"] is not None else None,
        source_name=str(row["source_name"]),
        source_row_id=str(row["source_row_id"]),
        source_message_ids=source_ids,
        participant_user_ids=participant_ids,
        asserted_by_user_id=int(row["asserted_by_user_id"]) if row["asserted_by_user_id"] is not None else None,
        created_at=float(row["created_at"]),
        valid_from=float(row["valid_from"]) if row["valid_from"] is not None else None,
        valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
        importance=float(row["importance"] or 0.0),
        confidence=float(row["confidence"] or 0.0),
        status=str(row["status"]),
        evidence_kind=str(row["evidence_kind"] or default_evidence_kind(str(row["doc_type"]))),
    )


def _knowledge_source_from_row(row: sqlite3.Row) -> RAGKnowledgeSource:
    return RAGKnowledgeSource(
        id=int(row["id"]),
        group_id=int(row["group_id"]),
        kind=str(row["kind"]),
        source_identity=str(row["source_identity"]),
        title=str(row["title"]),
        content_hash=str(row["content_hash"]),
        stable_prefix=str(row["stable_prefix"]),
        version=int(row["version"]),
        chunk_count=int(row["chunk_count"]),
        source_message_id=str(row["source_message_id"]),
        created_by=int(row["created_by"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _like_terms(query: str) -> list[str]:
    chunks = re.findall(r"[\u3400-\u9fff]{2,}|[A-Za-z0-9_+.-]{2,}", query)
    result: list[str] = []
    for chunk in chunks:
        clean = chunk.strip("你我他她它们的是了啊呀嘛吗呢吧哦请问一下记得之前以前上次刚才说过觉得")
        if len(clean) >= 2 and clean not in result:
            result.append(clean[:24])
    return result[:8]


def _fts_query(query: str) -> str:
    terms: list[str] = []
    for token in _like_terms(query):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) <= 6:
                candidates = [token]
            else:
                candidates = [token[index : index + 4] for index in range(0, len(token) - 3)]
        else:
            candidates = [token]
        for candidate in candidates:
            if len(candidate) >= 3 and candidate not in terms:
                terms.append(candidate)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:16])
