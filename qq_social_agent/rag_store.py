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
    created_at: float
    importance: float
    confidence: float
    status: str


@dataclass(frozen=True)
class RankedDocument:
    document: RAGDocument
    score: float


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
              asserted_by_user_id integer,
              created_at real not null,
              valid_from real,
              valid_to real,
              importance real not null default 0.5,
              confidence real not null default 0.6,
              status text not null default 'active',
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
              error text not null default '',
              created_at real not null
            );
            create index if not exists idx_rag_retrieval_events_time
              on rag_retrieval_events(created_at desc);
            """
        )
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
        speaker_user_id: int | None = None,
        subject_user_id: int | None = None,
        asserted_by_user_id: int | None = None,
        created_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        importance: float = 0.5,
        confidence: float = 0.6,
        status: str = "active",
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
              source_name, source_row_id, source_message_ids_json, asserted_by_user_id,
              created_at, valid_from, valid_to, importance, confidence, status,
              content_hash, embedding_status, indexed_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            on conflict(stable_key) do update set
              group_id=excluded.group_id,
              doc_type=excluded.doc_type,
              content=excluded.content,
              speaker_user_id=excluded.speaker_user_id,
              subject_user_id=excluded.subject_user_id,
              source_name=excluded.source_name,
              source_row_id=excluded.source_row_id,
              source_message_ids_json=excluded.source_message_ids_json,
              asserted_by_user_id=excluded.asserted_by_user_id,
              created_at=excluded.created_at,
              valid_from=excluded.valid_from,
              valid_to=excluded.valid_to,
              importance=excluded.importance,
              confidence=excluded.confidence,
              status=excluded.status,
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
                asserted_by_user_id,
                float(created_at or now),
                valid_from,
                valid_to,
                max(0.0, min(1.0, float(importance))),
                max(0.0, min(1.0, float(confidence))),
                str(status)[:24] or "active",
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

    def record_retrieval(
        self,
        *,
        group_id: int,
        query: str,
        route: str,
        lexical_count: int,
        semantic_count: int,
        injected_ids: list[int],
        elapsed_ms: int,
        cache_hit: bool,
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            insert into rag_retrieval_events(
              group_id, query_hash, route, lexical_count, semantic_count,
              injected_count, elapsed_ms, cache_hit, source_ids_json, error, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                error[:240],
                time.time(),
            ),
        )
        self.conn.commit()

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
        return {
            "fts5": self.fts_available,
            "documents": sum(counts.values()),
            "document_types": counts,
            "embedding_status": embedding,
            "retrievals_1h": int(recent["count"] or 0),
            "average_retrieval_ms_1h": round(float(recent["avg_ms"] or 0.0), 1),
            "last_retrieval_at": float(recent["last_at"] or 0.0) or None,
        }

    def close(self) -> None:
        self.conn.close()


def _document_from_row(row: sqlite3.Row) -> RAGDocument:
    try:
        source_ids = tuple(str(value) for value in json.loads(str(row["source_message_ids_json"])))
    except (json.JSONDecodeError, TypeError):
        source_ids = ()
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
        created_at=float(row["created_at"]),
        importance=float(row["importance"] or 0.0),
        confidence=float(row["confidence"] or 0.0),
        status=str(row["status"]),
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
