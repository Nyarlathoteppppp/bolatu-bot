from __future__ import annotations

import asyncio
import time

from qq_social_agent.memory import MemoryStore
from qq_social_agent.rag_indexer import RAGIndexer
from qq_social_agent.rag_retriever import RAGService
from qq_social_agent.rag_router import plan_rag_query
from qq_social_agent.rag_store import RAGStore


def test_rag_router_only_uses_semantic_for_memory_like_queries() -> None:
    casual = plan_rag_query("今天吃鹅腿吗", addressed=True)
    memory = plan_rag_query("你还记得以前谁说过菲尔兹奖吗", addressed=True)

    assert casual.lexical is True
    assert casual.semantic is False
    assert memory.semantic is True
    assert memory.route == "explicit_memory"


def test_rag_indexer_preserves_speaker_time_and_source(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.add_message(
        1,
        10001,
        "代代",
        "今年北大出了两个菲尔兹奖",
        created_at=1000,
        source_message_id=71184,
    )
    memory.add_message(
        1,
        10002,
        "小鸟",
        "这个消息是哪来的",
        created_at=1001,
        source_message_id=71185,
    )
    memory.conn.close()

    store = RAGStore(db_path)
    stats = RAGIndexer(store).sync_all()
    rows = store.lexical_search(
        1,
        "以前谁说过菲尔兹奖",
        limit=5,
        doc_types=("conversation",),
    )

    assert stats["conversation"] == 1
    assert len(rows) == 1
    assert "代代[10001]" in rows[0].document.content
    assert rows[0].document.source_message_ids == ("71184", "71185")
    store.close()


def test_rag_does_not_return_disputed_memory_atom(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.upsert_memory_atom(
        atom_type="fact",
        group_id=1,
        content="王虹获得了菲尔兹奖",
        source="message:11",
        source_message_id=11,
        evidence_type="message",
        confidence=0.5,
        status="disputed",
    )
    memory.conn.close()

    store = RAGStore(db_path)
    RAGIndexer(store).sync_all()
    rows = store.lexical_search(
        1,
        "王虹菲尔兹奖",
        limit=5,
        doc_types=("memory_atom",),
    )

    assert rows == []
    store.close()


def test_rag_store_refreshes_embedding_when_content_changes(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    store = RAGStore(db_path)
    document_id = store.upsert_document(
        stable_key="manual:1",
        group_id=1,
        doc_type="summary",
        content="旧内容示例",
        source_name="manual",
        source_row_id=1,
    )
    document = store.pending_embedding_documents(limit=1)[0]
    store.save_embeddings([document], [[1.0, 0.0]], "test-model")
    assert store.status_snapshot()["embedding_status"]["ready"] == 1

    changed_id = store.upsert_document(
        stable_key="manual:1",
        group_id=1,
        doc_type="summary",
        content="新内容示例",
        source_name="manual",
        source_row_id=1,
    )
    store.commit()

    assert changed_id == document_id
    assert store.pending_embedding_documents(limit=1)[0].content == "新内容示例"
    store.close()


def test_rag_service_lexical_fallback_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_TEST_KEY", raising=False)
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.add_message(
        1,
        10001,
        "代代",
        "去年群里认真聊过三维挂谷猜想",
        created_at=time.time() - 3600,
        source_message_id=99,
    )
    memory.conn.close()
    service = RAGService(
        db_path,
        {
            "enabled": True,
            "mode": "hybrid",
            "embedding": {"enabled": True, "api_key_env": "RAG_TEST_KEY"},
            "retrieval": {"exclude_recent_seconds": 0},
        },
    )

    result = asyncio.run(
        service.retrieve(
            group_id=1,
            query="以前谁聊过三维挂谷猜想",
            addressed=True,
        )
    )

    assert result.lexical_count >= 1
    assert result.semantic_count == 0
    assert "代代" in result.context
    asyncio.run(service.close())
