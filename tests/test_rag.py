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


def test_person_past_resolves_alias_and_unifies_structured_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_TEST_KEY", raising=False)
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.add_message(1, 184589072, "小鸟", "以前小鸟最爱讨论算法岗", created_at=1000)
    memory.upsert_memory_atom(
        atom_type="preference",
        group_id=1,
        subject_user_id=184589072,
        content="小鸟过去偏爱算法岗话题",
        source="message:1",
        confidence=0.8,
        status="active",
    )
    memory.conn.close()
    service = RAGService(
        db_path,
        {
            "enabled": True,
            "member_aliases": {"184589072": ["小鸟"]},
            "embedding": {"enabled": False},
            "retrieval": {"exclude_recent_seconds": 0},
        },
    )

    result = asyncio.run(
        service.retrieve(group_id=1, query="小鸟以前喜欢聊什么", addressed=True)
    )

    assert result.plan.route == "person_past"
    assert result.resolved_members[0].user_id == 184589072
    assert any(hit.document.doc_type == "memory_atom" for hit in result.hits)
    assert any("人物匹配" in hit.reasons for hit in result.hits)
    asyncio.run(service.close())


def test_retrieval_feedback_changes_unified_ranking(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    service = RAGService(
        db_path,
        {"enabled": True, "embedding": {"enabled": False}, "retrieval": {"exclude_recent_seconds": 0}},
    )
    first_id = service.store.upsert_document(
        stable_key="manual:first", group_id=1, doc_type="summary", content="菲尔兹奖讨论版本甲",
        source_name="manual", source_row_id="first",
    )
    service.store.upsert_document(
        stable_key="manual:second", group_id=1, doc_type="summary", content="菲尔兹奖讨论版本乙",
        source_name="manual", source_row_id="second",
    )
    service.store.commit()

    initial = asyncio.run(service.retrieve(group_id=1, query="菲尔兹奖讨论", addressed=True))
    initially_first_id = initial.hits[0].document.id
    service.add_feedback(group_id=1, position=1, label="irrelevant", operator_id=9)
    reranked = asyncio.run(service.retrieve(group_id=1, query="菲尔兹奖讨论", addressed=True))

    assert reranked.hits[0].document.id != initially_first_id
    assert any("反馈降权" in hit.reasons for hit in reranked.hits if hit.document.id == initially_first_id)
    asyncio.run(service.close())


def test_file_knowledge_is_chunked_and_retrievable(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    service = RAGService(
        db_path,
        {"enabled": True, "embedding": {"enabled": False}, "retrieval": {"exclude_recent_seconds": 0}},
    )
    count = service.ingest_knowledge(
        group_id=1,
        kind="file",
        source_identity="file-123",
        title="算法资料.txt",
        content="这份文件专门解释三维挂谷猜想。" * 80,
        source_message_id="88",
    )
    result = asyncio.run(
        service.retrieve(group_id=1, query="那个文件里的三维挂谷猜想资料", addressed=True)
    )

    assert count >= 2
    assert result.plan.route == "knowledge"
    assert any(hit.document.doc_type == "file_knowledge" for hit in result.hits)
    assert "文件知识库" in result.context
    asyncio.run(service.close())
