from __future__ import annotations

import asyncio
import time

from qq_social_agent.memory import MemoryStore
from qq_social_agent.rag_indexer import RAGIndexer
from qq_social_agent.rag_retriever import RAGService
from qq_social_agent.rag_router import plan_rag_query
from qq_social_agent.rag_store import RAGStore
from qq_social_agent.temporal_evidence import (
    TemporalIntent,
    detect_temporal_intent,
    recency_adjustment,
    statements_conflict,
)


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


def test_temporal_evidence_prefers_recent_for_current_questions() -> None:
    now = time.time()

    assert detect_temporal_intent("小鸟现在还准备考研吗") is TemporalIntent.CURRENT
    assert detect_temporal_intent("小鸟以前准备考研吗") is TemporalIntent.HISTORICAL
    assert recency_adjustment(now - 3600, TemporalIntent.CURRENT, now=now) > recency_adjustment(
        now - 400 * 86400,
        TemporalIntent.CURRENT,
        now=now,
    )
    assert statements_conflict("小鸟已经放弃考研", "小鸟准备考研", ["考研"])


def test_rag_context_labels_reported_claim_as_non_objective(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.add_message(1, 10, "甲", "甲说自己准备考研", created_at=time.time() - 3600)
    memory.conn.close()
    service = RAGService(
        db_path,
        {"enabled": True, "embedding": {"enabled": False}, "retrieval": {"exclude_recent_seconds": 0}},
    )

    result = asyncio.run(service.retrieve(group_id=1, query="准备考研", addressed=True))

    assert "性质=说话者当时陈述" in result.context
    asyncio.run(service.close())


def test_current_question_downgrades_explicitly_contradicted_old_claim(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    service = RAGService(
        db_path,
        {"enabled": True, "embedding": {"enabled": False}, "retrieval": {"exclude_recent_seconds": 0}},
    )
    now = time.time()
    old_id = service.store.upsert_document(
        stable_key="claim:old",
        group_id=1,
        doc_type="conversation",
        content="小鸟准备考研",
        source_name="messages",
        source_row_id="1",
        speaker_user_id=7,
        participant_user_ids=(7,),
        created_at=now - 400 * 86400,
        evidence_kind="reported_claim",
    )
    new_id = service.store.upsert_document(
        stable_key="claim:new",
        group_id=1,
        doc_type="conversation",
        content="小鸟已经放弃考研",
        source_name="messages",
        source_row_id="2",
        speaker_user_id=7,
        participant_user_ids=(7,),
        created_at=now - 86400,
        evidence_kind="reported_claim",
    )
    service.store.commit()

    result = asyncio.run(service.retrieve(group_id=1, query="考研 现在", addressed=True))

    assert result.hits[0].document.id == new_id
    old_hit = next(hit for hit in result.hits if hit.document.id == old_id)
    assert "较新反证降权" in old_hit.reasons
    asyncio.run(service.close())


def test_knowledge_source_versions_deduplicate_and_soft_delete(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    service = RAGService(db_path, {"enabled": True, "embedding": {"enabled": False}})

    first_count = service.ingest_knowledge(
        group_id=1,
        kind="file",
        source_identity="file-a",
        title="资料.txt",
        content="第一版资料里说明了三维挂谷猜想。" * 50,
        created_by=9,
    )
    duplicate_count = service.ingest_knowledge(
        group_id=1,
        kind="file",
        source_identity="file-copy",
        title="资料副本.txt",
        content="第一版资料里说明了三维挂谷猜想。" * 50,
        created_by=9,
    )
    sources = service.store.knowledge_sources(1)

    assert duplicate_count == first_count
    assert len(sources) == 1
    assert sources[0].version == 1

    service.ingest_knowledge(
        group_id=1,
        kind="file",
        source_identity="file-a",
        title="资料.txt",
        content="第二版资料修正了三维挂谷猜想的说明。" * 50,
        created_by=9,
    )
    source = service.store.knowledge_sources(1)[0]
    assert source.version == 2
    assert service.reindex_knowledge_source(1, source.id) == source.chunk_count
    assert service.delete_knowledge_source(1, source.id)
    assert service.store.knowledge_sources(1) == []
    inactive = service.store.conn.execute(
        "select count(*) from rag_documents where stable_key like ? and status='inactive'",
        (f"{source.stable_prefix}%",),
    ).fetchone()[0]
    assert inactive >= first_count
    asyncio.run(service.close())
