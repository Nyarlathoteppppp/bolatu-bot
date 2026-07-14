from __future__ import annotations

import asyncio

from qq_social_agent.memory import MemoryStore
from qq_social_agent.rag_admin import RAGAdminController
from qq_social_agent.rag_retriever import RAGService


def _controller(tmp_path) -> tuple[RAGAdminController, RAGService]:
    db_path = tmp_path / "bot.sqlite3"
    memory = MemoryStore(db_path)
    memory.conn.close()
    service = RAGService(db_path, {"enabled": True, "embedding": {"enabled": False}})
    return RAGAdminController(service), service


def test_rag_admin_recognizes_commands_without_claiming_normal_private_text(tmp_path) -> None:
    controller, service = _controller(tmp_path)

    assert controller.matches("RAG状态")
    assert controller.matches("RAG知识库删除 3")
    assert not controller.matches("今天吃什么")
    asyncio.run(service.close())


def test_rag_admin_lists_and_soft_deletes_knowledge_source(tmp_path) -> None:
    controller, service = _controller(tmp_path)
    service.ingest_knowledge(
        group_id=1,
        kind="web",
        source_identity="https://example.com/a",
        title="测试网页",
        content="网页知识内容" * 100,
    )
    source_id = service.store.knowledge_sources(1)[0].id

    listing = asyncio.run(controller.handle("RAG知识库", group_id=1, operator_id=9))
    deletion = asyncio.run(controller.handle(f"RAG知识库删除 {source_id}", group_id=1, operator_id=9))

    assert listing.handled and "测试网页" in listing.text
    assert deletion.handled and "已删除" in deletion.text
    asyncio.run(service.close())
