from __future__ import annotations

import re
from dataclasses import dataclass

from .rag_retriever import RAGService


RAG_STATUS_COMMANDS = {"RAG状态", "rag状态", "RAG status", "/rag status"}
RAG_TEST_RE = re.compile(r"^(?:/)?(?:RAG测试|rag测试|RAG test|rag test)\s*[:：]?\s*(?P<query>.+)$", re.IGNORECASE)
RAG_FEEDBACK_RE = re.compile(
    r"^(?:/)?RAG反馈\s+(?P<label>相关|好|不相关|错人|人物错位|过期)\s+(?P<position>\d+)\s*(?:[:：]\s*(?P<note>.*))?$",
    re.IGNORECASE,
)
RAG_FEEDBACK_LIST_COMMANDS = {"RAG反馈列表", "rag反馈列表"}
RAG_EVAL_RUN_COMMANDS = {"RAG评测", "rag评测"}
RAG_EVAL_LIST_COMMANDS = {"RAG评测列表", "rag评测列表"}
RAG_EVAL_ADD_RE = re.compile(r"^(?:/)?RAG评测添加\s+(?P<body>.+)$", re.IGNORECASE | re.DOTALL)
RAG_EVAL_DELETE_RE = re.compile(r"^(?:/)?RAG评测删除\s+(?P<case_id>\d+)$", re.IGNORECASE)
RAG_KNOWLEDGE_LIST_COMMANDS = {"RAG知识库", "rag知识库", "RAG知识库列表", "rag知识库列表"}
RAG_KNOWLEDGE_DELETE_RE = re.compile(r"^(?:/)?RAG知识库删除\s+(?P<source_id>\d+)$", re.IGNORECASE)
RAG_KNOWLEDGE_REINDEX_RE = re.compile(r"^(?:/)?RAG知识库(?:重建|重索引)\s+(?P<source_id>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RAGAdminResult:
    handled: bool
    text: str = ""


class RAGAdminController:
    def __init__(self, service: RAGService):
        self.service = service

    def matches(self, text: str) -> bool:
        return bool(
            text in RAG_STATUS_COMMANDS
            or RAG_TEST_RE.match(text)
            or RAG_FEEDBACK_RE.match(text)
            or text in RAG_FEEDBACK_LIST_COMMANDS
            or text in RAG_EVAL_RUN_COMMANDS
            or text in RAG_EVAL_LIST_COMMANDS
            or RAG_EVAL_ADD_RE.match(text)
            or RAG_EVAL_DELETE_RE.match(text)
            or text in RAG_KNOWLEDGE_LIST_COMMANDS
            or RAG_KNOWLEDGE_DELETE_RE.match(text)
            or RAG_KNOWLEDGE_REINDEX_RE.match(text)
        )

    async def handle(self, text: str, *, group_id: int | None, operator_id: int) -> RAGAdminResult:
        if not self.matches(text):
            return RAGAdminResult(False)
        if text in RAG_STATUS_COMMANDS:
            return RAGAdminResult(True, self.status_report())
        if group_id is None:
            return RAGAdminResult(True, "没有可用目标群。")
        test_match = RAG_TEST_RE.match(text)
        if test_match is not None:
            return RAGAdminResult(True, await self.service.diagnostic_query(group_id, test_match.group("query").strip()))
        feedback_match = RAG_FEEDBACK_RE.match(text)
        if feedback_match is not None:
            labels = {
                "相关": "relevant", "好": "relevant", "不相关": "irrelevant",
                "错人": "wrong_person", "人物错位": "wrong_person", "过期": "stale",
            }
            try:
                document_id, label = self.service.add_feedback(
                    group_id=group_id,
                    position=int(feedback_match.group("position")),
                    label=labels[feedback_match.group("label")],
                    operator_id=operator_id,
                    note=(feedback_match.group("note") or "").strip(),
                )
                return RAGAdminResult(True, f"已记录 RAG 反馈：文档 {document_id} / {label}，后续统一重排会自动采用。")
            except ValueError as exc:
                return RAGAdminResult(True, f"RAG反馈失败：{exc}")
        if text in RAG_FEEDBACK_LIST_COMMANDS:
            return RAGAdminResult(True, self.service.feedback_report(group_id))
        if text in RAG_KNOWLEDGE_LIST_COMMANDS:
            return RAGAdminResult(True, self.service.knowledge_source_report(group_id))
        delete_match = RAG_KNOWLEDGE_DELETE_RE.match(text)
        if delete_match is not None:
            source_id = int(delete_match.group("source_id"))
            deleted = self.service.delete_knowledge_source(group_id, source_id)
            return RAGAdminResult(
                True,
                f"已删除知识库来源 #{source_id}，旧版本保留审计但不再检索。"
                if deleted else "没有找到可删除的知识库来源。",
            )
        reindex_match = RAG_KNOWLEDGE_REINDEX_RE.match(text)
        if reindex_match is not None:
            source_id = int(reindex_match.group("source_id"))
            count = self.service.reindex_knowledge_source(group_id, source_id)
            return RAGAdminResult(
                True,
                f"已将知识库来源 #{source_id} 的 {count} 个分块加入后台重索引。"
                if count else "没有找到可重建的知识库来源。",
            )
        if text in RAG_EVAL_LIST_COMMANDS:
            return RAGAdminResult(True, self.service.evaluation_case_report(group_id))
        add_match = RAG_EVAL_ADD_RE.match(text)
        if add_match is not None:
            parts = [part.strip() for part in re.split(r"\s*[|｜]\s*", add_match.group("body"))]
            if len(parts) < 2:
                return RAGAdminResult(True, "格式：RAG评测添加 问题 | 关键词1,关键词2 | QQ1,QQ2（QQ 可留空）")
            terms = [value.strip() for value in re.split(r"[,，]", parts[1]) if value.strip()]
            users = [int(value) for value in re.findall(r"\d{5,12}", parts[2])] if len(parts) >= 3 else []
            try:
                case_id = self.service.add_evaluation_case(
                    group_id=group_id,
                    query=parts[0],
                    expected_terms=terms,
                    expected_user_ids=users,
                    created_by=operator_id,
                )
                return RAGAdminResult(True, f"已保存 RAG 评测用例 #{case_id}。")
            except ValueError as exc:
                return RAGAdminResult(True, f"添加失败：{exc}")
        delete_eval_match = RAG_EVAL_DELETE_RE.match(text)
        if delete_eval_match is not None:
            deleted = self.service.store.delete_evaluation_case(group_id, int(delete_eval_match.group("case_id")))
            return RAGAdminResult(True, "已删除该评测用例。" if deleted else "没有找到该评测用例。")
        if text in RAG_EVAL_RUN_COMMANDS:
            return RAGAdminResult(True, await self.service.run_evaluation(group_id))
        return RAGAdminResult(False)

    def status_report(self) -> str:
        snapshot = self.service.status_snapshot()
        store = snapshot.get("store", {}) if isinstance(snapshot.get("store"), dict) else {}
        embedding = snapshot.get("embedding", {}) if isinstance(snapshot.get("embedding"), dict) else {}
        type_counts = store.get("document_types", {}) if isinstance(store.get("document_types"), dict) else {}
        status_counts = store.get("embedding_status", {}) if isinstance(store.get("embedding_status"), dict) else {}
        types_text = "、".join(f"{name}={count}" for name, count in sorted(type_counts.items())) or "无"
        embeddings_text = "、".join(f"{name}={count}" for name, count in sorted(status_counts.items())) or "无"
        return (
            "轻量混合 RAG 状态：\n"
            f"- enabled={snapshot.get('enabled')} mode={snapshot.get('mode')} FTS5={store.get('fts5')}\n"
            f"- 文档总数={store.get('documents', 0)}（{types_text}）\n"
            f"- 向量={embeddings_text}\n"
            f"- embedding={embedding.get('model')} available={embedding.get('available')} "
            f"calls={embedding.get('calls', 0)} failures={embedding.get('failures', 0)}\n"
            f"- 最近1小时检索={store.get('retrievals_1h', 0)}次，平均={store.get('average_retrieval_ms_1h', 0)}ms\n"
            f"- 人工反馈={store.get('feedback_count', 0)}条，评测用例={store.get('evaluation_case_count', 0)}条\n"
            f"- 知识库来源={store.get('active_knowledge_sources', 0)}个（命令：RAG知识库）\n"
            f"- 后台索引={snapshot.get('background_indexing')} query缓存={snapshot.get('query_cache_entries', 0)}\n"
            f"- 最近错误={snapshot.get('last_error') or embedding.get('last_error') or '无'}\n"
            "测试命令：RAG测试 以前谁聊过菲尔兹奖"
        )
