# -*- coding: utf-8 -*-
"""知识库工具层 — 供 MCP Server 与 Agent 进程内调用共享的纯逻辑。

约束设计：
- get_document 只允许读取"本会话中检索命中过"的 id（白名单），
  防止模型凭幻觉 id 拉取任意文档；
- 检索结果只回传答案摘要（AGENT_SNIPPET_CHARS 截断），
  完整原文必须显式调用 get_document —— 强制模型用步数换证据。
"""
from __future__ import annotations

from typing import Callable

from config import AGENT_SNIPPET_CHARS
from logger_config import logger


class KBSearchTools:
    """康语知识库工具集合（无状态协议 + 会话级白名单）。

    search_fn: (query: str, top_k: int) -> list[dict]
        命中项需含 id/question/answer/knowledge_point/similarity 字段，
        生产实现为混合检索 (语义+BM25→RRF→精排)，测试可注入 fake。
    library: {doc_id: {"question","answer","knowledge_point"}}
        全文库，get_document 用。
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], list[dict]],
        library: dict[str, dict],
        snippet_chars: int = AGENT_SNIPPET_CHARS,
    ):
        self._search_fn = search_fn
        self._library = library
        self._snippet_chars = snippet_chars
        self._exposed: set[str] = set()  # 本会话已暴露的 id（白名单）

    @property
    def exposed_ids(self) -> set[str]:
        return set(self._exposed)

    def reset_session(self):
        """清空检索白名单（评测逐题隔离用；交互式会话可不重置）"""
        self._exposed.clear()

    # ── 工具实现 ─────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: int = 5) -> dict:
        """混合检索：语义+BM25 双路召回 → RRF 融合 → 门控精排"""
        if not query or not query.strip():
            return {"error": "empty_query", "detail": "query 不能为空"}
        top_k = max(1, min(int(top_k or 5), 10))
        try:
            hits = self._search_fn(query.strip(), top_k)
        except Exception as e:  # 检索故障不抛裸异常，结构化降级
            logger.error(f"Agent 检索失败: {e}")
            return {"error": "search_failed", "detail": str(e)}

        results = []
        for h in hits:
            doc_id = str(h.get("id", ""))
            answer = h.get("answer", "") or ""
            results.append({
                "id": doc_id,
                "question": h.get("question", ""),
                "knowledge_point": h.get("knowledge_point", ""),
                "similarity": round(float(h.get("similarity", 0.0) or 0.0), 4),
                "snippet": answer[: self._snippet_chars],
            })
            if doc_id:
                self._exposed.add(doc_id)
        top_sim = max((r["similarity"] for r in results), default=0.0)
        return {"query": query, "results": results, "top_similarity": top_sim}

    def get_document(self, doc_id: str) -> dict:
        """读取文档全文；仅允许检索命中过的 id（白名单约束）"""
        doc_id = (doc_id or "").strip()
        if doc_id not in self._exposed:
            return {
                "error": "not_exposed",
                "detail": f"文档 {doc_id} 未在本会话检索结果中出现，拒绝读取；请先 hybrid_search",
            }
        doc = self._library.get(doc_id)
        if doc is None:
            return {"error": "not_found", "detail": f"知识库中不存在文档 {doc_id}"}
        return {
            "id": doc_id,
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "knowledge_point": doc.get("knowledge_point", ""),
        }

    def kb_stats(self) -> dict:
        """知识库元信息"""
        topics = {d.get("knowledge_point", "") for d in self._library.values()}
        return {"doc_count": len(self._library), "topic_count": len(topics - {""})}


def build_real_kb_tools():
    """组装生产工具：MacBERT 编码 + ChromaDB/BM25 混合检索（较重，懒加载）。

    返回 (KBSearchTools, HybridRetriever, Retriever)，后两者供评测/兜底管线复用。
    """
    import numpy as np

    from config import TOP_K_RETRIEVAL, init_env
    init_env()
    from rag_engine import Retriever
    from retriever.hybrid import HybridRetriever

    retriever = Retriever()
    retriever.load()
    qa = retriever.qa_library
    assert retriever.qa_embeddings is not None, "向量库缺失，请先运行 build_kb.py"

    hybrid = HybridRetriever()
    hybrid.build_index(qa, retriever.qa_embeddings)

    _cache: dict[str, np.ndarray] = {}

    def _encode(text: str) -> np.ndarray:
        key = text[:120]
        if key not in _cache:
            _cache[key] = retriever.encode(text)
            if len(_cache) > 128:
                _cache.clear()
        return _cache[key]

    def search_fn(query: str, top_k: int) -> list[dict]:
        return hybrid.search(query, _encode(query), top_k or TOP_K_RETRIEVAL)

    library = {
        str(i): {
            "question": qa["questions"][i],
            "answer": qa["answers"][i],
            "knowledge_point": qa["knowledge_points"][i],
        }
        for i in range(len(qa["questions"]))
    }
    tools = KBSearchTools(search_fn, library)
    logger.info(f"Agent 知识库工具就绪: {len(library)} 条")
    return tools, hybrid, retriever
