# -*- coding: utf-8 -*-
"""Agent 知识库工具层 + MCP Server 进程内往返测试（轻量，无需模型/网络）"""
import asyncio
import json

import pytest

from agent.mcp_server import create_server
from agent.tools import KBSearchTools


def make_library():
    return {
        "1": {"question": "感冒了怎么办", "answer": "多休息多喝水*" + "答" * 300, "knowledge_point": "呼吸科"},
        "2": {"question": "发烧怎么处理", "answer": "物理降温", "knowledge_point": "内科"},
        "3": {"question": "失眠怎么办", "answer": "规律作息", "knowledge_point": "神经科"},
    }


def make_tools(search_fn=None, **kw):
    lib = make_library()
    if search_fn is None:
        def default_search(query, top_k):
            hits = [{"id": k, **v} for k, v in lib.items() if any(c in v["question"] for c in query)]
            return sorted(hits, key=lambda d: -len(d["question"]))[:top_k]
        search_fn = default_search
    return KBSearchTools(search_fn, lib, **kw)


class TestKBSearchTools:
    def test_search_returns_snippet_not_full_answer(self):
        tools = make_tools(snippet_chars=20)
        out = tools.hybrid_search("感冒", 5)
        assert out["results"], "应有命中"
        doc = out["results"][0]
        assert len(doc["snippet"]) <= 20
        assert "answer" not in doc  # 全文必须走 get_document

    def test_search_exposes_ids_for_fetch(self):
        tools = make_tools()
        out = tools.hybrid_search("感冒", 5)
        doc_id = out["results"][0]["id"]
        full = tools.get_document(doc_id)
        assert "error" not in full
        assert len(full["answer"]) > 100  # 拿到的是完整原文

    def test_get_document_blocks_unexposed_id(self):
        tools = make_tools()
        out = tools.get_document("2")
        assert out["error"] == "not_exposed"
        assert "2" not in tools.exposed_ids  # 拒绝不产生暴露副作用

    def test_whitelist_survives_after_exposure(self):
        tools = make_tools()
        tools.hybrid_search("失眠", 5)
        assert "3" in tools.exposed_ids
        assert tools.get_document("999")["error"] == "not_exposed"

    def test_empty_query_rejected(self):
        out = make_tools().hybrid_search("   ", 5)
        assert out["error"] == "empty_query"

    def test_top_k_clamped(self):
        calls = []
        tools = make_tools(search_fn=lambda q, k: calls.append(k) or [])
        tools.hybrid_search("感冒", 99)
        assert calls[-1] == 10
        tools.hybrid_search("感冒", 0)
        assert calls[-1] == 5

    def test_search_failure_returns_structured_error(self):
        def boom(query, top_k):
            raise RuntimeError("chroma down")
        out = make_tools(search_fn=boom).hybrid_search("感冒", 5)
        assert out["error"] == "search_failed"
        assert "chroma down" in out["detail"]

    def test_top_similarity_and_kb_stats(self):
        tools = make_tools(search_fn=lambda q, k: [
            {"id": "1", "question": "q1", "answer": "a1", "knowledge_point": "t1", "similarity": 0.87},
        ])
        out = tools.hybrid_search("q", 5)
        assert out["top_similarity"] == 0.87
        stats = tools.kb_stats()
        assert stats["doc_count"] == 3


class TestMCPServerRoundtrip:
    """进程内 MCP 客户端/服务端真实协议往返"""

    def _run(self, fn):
        from mcp.client import Client
        server = create_server(make_tools())

        async def main():
            async with Client(server) as client:
                return await fn(client)
        return asyncio.run(main())

    def test_list_tools_exposes_kb_tools(self):
        async def check(client):
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"hybrid_search", "get_document", "kb_stats"} <= names
            search = next(t for t in tools.tools if t.name == "hybrid_search")
            assert "query" in search.input_schema["properties"]
            assert search.description  # docstring 进了 schema
        self._run(check)

    def test_call_hybrid_search_json_roundtrip(self):
        async def check(client):
            res = await client.call_tool("hybrid_search", {"query": "感冒", "top_k": 3})
            assert not res.is_error
            payload = json.loads(res.content[0].text)
            assert payload["results"][0]["snippet"]
        self._run(check)

    def test_whitelist_enforced_over_protocol(self):
        async def check(client):
            res = await client.call_tool("get_document", {"doc_id": "2"})
            payload = json.loads(res.content[0].text)
            assert payload["error"] == "not_exposed"
            # 检索后再取全文 — 同一 server 实例共享白名单状态
            await client.call_tool("hybrid_search", {"query": "发烧", "top_k": 3})
            res2 = await client.call_tool("get_document", {"doc_id": "2"})
            payload2 = json.loads(res2.content[0].text)
            assert payload2.get("answer") == "物理降温"
        self._run(check)

    def test_kb_stats_resource_registered(self):
        async def check(client):
            resources = await client.list_resources()
            uris = {str(r.uri) for r in resources.resources}
            assert "kangyu://kb/stats" in uris
            content = await client.read_resource("kangyu://kb/stats")
            text = content.contents[0].text if hasattr(content.contents[0], "text") else str(content.contents[0])
            assert "康语知识库" in text and "3 条" in text
        self._run(check)
