# -*- coding: utf-8 -*-
"""康语知识库 MCP Server (stdio)。

将混合检索 / 全文读取 / 知识库元信息封装为标准 MCP 工具，
可被本项目 Agent 客户端调用，也可被任意 MCP 客户端
(如 Claude Desktop / Cursor) 直接接入。

单独运行:  python agent/mcp_server.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer

from agent.tools import KBSearchTools
from logger_config import logger

SERVER_NAME = "kangyu-kb"


def create_server(tools: KBSearchTools) -> MCPServer:
    """由工具实例构建 MCP Server（依赖注入，便于测试）。"""
    server = MCPServer(
        name=SERVER_NAME,
        title="康语医疗健康知识库",
        description="混合检索(语义+BM25→RRF→精排)与全文读取工具集，读取受检索白名单约束。",
    )

    @server.tool()
    def hybrid_search(query: str, top_k: int = 5) -> dict:
        """混合检索健康问答知识库：MacBERT 语义 + BM25 双路召回，RRF 融合，门控精排。

        返回命中列表(id/question/knowledge_point/similarity/snippet 答案摘要)，
        需要完整原文时用返回的 id 调用 get_document。"""
        return tools.hybrid_search(query, top_k)

    @server.tool()
    def get_document(doc_id: str) -> dict:
        """读取知识库文档全文。只允许读取本会话中检索命中过的 id（白名单约束）。"""
        return tools.get_document(doc_id)

    @server.tool()
    def kb_stats() -> dict:
        """查询知识库规模：文档条数与主题数。"""
        return tools.kb_stats()

    @server.resource("kangyu://kb/stats")
    def read_kb_stats() -> str:
        """知识库元信息（MCP resource 形式）"""
        stats = tools.kb_stats()
        return f"康语知识库: {stats['doc_count']} 条问答 / {stats['topic_count']} 个主题"

    return server


def main():
    from agent.tools import build_real_kb_tools

    kb_tools, _, _ = build_real_kb_tools()
    server = create_server(kb_tools)
    logger.info("康语 MCP Server 启动 (stdio)")
    server.run(transport="stdio")  # 同步阻塞，内部自管事件循环


if __name__ == "__main__":
    main()
