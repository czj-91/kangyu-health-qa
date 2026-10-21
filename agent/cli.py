# -*- coding: utf-8 -*-
"""Agent 命令行入口。

用法:
  # 单次提问（stdio 传输 = 真实 MCP 子进程协议）
  python -m agent.cli -q "感冒了怎么办"

  # 进程内传输（跳过子进程，快速调试）
  python -m agent.cli -q "失眠怎么办" --transport inproc

  # 交互模式
  python -m agent.cli -i

需要支持 Function Calling 的 API 后端:
  export AGENT_API_BASE_URL=... AGENT_API_KEY=sk-... AGENT_API_MODEL=...
未单独设置时回落到 API_BASE_URL / API_KEY / API_MODEL。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from config import (
    AGENT_API_BASE_URL, AGENT_API_KEY, AGENT_API_MODEL,
    AGENT_MAX_STEPS, AGENT_TOKEN_BUDGET, AGENT_TRACE_DIR, init_env,
)

init_env()

from agent.loop import AgentRunner, mcp_tools_to_openai
from agent.mcp_server import SERVER_NAME, create_server
from logger_config import logger

SERVER_SCRIPT = str(ROOT / "agent" / "mcp_server.py")


class MCPBridge:
    """MCP Client → AgentRunner 的 tool_invoke 桥（返回值统一解析为 dict）"""

    def __init__(self, client: Client):
        self.client = client

    async def list_kb_schemas(self) -> list[dict]:
        tools = await self.client.list_tools()
        return mcp_tools_to_openai(tools.tools)

    async def invoke(self, name: str, args: dict) -> dict:
        res = await self.client.call_tool(name, args)
        text = ""
        for block in res.content or []:
            if getattr(block, "type", "") == "text":
                text = block.text
                break
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"raw": text}

    async def kb_stats(self) -> dict:
        return await self.invoke("kb_stats", {})


def _make_client(transport: str, kb_tools) -> Client:
    if transport == "inproc":
        return Client(create_server(kb_tools))
    return Client(StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT]))


async def ask_once(args_obj, question: str, client: Client, bridge: MCPBridge,
                   runner: AgentRunner, schemas: list[dict]):
    result = await runner.run_async(question, schemas)
    print("\n" + result.answer)
    print("-" * 60)
    flag = {"answer": "✅ Agent 回答", "clarification": "❓ 澄清反问",
            "degraded": f"⚠️ 已降级({result.degrade_reason})"}[result.final_type]
    print(f"{flag} | 步数 {result.steps} | 工具调用 {result.tool_calls} | "
          f"tokens {result.tokens_used} | {result.latency_ms:.0f}ms | "
          f"证据 {','.join(result.evidence_ids) or '-'}")
    if result.trace_path:
        print(f"轨迹: {result.trace_path}")


async def amain() -> int:
    parser = argparse.ArgumentParser(description="康语 Agentic RAG CLI")
    parser.add_argument("-q", "--question", help="单次提问")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("--transport", choices=["stdio", "inproc"], default="stdio",
                        help="stdio=独立 MCP 子进程(真实协议) / inproc=进程内 MCP Server")
    parser.add_argument("--max-steps", type=int, default=AGENT_MAX_STEPS)
    parser.add_argument("--token-budget", type=int, default=AGENT_TOKEN_BUDGET)
    parser.add_argument("--no-trace", action="store_true", help="不保存决策轨迹 JSONL")
    parser.add_argument("--no-pipeline-fallback", action="store_true",
                        help="降级时不回落到固定管线（用于调试真实失败）")
    parser.add_argument("--no-llm", action="store_true", help="只测 MCP 工具链路，不调 LLM")
    args = parser.parse_args()

    if not args.no_llm and not args.question and not args.interactive:
        parser.print_help()
        return 1

    from api_llm import APILLMEngine
    from agent.tools import build_real_kb_tools

    kb_tools, hybrid, retriever = build_real_kb_tools()

    if args.no_llm:
        print(f"MCP 链路自检: transport={args.transport}")
        async with _make_client(args.transport, kb_tools) as client:
            bridge = MCPBridge(client)
            stats = await bridge.kb_stats()
            schemas = await bridge.list_kb_schemas()
            print(f"  Server 工具: {[s['function']['name'] for s in schemas]}")
            print(f"  知识库: {stats}")
            r = await bridge.invoke("hybrid_search", {"query": "感冒了怎么办", "top_k": 3})
            print(f"  检索命中 {len(r.get('results', []))} 条, "
                  f"top_similarity={r.get('top_similarity')}")
        return 0

    llm = APILLMEngine(base_url=AGENT_API_BASE_URL, api_key=AGENT_API_KEY,
                       model=AGENT_API_MODEL)
    if not AGENT_API_KEY:
        print("错误: 未配置 AGENT_API_KEY / API_KEY — Agent 需要支持 Function Calling 的远程模型。\n"
              "       可先用 --no-llm 自检 MCP 工具链路。", file=sys.stderr)
        return 2

    fallback = None
    if not args.no_pipeline_fallback:
        try:
            from rag_engine import RAGEngine
            engine = RAGEngine(retriever, llm=llm)
            fallback = engine.ask
        except Exception as e:
            logger.warning(f"降级管线不可用: {e}")

    runner = AgentRunner(
        llm=llm, tool_invoke=None,  # 稍后按 client 绑定
        max_steps=args.max_steps, token_budget=args.token_budget,
        fallback=fallback, trace_dir=None if args.no_trace else AGENT_TRACE_DIR,
    )

    async with _make_client(args.transport, kb_tools) as client:
        bridge = MCPBridge(client)
        runner.tool_invoke = bridge.invoke
        schemas = await bridge.list_kb_schemas()
        logger.info(f"MCP Server [{SERVER_NAME}] 工具: "
                    f"{[s['function']['name'] for s in schemas]}")
        stats = await bridge.kb_stats()
        print(f"康语 Agent 就绪 — 知识库 {stats.get('doc_count')} 条 | "
              f"模型 {AGENT_API_MODEL} | 传输 {args.transport}")

        if args.question:
            await ask_once(args, args.question, client, bridge, runner, schemas)
            return 0

        while True:
            try:
                q = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q in ("exit", "quit"):
                break
            await ask_once(args, q, client, bridge, runner, schemas)
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
