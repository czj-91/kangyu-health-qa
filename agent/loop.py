# -*- coding: utf-8 -*-
"""康语 Agentic RAG — Function Calling 决策循环。

与固定管线的区别：检索次数、查询改写、是否读全文由 LLM 自主决定，
循环层负责"确定性"：
- 步数 / token / 墙钟三重预算，超限先强制终止提醒，仍不终止则降级；
- finalize_answer 证据归属校验：引用的 evidence_ids 必须真实检索命中；
- 任何失败路径（LLM 异常、预算耗尽、反复拒绝）统一降级到既有管线
  (RAGEngine.ask → 检索原文/兜底话术)，用户永远得到一个答案。

传输层无关：工具调用经由 `tool_invoke(name, args)` 协程注入，
生产路径是 MCP Client(stdio/in-process)，测试注入 fake。
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from agent.contracts import AgentResult, ClarifyPayload, FinalAnswerPayload
from config import (
    AGENT_MAX_STEPS, AGENT_MAX_TOKENS, AGENT_MAX_TOOL_RESULT_CHARS,
    AGENT_TOKEN_BUDGET, AGENT_TRACE_DIR, AGENT_WALL_TIMEOUT,
)
from logger_config import logger

ToolInvoke = Callable[[str, dict], Awaitable[Any]]

# ── 终止控制工具（不进 MCP，由循环层消化）──────────────────

CONTROL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "给出最终回答并终止本轮。answer 为回答正文；evidence_ids 必须是本会话检索命中过的文档 id（1-5 个）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "面向用户的最终回答正文（中文）"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "引用的知识库文档 id 列表",
                    },
                },
                "required": ["answer", "evidence_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "信息不足时向用户反问以补全问题后终止本轮（不会给出健康建议，仅请求澄清）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_to_user": {"type": "string", "description": "向用户提出的澄清问题"},
                    "reason": {"type": "string", "description": "为何需要澄清（内部记录）"},
                },
                "required": ["question_to_user"],
            },
        },
    },
]

FORCED_NOTE = (
    "预算已用尽。现在只能调用 finalize_answer 或 ask_clarification 结束本轮，"
    "不得再调用检索/读取工具。请基于已获得的信息给出最终回答（或澄清反问）。"
)

NO_TOOL_CALL_NOTE = (
    "请不要直接输出正文。你必须通过 finalize_answer 提交带证据引用的回答，"
    "或 ask_clarification 反问用户。"
)


def build_system_prompt(max_steps: int, token_budget: int) -> str:
    return f"""你是「康语」医疗健康知识问答 Agent，基于知识库检索回答用户健康咨询。

工作方式（工具决策循环）：
1. 先调用 hybrid_search 检索知识库；结果只含答案摘要，需要完整原文时用命中 id 调用 get_document。
2. 检索置信度偏低或问题含糊/截断时，可换一种说法重新检索；多次尝试仍不确定时，调用 ask_clarification 向用户反问。
3. 回答必须通过 finalize_answer 终止，evidence_ids 引用本会话工具真实返回过的文档 id；凭空编造的 id 会被拒绝并要求修正。
4. 最多 {max_steps} 个决策步、token 预算约 {token_budget}；预算将尽会收到强制终止提醒，届时只能终止。

内容安全（必须遵守）：
- 只做健康科普：不诊断、不指定用药剂量、不给个体化治疗方案；涉及具体病情时建议到正规医疗机构就诊。
- 用户描述的当前症状若属剧烈胸痛、呼吸困难、意识不清、抽搐、大出血、疑似中风/心梗等急症，第一句建议立即拨打 120 或前往急诊；一般知识性提问不要提急症提醒。
- 不编造数据；回答用中文，通俗、有条理，直接以回答正文开头。"""


def mcp_tools_to_openai(tools: list[Any]) -> list[dict]:
    """MCP Tool 列表 → OpenAI function-calling schema 列表（只保留 JSON Schema 核心字段）"""
    schemas = []
    for t in tools:
        raw = dict(t.input_schema or {})
        params = {
            "type": raw.get("type", "object"),
            "properties": raw.get("properties", {}),
        }
        if raw.get("required"):
            params["required"] = raw["required"]
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": params,
            },
        })
    return schemas


def _truncate_json(obj: Any, limit: int) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) > limit:
        text = json.dumps(
            {"_truncated": text[:limit], "_note": "结果过长已截断，可用 get_document 读取指定文档全文"},
            ensure_ascii=False,
        )
    return text


class AgentRunner:
    """Function Calling 决策循环（传输层无关，同步入口 run()）。"""

    def __init__(
        self,
        llm: Any,
        tool_invoke: ToolInvoke,
        *,
        max_steps: int = AGENT_MAX_STEPS,
        token_budget: int = AGENT_TOKEN_BUDGET,
        wall_timeout: int = AGENT_WALL_TIMEOUT,
        fallback: Callable[[str], dict] | None = None,
        trace_dir: str | Path | None = AGENT_TRACE_DIR,
    ):
        self.llm = llm
        self.tool_invoke = tool_invoke
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.wall_timeout = wall_timeout
        self.fallback = fallback
        self.trace_dir = Path(trace_dir) if trace_dir else None

    # ── 公开入口 ────────────────────────────────────────

    def run(self, question: str, kb_schemas: list[dict]) -> AgentResult:
        return asyncio.run(self.run_async(question, kb_schemas))

    async def run_async(self, question: str, kb_schemas: list[dict]) -> AgentResult:
        t0 = time.monotonic()
        session = _Session(question, build_system_prompt(self.max_steps, self.token_budget))
        trace: list[dict] = []
        terminal: AgentResult | None = None

        try:
            schemas = list(kb_schemas) + CONTROL_TOOL_SCHEMAS
            terminal = await self._loop(question, schemas, session, trace)
        except Exception as e:  # 循环层任何未预期异常 → 降级，绝不裸抛
            logger.error(f"Agent 循环异常: {e}")
            trace.append({"type": "crash", "detail": str(e)})
            terminal = self._degrade(question, session, f"crash:{e}")

        if terminal is None:  # 理论不可达，防御
            terminal = self._degrade(question, session, "unknown")

        terminal.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        trace.append({"type": "final", "final_type": terminal.final_type,
                      "degrade_reason": terminal.degrade_reason,
                      "steps": terminal.steps, "tokens": terminal.tokens_used})
        terminal.trace_path = self._save_trace(question, trace, t0)
        return terminal

    # ── 主循环 ──────────────────────────────────────────

    async def _loop(self, question, schemas, s: "_Session", trace) -> AgentResult:
        messages = [{"role": "system", "content": s.system}, {"role": "user", "content": question}]
        nudged = False
        forced = False

        for step in range(1, self.max_steps + 2):
            if time.monotonic() - s.started > self.wall_timeout:
                return self._degrade(question, s, "wall_timeout")

            over_budget = s.tokens_used > self.token_budget
            if (step > self.max_steps or over_budget) and not forced:
                forced = True
                messages.append({"role": "user", "content": FORCED_NOTE})
                trace.append({"type": "force", "step": step,
                              "reason": "token_budget" if over_budget else "step_budget"})
            if forced:
                schemas = CONTROL_TOOL_SCHEMAS  # 只允许终止类工具

            try:
                resp = self.llm.chat_with_tools(messages, schemas, max_tokens=AGENT_MAX_TOKENS)
            except Exception as e:
                logger.warning(f"Agent LLM 调用失败: {e}")
                trace.append({"type": "llm_error", "step": step, "detail": str(e)})
                return self._degrade(question, s, f"llm_error:{e}")

            s.tokens_used += int((resp.get("usage") or {}).get("total_tokens", 0))
            msg = resp.get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            s.steps = step
            trace.append({"type": "assistant", "step": step,
                          "content": content[:300], "tool_calls": [
                              {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"][:300]}
                              for tc in tool_calls]})

            if not tool_calls:
                if forced:
                    return self._degrade(question, s, "forced_no_tool_call")
                if nudged:
                    return self._degrade(question, s, "no_tool_call")
                nudged = True
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": NO_TOOL_CALL_NOTE})
                continue

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                s.tool_calls += 1

                result, outcome = await self._execute(name, args, s, forced, messages)
                if isinstance(result, dict):
                    hit_ids = [str(d.get("id", "")) for d in result.get("results", []) if d.get("id")]
                    trace_ids = hit_ids or result.get("ids", [])
                else:
                    trace_ids = []
                trace.append({"type": "tool", "step": step, "tool": name,
                              "args": {k: str(v)[:120] for k, v in args.items()},
                              "outcome": outcome,
                              "result_ids": trace_ids})
                if outcome == "terminal":
                    return result["terminal"]
                messages.append({
                    "role": "tool", "tool_call_id": tc.get("id", f"call_{s.tool_calls}"),
                    "name": name, "content": _truncate_json(result, AGENT_MAX_TOOL_RESULT_CHARS),
                })
                if forced and outcome != "rejected" and name not in ("finalize_answer", "ask_clarification"):
                    # 强制阶段偷跑检索：结果照常回喂，同时提醒终止
                    messages.append({"role": "user", "content": FORCED_NOTE})

        return self._degrade(question, s, "loop_exit")

    async def _execute(self, name: str, args: dict, s: "_Session", forced: bool, messages):
        """执行单个工具调用。返回 (回喂结果, outcome∈{ok,terminal,rejected})"""
        if name == "finalize_answer":
            return self._try_finalize(args, s)

        if name == "ask_clarification":
            try:
                payload = ClarifyPayload(**args)
            except ValidationError as e:
                return ({"error": "invalid_args", "detail": str(e)}, "rejected")
            terminal = AgentResult(
                question=s.question, answer=payload.question_to_user,
                final_type="clarification", steps=s.steps, tool_calls=s.tool_calls,
                tokens_used=s.tokens_used, exposed_ids=sorted(s.evidence),
            )
            return ({"ids": [], "terminal": terminal}, "terminal")

        # 知识库工具 → 透传给 MCP 桥
        if forced:
            return ({"error": "budget_exhausted",
                     "detail": "步数/token 预算已用尽，禁止继续检索，请立刻终止"}, "rejected")
        try:
            result = await self.tool_invoke(name, args)
        except Exception as e:
            logger.warning(f"MCP 工具调用失败 {name}: {e}")
            return ({"error": "tool_failed", "detail": str(e)}, "ok")

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"raw": result}
        if not isinstance(result, dict):
            result = {"value": result}

        s.record_tool_result(name, result)
        return (result, "ok")

    def _try_finalize(self, args: dict, s: "_Session"):
        try:
            payload = FinalAnswerPayload(**args)
        except ValidationError as e:
            s.rejections += 1
            return ({"error": "invalid_payload",
                     "detail": f"字段校验失败: {e.errors() if hasattr(e,'errors') else e}"}, "rejected")

        unknown = [i for i in payload.evidence_ids if i not in s.evidence]
        if unknown:
            s.rejections += 1
            return ({"error": "invalid_evidence",
                     "detail": f"以下 id 未在本会话检索结果中出现，请修正: {unknown}"}, "rejected")

        terminal = AgentResult(
            question=s.question, answer=payload.answer,
            final_type="answer", steps=s.steps, tool_calls=s.tool_calls,
            tokens_used=s.tokens_used, evidence_ids=payload.evidence_ids,
            exposed_ids=sorted(s.evidence),
        )
        return ({"ids": payload.evidence_ids, "terminal": terminal}, "terminal")

    def _degrade(self, question: str, s: "_Session", reason: str) -> AgentResult:
        """降级链：既有管线 RAGEngine.ask（内部自带检索原文→兜底话术两级降级）"""
        logger.warning(f"Agent 降级: {reason}")
        answer, mode = "", "degraded"
        if self.fallback is not None:
            try:
                fb = self.fallback(question)
                answer = fb.get("answer", "")
                mode = fb.get("mode", "retrieval")
            except Exception as e:
                logger.error(f"降级管线也失败: {e}")
        from prompts import NO_RESULT_HINT
        if not answer:
            answer = NO_RESULT_HINT
        return AgentResult(
            question=question, answer=answer,
            final_type="degraded", degrade_reason=f"{reason}(pipeline_mode={mode})",
            steps=s.steps, tool_calls=s.tool_calls, tokens_used=s.tokens_used,
            exposed_ids=sorted(s.evidence),
        )

    def _save_trace(self, question: str, trace: list[dict], t0: float) -> str:
        if self.trace_dir is None:
            return ""
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            import hashlib
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qh = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
            path = self.trace_dir / f"trace_{stamp}_{qh}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                meta = {"type": "meta", "question": question,
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for ev in trace:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            return str(path)
        except Exception as e:
            logger.warning(f"轨迹保存失败: {e}")
            return ""


class _Session:
    """单次问答的运行态：证据集合与计数器"""

    def __init__(self, question: str, system: str):
        self.question = question
        self.system = system
        self.evidence: dict[str, dict] = {}  # id -> 摘要信息
        self.steps = 0
        self.tool_calls = 0
        self.tokens_used = 0
        self.rejections = 0
        self.started = time.monotonic()

    def record_tool_result(self, name: str, result: dict):
        if name == "hybrid_search" and isinstance(result.get("results"), list):
            for doc in result["results"]:
                doc_id = str(doc.get("id", ""))
                if doc_id:
                    self.evidence.setdefault(doc_id, {
                        "question": doc.get("question", ""),
                        "knowledge_point": doc.get("knowledge_point", ""),
                        "similarity": doc.get("similarity", 0),
                    })
