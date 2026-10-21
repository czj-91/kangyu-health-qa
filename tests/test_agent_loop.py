# -*- coding: utf-8 -*-
"""Agent 决策循环单元测试 — FakeLLM 脚本化响应，无网络/无模型"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.loop import (
    CONTROL_TOOL_SCHEMAS, FORCED_NOTE, AgentRunner,
    _truncate_json, build_system_prompt, mcp_tools_to_openai,
)
from prompts import NO_RESULT_HINT

KB_SCHEMA = [{
    "type": "function",
    "function": {"name": "hybrid_search", "description": "检索", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}},
}, {
    "type": "function",
    "function": {"name": "get_document", "description": "读全文", "parameters": {
        "type": "object", "properties": {"doc_id": {"type": "string"}}}},
}]


def search_result(query="q"):
    return {"query": query, "top_similarity": 0.9, "results": [
        {"id": "1", "question": "感冒了怎么办", "knowledge_point": "呼吸科",
         "similarity": 0.9, "snippet": "多休息"},
    ]}


def msg_with_calls(calls, content="", tokens=100):
    return {"message": {"content": content, "tool_calls": [
        {"id": f"call_{i}", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
        for i, (n, a) in enumerate(calls)]}, "usage": {"total_tokens": tokens}}


def msg_plain(content, tokens=50):
    return {"message": {"content": content, "tool_calls": []}, "usage": {"total_tokens": tokens}}


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []  # [(messages, tools)]

    def chat_with_tools(self, messages, tools, max_tokens=None):
        self.calls.append((messages, tools))
        r = self.responses.pop(0) if self.responses else msg_plain("没有更多脚本")
        if isinstance(r, Exception):
            raise r
        return r


class FakeBridge:
    """假 MCP 桥：记录调用，返回脚本化结果"""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if name in self.results:
            r = self.results[name]
            return r(args) if callable(r) else r
        if name == "hybrid_search":
            return search_result(args.get("query", ""))
        if name == "get_document":
            return {"id": args.get("doc_id"), "question": "q", "answer": "全文", "knowledge_point": "t"}
        return {"error": "unknown_tool", "detail": name}


def make_runner(llm, bridge=None, **kw):
    kw.setdefault("trace_dir", None)
    runner = AgentRunner(llm, (bridge or FakeBridge()).invoke,
                         fallback=kw.pop("fallback", lambda q: {"answer": "管线答案", "mode": "retrieval"}),
                         **kw)
    return runner


def run_q(runner, question="感冒了怎么办"):
    return runner.run(question, KB_SCHEMA)


class TestHappyPaths:
    def test_search_then_finalize(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒", "top_k": 5})]),
            msg_with_calls([("finalize_answer", {"answer": "多喝水多休息", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "answer"
        assert r.answer == "多喝水多休息"
        assert r.evidence_ids == ["1"]
        assert r.steps == 2
        assert r.tokens_used == 200

    def test_clarification_terminates(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "不舒服"})]),
            msg_with_calls([("ask_clarification", {"question_to_user": "请问具体是哪里不舒服？"})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "clarification"
        assert "哪里不舒服" in r.answer

    def test_fetch_document_flow(self):
        bridge = FakeBridge()
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("get_document", {"doc_id": "1"})]),
            msg_with_calls([("finalize_answer", {"answer": "依据全文回答", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm, bridge))
        assert r.final_type == "answer"
        assert ("get_document", {"doc_id": "1"}) in bridge.calls


class TestEvidenceValidation:
    def test_hallucinated_evidence_rejected_then_corrected(self, tmp_path):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "答案", "evidence_ids": ["999"]})]),
            msg_with_calls([("finalize_answer", {"answer": "修正后答案", "evidence_ids": ["1"]})]),
        ])
        runner = AgentRunner(llm, FakeBridge().invoke,
                             fallback=None, trace_dir=tmp_path)
        r = runner.run("感冒了怎么办", KB_SCHEMA)
        assert r.final_type == "answer"
        assert r.answer == "修正后答案"
        trace_text = open(r.trace_path, encoding="utf-8").read()
        assert "invalid_evidence" in trace_text or "rejected" in trace_text

    def test_empty_evidence_fails_pydantic(self):
        llm = FakeLLM([
            msg_with_calls([("finalize_answer", {"answer": "无证据答案", "evidence_ids": []})]),
            msg_with_calls([("ask_clarification", {"question_to_user": "能补充下吗"})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "clarification"  # 空 evidence 被拒后模型转向澄清

    def test_degenerate_evidence_string_dedup(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "规律作息可改善失眠。", "evidence_ids": ["1", "1", " "]})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "answer"
        assert r.evidence_ids == ["1"]

    def test_whitespace_answer_rejected_by_contract(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "   ", "evidence_ids": ["1"]})]),
            msg_with_calls([("ask_clarification", {"question_to_user": "请问还有什么症状？"})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "clarification"  # 空白正文被契约拒绝 → 模型转向澄清


class TestBudgetAndDegradation:
    def test_llm_error_degrades_to_pipeline(self):
        llm = FakeLLM([RuntimeError("connection reset")])
        r = run_q(make_runner(llm))
        assert r.final_type == "degraded"
        assert "llm_error" in r.degrade_reason
        assert r.answer == "管线答案"

    def test_no_fallback_uses_no_result_hint(self):
        llm = FakeLLM([RuntimeError("boom")])
        runner = AgentRunner(llm, FakeBridge().invoke, fallback=None, trace_dir=None)
        r = runner.run("感冒", KB_SCHEMA)
        assert r.answer == NO_RESULT_HINT

    def test_no_tool_call_nudge_then_finalize(self):
        llm = FakeLLM([
            msg_plain("我觉得感冒要休息"),  # 直接输出正文 → 被引导
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "规范回答", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm))
        assert r.final_type == "answer"
        # 第二次模型调用前，messages 里应已注入"必须走工具终止"的引导语
        assert any("finalize_answer 提交" in (m.get("content") or "") for m in llm.calls[1][0])

    def test_twice_no_tool_call_degrades(self):
        llm = FakeLLM([msg_plain("A"), msg_plain("B")])
        r = run_q(make_runner(llm))
        assert r.final_type == "degraded"
        assert "no_tool_call" in r.degrade_reason

    def test_step_budget_forces_control_only(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "收尾", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm, max_steps=1))
        assert r.final_type == "answer"
        last_tools = llm.calls[-1][1]
        assert last_tools == CONTROL_TOOL_SCHEMAS  # 强制阶段只暴露终止工具

    def test_never_terminates_degrades_loop_exit(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
        ])
        r = run_q(make_runner(llm, max_steps=1))
        assert r.final_type == "degraded"
        assert r.degrade_reason.startswith(("loop_exit", "forced_no_tool_call"))

    def test_token_budget_triggers_force(self):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})], tokens=99999),
            msg_with_calls([("finalize_answer", {"answer": "预算内收尾", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm, token_budget=1000))
        assert r.final_type == "answer"
        assert any(m.get("content", "").startswith(FORCED_NOTE[:10]) for m in llm.calls[-1][0])

    def test_unknown_tool_passthrough_error(self):
        bridge = FakeBridge()
        llm = FakeLLM([
            msg_with_calls([("mystery_tool", {"x": 1})]),
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "感冒以休息对症为主。", "evidence_ids": ["1"]})]),
        ])
        r = run_q(make_runner(llm, bridge))
        assert r.final_type == "answer"
        assert ("mystery_tool", {"x": 1}) in bridge.calls


class TestTraceAndHelpers:
    def test_trace_saved_as_jsonl(self, tmp_path):
        llm = FakeLLM([
            msg_with_calls([("hybrid_search", {"query": "感冒"})]),
            msg_with_calls([("finalize_answer", {"answer": "A", "evidence_ids": ["1"]})]),
        ])
        runner = AgentRunner(llm, FakeBridge().invoke, fallback=None, trace_dir=tmp_path / "tr")
        r = runner.run("感冒了怎么办", KB_SCHEMA)
        lines = [json.loads(l) for l in open(r.trace_path, encoding="utf-8")]
        assert lines[0]["type"] == "meta" and lines[0]["question"] == "感冒了怎么办"
        types = [l["type"] for l in lines]
        assert "assistant" in types and "tool" in types and "final" in types

    def test_truncate_json_stays_valid(self):
        small = _truncate_json({"a": "b"}, 100)
        assert json.loads(small) == {"a": "b"}
        big = _truncate_json({"data": "x" * 5000}, 100)
        parsed = json.loads(big)
        assert parsed["_note"]

    def test_mcp_tools_to_openai(self):
        tools = [SimpleNamespace(
            name="hybrid_search", description="检索",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                          "required": ["query"], "additionalProperties": False})]
        out = mcp_tools_to_openai(tools)
        fn = out[0]["function"]
        assert out[0]["type"] == "function"
        assert fn["parameters"]["required"] == ["query"]
        assert "additionalProperties" not in fn["parameters"]

    def test_system_prompt_carries_budgets(self):
        p = build_system_prompt(7, 8888)
        assert "7 个决策步" in p and "8888" in p
        assert "120" in p  # 急症护栏保留
