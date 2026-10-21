# -*- coding: utf-8 -*-
"""Agent 数据契约 — Pydantic 定义工具输入/输出与终态载荷。

finalize_answer 的证据校验是 Agent 的核心约束：模型必须引用
本会话中真实检索命中过的 doc id 才能终止，凭空编造的 id 会被
拒绝并回喂给模型自我修正（证据归属校验）。
"""
from pydantic import BaseModel, Field, field_validator


class EvidenceDoc(BaseModel):
    """单条检索证据（答案截断为摘要，全文需 get_document 读取）"""
    id: str
    question: str
    knowledge_point: str = ""
    similarity: float = 0.0
    snippet: str = ""


class SearchOutput(BaseModel):
    """hybrid_search 工具返回"""
    query: str
    results: list[EvidenceDoc] = Field(default_factory=list)
    top_similarity: float = 0.0


class ToolError(BaseModel):
    """工具错误统一结构 — 以 isError/结构化 JSON 返回，不抛裸异常"""
    error: str
    detail: str = ""


class FinalAnswerPayload(BaseModel):
    """finalize_answer 载荷：答案 + 证据 id 列表（至少 1 条、至多 5 条）"""
    answer: str = Field(min_length=2)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer 不能是空白字符")
        return v

    @field_validator("evidence_ids")
    @classmethod
    def _no_blank_ids(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("evidence_ids 不能为空")
        # 去重保序
        return list(dict.fromkeys(cleaned))


class ClarifyPayload(BaseModel):
    """ask_clarification 载荷：向用户反问以补全信息后终止本轮"""
    question_to_user: str = Field(min_length=2)
    reason: str = ""


class AgentResult(BaseModel):
    """一次 Agent 问答的最终产物"""
    question: str
    answer: str
    final_type: str  # "answer" | "clarification" | "degraded"
    degrade_reason: str = ""            # final_type=degraded 时的原因
    evidence_ids: list[str] = Field(default_factory=list)   # 实际引用（finalize 校验过）
    exposed_ids: list[str] = Field(default_factory=list)    # 本轮检索命中过的全部 id
    steps: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    latency_ms: float = 0.0
    trace_path: str = ""
