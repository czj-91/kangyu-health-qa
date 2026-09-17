"""
RAG 引擎 v2 — 混合检索 + Reranker + 查询重写 + 多轮对话 + 流式生成
"""
import os
import json
import numpy as np
import torch
from typing import Protocol, runtime_checkable, Generator

from config import (
    init_env, EMBED_MODEL_NAME, FINE_TUNED_MODEL_DIR, TOP_K_RETRIEVAL,
    MAX_SEQ_LENGTH, RAG_MODE, ENABLE_QUERY_REWRITE,
)
from prompts import build_rag_prompt, QUERY_REWRITE_PROMPT, sanitize_answer
from retriever.hybrid import HybridRetriever
from logger_config import logger

init_env()
# 线程数由 PyTorch 自动管理，不再硬编码限制

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


# ── LLM 接口协议 — 兼容 LLMEngine (本地) 和 APILLMEngine (远程) ──
@runtime_checkable
class LLMLike(Protocol):
    """LLM 引擎统一接口协议 — 本地模型和远程 API 后端均需实现"""
    loaded: bool

    def generate(self, prompt: str) -> str: ...
    def generate_stream(self, prompt: str) -> Generator[str, None, None]: ...
    def load(self) -> None: ...


class Retriever:
    """嵌入模型加载 + 问题编码 — 供 HybridRetriever 使用"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.qa_library = None
        self.qa_embeddings = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"嵌入模型设备: {self.device}")
        if os.path.exists(os.path.join(FINE_TUNED_MODEL_DIR, "config.json")):
            logger.info("加载微调后的 MacBERT 模型...")
            full_model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
            self.model = full_model.bert
            self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
        else:
            logger.info("加载原始 MacBERT 模型...")
            self.model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)

        self.model = self.model.to(self.device)
        self.model.eval()

        qa_path = os.path.join(FINE_TUNED_MODEL_DIR, "qa_library.json")
        emb_path = os.path.join(FINE_TUNED_MODEL_DIR, "qa_embeddings.npy")
        if os.path.exists(qa_path) and os.path.exists(emb_path):
            import json
            with open(qa_path, "r", encoding="utf-8") as f:
                self.qa_library = json.load(f)
            self.qa_embeddings = np.load(emb_path)
            logger.info(f"问答库加载完成: {len(self.qa_library['questions'])} 条")
        else:
            logger.warning("问答库未找到，请先运行 train.py")
            self.qa_library = {"questions": [], "answers": [], "knowledge_points": []}
            self.qa_embeddings = None

    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def get_sample_questions(self) -> list[dict]:
        if not self.qa_library:
            return []
        seen = set()
        samples = []
        for q, kp in zip(self.qa_library["questions"], self.qa_library["knowledge_points"]):
            if kp not in seen:
                samples.append({"question": q, "knowledge_point": kp})
                seen.add(kp)
        return samples[:10]


class RAGEngine:
    """RAG 引擎 v2 — 支持混合检索、流式生成、多轮对话"""

    def __init__(self, retriever: Retriever, llm: LLMLike | None = None):
        self.retriever = retriever
        self.llm = llm
        self.hybrid = HybridRetriever()

        # 构建 ChromaDB 和 BM25 索引
        if retriever.qa_library and retriever.qa_embeddings is not None:
            self.hybrid.build_index(retriever.qa_library, retriever.qa_embeddings)

        self._mode = RAG_MODE
        self._resolve_mode()

        # 查询编码缓存 — 避免重复编码相同问题
        self._embed_cache: dict[str, np.ndarray] = {}
        self._embed_cache_max = 128

    def _resolve_mode(self):
        if self._mode == "rag":
            self.effective_mode = "rag" if (self.llm and self.llm.loaded) else "retrieval"
        elif self._mode == "retrieval":
            self.effective_mode = "retrieval"
        else:
            self.effective_mode = "rag" if (self.llm and self.llm.loaded) else "retrieval"
        logger.info(f"RAG 运行模式: {self.effective_mode}")

    def resolve_mode(self):
        """公开的模式解析入口 — 供外部在 LLM 加载后重新判断运行模式"""
        self._resolve_mode()

    @property
    def mode(self) -> str:
        return self.effective_mode

    def _rewrite_query(self, question: str, chat_history: str) -> str:
        """使用 LLM 改写查询 — 展开指代词、添加上下文"""
        if not ENABLE_QUERY_REWRITE or not self.llm or not self.llm.loaded:
            return question
        if not chat_history:
            return question
        try:
            prompt = QUERY_REWRITE_PROMPT.format(chat_history=chat_history, question=question)
            rewritten = self.llm.generate(prompt).strip()
            if rewritten and len(rewritten) >= 3:
                logger.info(f"查询改写: '{question[:40]}...' → '{rewritten[:40]}...'")
                return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
        return question

    def _encode_with_cache(self, text: str) -> np.ndarray:
        """带缓存的编码 — 相同查询直接返回缓存结果"""
        cache_key = text.strip()[:120]
        if cache_key in self._embed_cache:
            logger.debug("命中编码缓存")
            return self._embed_cache[cache_key]
        emb = self.retriever.encode(text)
        if len(self._embed_cache) < self._embed_cache_max:
            self._embed_cache[cache_key] = emb
        return emb

    def search(self, question: str, chat_history: str = "", top_k: int = 0) -> list[dict]:
        """混合检索"""
        # 查询改写
        search_query = self._rewrite_query(question, chat_history)
        query_emb = self._encode_with_cache(search_query)
        return self.hybrid.search(search_query, query_emb, top_k)

    def ask(self, question: str, chat_history: str = "", top_k: int = 0) -> dict:
        """同步问答"""
        results = self.search(question, chat_history, top_k)
        if not results:
            return {
                "answer": "抱歉，未找到相关答案。请尝试其他问题。",
                "mode": self.effective_mode,
                "results": [],
            }

        best = results[0]

        # 检索模式
        if self.effective_mode == "retrieval":
            return {
                "answer": best["answer"],
                "mode": "retrieval",
                "matched_question": best["question"],
                "similarity": best["similarity"],
                "knowledge_point": best["knowledge_point"],
                "results": results,
            }

        # RAG 模式
        if self.llm is None or not self.llm.loaded:
            logger.warning("RAG 模式需要 LLM 但未加载，回退检索模式")
            return {
                "answer": best["answer"],
                "mode": "retrieval",
                "matched_question": best["question"],
                "similarity": best["similarity"],
                "knowledge_point": best["knowledge_point"],
                "results": results,
            }

        prompt = build_rag_prompt(question, results, chat_history)
        try:
            generated = sanitize_answer(self.llm.generate(prompt))
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return {
                "answer": best["answer"],
                "mode": "retrieval",
                "matched_question": best["question"],
                "similarity": best["similarity"],
                "knowledge_point": best["knowledge_point"],
                "results": results,
            }

        return {
            "answer": generated,
            "mode": "rag",
            "sources": [
                {"question": r["question"], "knowledge_point": r["knowledge_point"], "similarity": r["similarity"]}
                for r in results
            ],
            "similarity": best["similarity"],
            "knowledge_point": best["knowledge_point"],
            "results": results,
        }

    def ask_stream(self, question: str, chat_history: str = "", top_k: int = 0):
        """流式问答 — 生成器"""
        results = self.search(question, chat_history, top_k)
        if not results:
            yield json.dumps({"type": "answer", "text": "抱歉，未找到相关答案。"})
            return

        best = results[0]

        if self.effective_mode == "retrieval":
            yield json.dumps({
                "type": "done",
                "answer": best["answer"],
                "mode": "retrieval",
                "similarity": best["similarity"],
                "knowledge_point": best["knowledge_point"],
                "sources": [
                    {"question": r["question"], "knowledge_point": r["knowledge_point"], "similarity": r["similarity"]}
                    for r in results
                ],
                "results": results,
            }, ensure_ascii=False)
            return

        # RAG 流式
        if self.llm is None or not self.llm.loaded:
            yield json.dumps({
                "type": "done",
                "answer": best["answer"],
                "mode": "retrieval",
                "similarity": best["similarity"],
                "knowledge_point": best["knowledge_point"],
                "sources": [
                    {"question": r["question"], "knowledge_point": r["knowledge_point"], "similarity": r["similarity"]}
                    for r in results
                ],
                "results": results,
            }, ensure_ascii=False)
            return

        prompt = build_rag_prompt(question, results, chat_history)
        try:
            for token in self.llm.generate_stream(prompt):
                yield token
        except Exception as e:
            logger.error(f"LLM 流式生成失败: {e}")
            yield str(e)
