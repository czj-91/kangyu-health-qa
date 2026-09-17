"""BM25 关键词检索 — 弥补语义检索对专有名词的盲区"""
import numpy as np
from rank_bm25 import BM25Okapi

from config import TOP_K_BM25
from logger_config import logger


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self):
        self.bm25 = None
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.knowledge_points: list[str] = []
        self._tokenized: list[list[str]] = []

    def build_index(self, qa_library: dict):
        """从问答库构建 BM25 索引"""
        self.questions = qa_library["questions"]
        self.answers = qa_library["answers"]
        self.knowledge_points = qa_library["knowledge_points"]

        # 中文按字符级分词（简单且有效）
        self._tokenized = [list(q) for q in self.questions]
        self.bm25 = BM25Okapi(self._tokenized)
        logger.info(f"BM25 索引构建完成: {len(self.questions)} 条")

    def search(self, query: str, top_k: int = 0) -> list[dict]:
        """BM25 关键词检索"""
        if self.bm25 is None:
            return []

        k = top_k or TOP_K_BM25
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:min(k, len(scores))]

        # 归一化分数
        max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0

        return [
            {
                "id": str(idx),
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "knowledge_point": self.knowledge_points[idx],
                "score": float(scores[idx]),
                "similarity": round(float(scores[idx]) / max_score, 4),  # 归一化到 [0,1]
            }
            for idx in top_indices
            if scores[idx] > 0
        ]
