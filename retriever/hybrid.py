"""混合检索 + RRF 融合 + BGE Reranker 精排"""
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from config import TOP_K_RETRIEVAL, TOP_K_SEMANTIC, TOP_K_BM25, RRF_K, ENABLE_RERANKER, RERANKER_MODEL, RERANK_SKIP_CONFIDENCE
from logger_config import logger
from retriever.vector_store import ChromaVectorStore
from retriever.bm25_retriever import BM25Retriever


class HybridRetriever:
    """混合检索引擎: 语义 + BM25 → RRF 融合 → Reranker 精排"""

    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = None
        self._reranker_loaded = False

    def build_index(self, qa_library: dict, embeddings: np.ndarray):
        """构建双路索引"""
        logger.info("构建混合检索引擎索引...")
        self.vector_store.build_from_qa_library(qa_library, embeddings)
        self.bm25.build_index(qa_library)

    def _load_reranker(self):
        """延迟加载 Reranker"""
        if self._reranker_loaded or not ENABLE_RERANKER:
            return
        try:
            logger.info(f"加载 Reranker: {RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            if torch.cuda.is_available():
                self.reranker.model = self.reranker.model.to("cuda")
            self._reranker_loaded = True
            logger.info("Reranker 加载完成")
        except Exception as e:
            logger.warning(f"Reranker 加载失败，跳过精排: {e}")
            self._reranker_loaded = True  # 不再重试

    def _rrf_fusion(self, semantic_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion — 合并双路检索结果"""
        doc_map: dict[str, dict] = {}
        all_keys: list[tuple[str, int, str]] = []  # (doc_key, rank, source)

        for rank, doc in enumerate(semantic_results):
            key = doc["id"]
            doc_map[key] = {**doc}
            all_keys.append((key, rank, "semantic"))

        for rank, doc in enumerate(bm25_results):
            key = doc["id"]
            if key not in doc_map:
                doc_map[key] = {**doc}
            all_keys.append((key, rank, "bm25"))

        # RRF 分数计算
        rrf_scores: dict[str, float] = {}
        for key, rank, __source in all_keys:
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        merged = [{
            **doc_map[key],
            "fusion_score": round(rrf_scores[key], 4),
        } for key in sorted_keys]

        return merged

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Reranker 精排"""
        if not self.reranker or len(candidates) <= 1:
            return candidates[:top_k]

        # 以问题文本对齐打分: 长答案在 512 截断下会稀释相关性信号
        pairs = [(query, doc["question"]) for doc in candidates]
        scores = self.reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]

        reranked = []
        for i in sorted_indices[:top_k]:
            candidates[i]["rerank_score"] = round(float(scores[i]), 4)
            if float(scores.max()) > 0:
                candidates[i]["similarity"] = round(float(scores[i]) / float(max(float(scores.max()), 1e-8)), 4)
            else:
                candidates[i]["similarity"] = float(candidates[i].get("similarity", 0))
            reranked.append(candidates[i])

        return reranked

    @staticmethod
    def _sanitize(item: dict) -> dict:
        """将 numpy 类型转换为 Python 原生类型"""
        out = {}
        for k, v in item.items():
            if isinstance(v, (np.floating, np.float32, np.float64)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, np.int32, np.int64)):
                out[k] = int(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def search(self, query: str, query_embedding: np.ndarray, top_k: int = 0) -> list[dict]:
        """混合检索主流程"""
        k = top_k or TOP_K_RETRIEVAL
        fusion_candidates = min(TOP_K_SEMANTIC + TOP_K_BM25, 30)

        # 1. 双路召回
        semantic_results = self.vector_store.search(query_embedding, TOP_K_SEMANTIC)
        bm25_results = self.bm25.search(query, TOP_K_BM25)

        logger.debug(f"语义召回: {len(semantic_results)}, BM25 召回: {len(bm25_results)}")

        # 2. RRF 融合
        merged = self._rrf_fusion(semantic_results, bm25_results)
        candidates = merged[:fusion_candidates]

        # 3. Reranker 精排 (精确命中直通, 避免精排破坏完全匹配)
        top_conf = float(candidates[0].get("similarity", 0.0) or 0.0) if candidates else 0.0
        if ENABLE_RERANKER and len(candidates) > k and top_conf < RERANK_SKIP_CONFIDENCE:
            self._load_reranker()
            if self.reranker:
                final = self._rerank(query, candidates, k)
            else:
                final = candidates[:k]
        else:
            final = candidates[:k]

        # 确保所有数值都是 Python 原生类型（避免 JSON 序列化错误）
        final = [self._sanitize(item) for item in final]

        logger.debug(f"最终返回: {len(final)} 条, "
                     f"最高分: {final[0]['similarity']:.4f}" if final else "无结果")

        return final
