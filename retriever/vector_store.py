"""ChromaDB 向量存储 — 替代 NumPy 暴力搜索"""
import numpy as np
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, TOP_K_SEMANTIC
from logger_config import logger


class ChromaVectorStore:
    """基于 ChromaDB 的语义向量检索"""

    def __init__(self, collection_name: str = "qa_library"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None

    def exists(self) -> bool:
        try:
            names = self.client.list_collections()
            return any(
                (n if isinstance(n, str) else n.name) == self.collection_name
                for n in names
            )
        except Exception as e:
            logger.warning("ChromaDB exists() 检查失败: {}", e)
            return False

    def _get_or_reconnect(self):
        """获取集合引用，自动处理过期引用"""
        if self.collection is not None:
            try:
                self.collection.count()
                return
            except Exception as e:
                logger.debug("集合引用已过期，重新连接: {}", e)
                self.collection = None

        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.warning("ChromaDB 获取集合失败: {}", e)
            self.collection = None

    def build_from_qa_library(self, qa_library: dict, embeddings: np.ndarray):
        """从问答库和嵌入向量构建 ChromaDB 索引"""
        # 先清理旧集合（可能残留自上次异常退出）
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.debug("删除旧集合时异常（可能不存在）: {}", e)

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        questions = qa_library["questions"]
        answers = qa_library["answers"]
        kps = qa_library["knowledge_points"]

        batch_size = 100
        for i in range(0, len(questions), batch_size):
            end = min(i + batch_size, len(questions))
            ids = [str(j) for j in range(i, end)]
            docs = questions[i:end]
            metas = [
                {"answer": answers[j], "knowledge_point": kps[j], "question": questions[j]}
                for j in range(i, end)
            ]
            embs = embeddings[i:end].tolist()
            self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

        logger.info(f"ChromaDB 索引构建完成: {self.collection.count()} 条, 路径: {CHROMA_PERSIST_DIR}")

    def search(self, query_embedding: np.ndarray, top_k: int = 0) -> list[dict]:
        """语义向量检索"""
        self._get_or_reconnect()
        if self.collection is None:
            logger.warning("ChromaDB 集合不可用，返回空结果")
            return []

        k = top_k or TOP_K_SEMANTIC
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB 查询失败: {e}")
            return []

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0
                output.append({
                    "id": doc_id,
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "knowledge_point": meta["knowledge_point"],
                    "similarity": round(1.0 - distance, 4),
                })
        return output
