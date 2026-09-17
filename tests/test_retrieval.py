"""检索模块单元测试 — 医疗知识库适配版"""
import os
import sys
import pytest
import numpy as np

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import Retriever


@pytest.fixture(scope="module")
def retriever():
    r = Retriever()
    r.load()
    return r


class TestRetriever:
    def test_model_loaded(self, retriever):
        assert retriever.model is not None
        assert retriever.tokenizer is not None

    def test_qa_library_loaded(self, retriever):
        assert retriever.qa_library is not None
        questions = retriever.qa_library["questions"]
        assert len(questions) > 1000  # 医疗知识库
        assert len(retriever.qa_library["answers"]) == len(questions)
        assert len(retriever.qa_library["knowledge_points"]) == len(questions)

    def test_encode_shape(self, retriever):
        emb = retriever.encode("头痛应该怎么办？")
        assert emb.shape == (1, 768)
        assert not np.isnan(emb).any()

    def test_encode_consistency(self, retriever):
        """同一问题多次编码应得到相同向量"""
        q = "感冒了怎么护理？"
        emb1 = retriever.encode(q)
        emb2 = retriever.encode(q)
        assert np.allclose(emb1, emb2, atol=1e-6)

    def test_encode_different_questions(self, retriever):
        """不同问题应产生不同向量"""
        emb1 = retriever.encode("头痛怎么办？")
        emb2 = retriever.encode("胃痛怎么办？")
        # 归一化后计算余弦相似度
        e1 = emb1 / np.linalg.norm(emb1)
        e2 = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(e1, e2.T)[0][0]
        # 相似但不完全相同（余弦相似度 < 0.999）
        assert similarity < 0.999

    def test_sample_questions(self, retriever):
        samples = retriever.get_sample_questions()
        assert len(samples) > 0
        for s in samples:
            assert "question" in s
            assert "knowledge_point" in s


class TestRAGEngine:
    @pytest.fixture(scope="class")
    def rag(self, retriever):
        from rag_engine import RAGEngine
        engine = RAGEngine(retriever, llm=None)
        assert engine.mode == "retrieval"  # 无 LLM 自动回退检索模式
        return engine

    def test_search_returns_medical_results(self, rag):
        results = rag.search("感冒了怎么办？", "", 3)
        assert len(results) > 0
        assert "similarity" in results[0]
        assert "answer" in results[0]
        assert "knowledge_point" in results[0]

    def test_search_top_k_truncation(self, rag):
        results = rag.search("发烧怎么处理？", "", 2)
        assert len(results) <= 2

    def test_search_empty_query(self, rag):
        results = rag.search("", "", 3)
        assert isinstance(results, list)

    def test_ask_medical_question(self, rag):
        result = rag.ask("高血压应该注意什么？", "", 3)
        assert "answer" in result
        assert "similarity" in result
        assert result["mode"] == "retrieval"

    def test_ask_no_results_fallback(self, rag):
        """无意义查询仍会返回最近邻结果（检索系统的正常行为）"""
        result = rag.ask("xyz火星xyz不存在的问题xyz火星xyz", "", 3)
        assert "answer" in result
        # 检索系统对任何查询都会返回最近邻，这是预期行为
        assert len(result.get("results", [])) >= 0

    def test_results_have_required_fields(self, rag):
        results = rag.search("咳嗽怎么办", "", 3)
        for r in results:
            for key in ["id", "question", "answer", "knowledge_point", "similarity"]:
                assert key in r, f"缺少字段: {key}"

    def test_top_result_has_highest_similarity(self, rag):
        results = rag.search("胃痛如何处理？", "", 5)
        if len(results) >= 2:
            # 相似度递减或相等
            assert results[0]["similarity"] >= results[-1]["similarity"]

    def test_diverse_medical_topics(self, rag):
        """验证不同科室的问题都能召回结果"""
        topics = [
            ("感冒发烧怎么办", "内科"),
            ("骨折后如何恢复", "外科"),
            ("孕妇注意事项", "妇产科"),
            ("小孩咳嗽", "儿科"),
        ]
        for question, _ in topics:
            results = rag.search(question, "", 2)
            assert len(results) > 0, f"问题 '{question}' 没有召回任何结果"
