"""混合检索 RRF 融合 + Reranker 单元测试"""
import numpy as np
import pytest

pytestmark = pytest.mark.integration
from retriever.hybrid import HybridRetriever


class TestRRFFusion:
    def setup_method(self):
        self.hybrid = HybridRetriever()

    def _make_doc(self, doc_id: str, score: float) -> dict:
        return {
            "id": doc_id,
            "question": f"问题{doc_id}",
            "answer": f"答案{doc_id}",
            "knowledge_point": "Python基础",
            "similarity": score,
        }

    def test_rrf_fusion_merges_both_sources(self):
        semantic = [
            self._make_doc("1", 0.9),
            self._make_doc("2", 0.8),
        ]
        bm25 = [
            self._make_doc("2", 0.7),
            self._make_doc("3", 0.6),
        ]
        result = self.hybrid._rrf_fusion(semantic, bm25)
        assert len(result) == 3
        ids = [d["id"] for d in result]
        assert "1" in ids
        assert "2" in ids
        assert "3" in ids

    def test_rrf_fusion_sorts_by_fusion_score(self):
        semantic = [
            self._make_doc("A", 0.9),
            self._make_doc("B", 0.5),
            self._make_doc("C", 0.3),
        ]
        bm25 = [
            self._make_doc("B", 0.9),
            self._make_doc("C", 0.7),
        ]
        result = self.hybrid._rrf_fusion(semantic, bm25)
        # B 在 semantic 排第2、bm25 排第1，综合应排第1
        # A 仅在 semantic 排第1，C 在 semantic 排第3、bm25 排第2
        assert result[0]["id"] == "B"

    def test_rrf_fusion_empty_inputs(self):
        result = self.hybrid._rrf_fusion([], [])
        assert result == []

    def test_rrf_fusion_single_source(self):
        semantic = [self._make_doc("1", 0.9)]
        result = self.hybrid._rrf_fusion(semantic, [])
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_rrf_fusion_has_fusion_score(self):
        semantic = [self._make_doc("1", 0.9)]
        result = self.hybrid._rrf_fusion(semantic, [])
        assert "fusion_score" in result[0]
        assert isinstance(result[0]["fusion_score"], float)


class TestSanitize:
    def setup_method(self):
        self.hybrid = HybridRetriever()

    def test_sanitize_numpy_float(self):
        doc = {"score": np.float32(0.95), "name": "test"}
        result = self.hybrid._sanitize(doc)
        assert isinstance(result["score"], float)
        assert result["score"] == pytest.approx(0.95, rel=1e-4)

    def test_sanitize_numpy_int(self):
        doc = {"count": np.int64(42), "name": "test"}
        result = self.hybrid._sanitize(doc)
        assert isinstance(result["count"], int)
        assert result["count"] == 42

    def test_sanitize_numpy_array(self):
        doc = {"vec": np.array([1.0, 2.0]), "name": "test"}
        result = self.hybrid._sanitize(doc)
        assert isinstance(result["vec"], list)

    def test_sanitize_preserves_str(self):
        doc = {"name": "hello", "value": 3.14}
        result = self.hybrid._sanitize(doc)
        assert result["name"] == "hello"
        assert result["value"] == 3.14
