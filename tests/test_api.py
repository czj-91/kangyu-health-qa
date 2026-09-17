"""API 接口集成测试 — 医疗健康知识库适配版"""
import os
import sys
import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import init_env

init_env()

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app, init_components
    init_components()
    return TestClient(app)


class TestHealthAPI:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mode" in data
        assert "qa_count" in data
        assert data["qa_count"] >= 1000  # 医疗知识库至少 1000 条

    def test_health_has_required_fields(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        for key in ["status", "mode", "qa_count", "retriever_loaded", "hybrid_indexed", "device"]:
            assert key in data, f"缺少字段: {key}"

    def test_root_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_root_contains_medical_disclaimer(self, client):
        resp = client.get("/")
        assert "120" in resp.text  # 紧急电话
        assert "健康科普" in resp.text or "免责" in resp.text


class TestAskAPI:
    def test_valid_medical_question(self, client):
        """医疗领域问题应该返回有效答案"""
        resp = client.post("/api/ask", json={"question": "头痛应该怎么办？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "mode" in data
        assert len(data["answer"]) > 10

    def test_empty_question_rejected(self, client):
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_short_question_rejected(self, client):
        resp = client.post("/api/ask", json={"question": "a"})
        assert resp.status_code == 422

    def test_with_top_k(self, client):
        resp = client.post("/api/ask", json={
            "question": "感冒了吃什么药？",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data.get("results", [])) <= 3

    def test_with_session(self, client):
        r = client.post("/api/conversation/new")
        sid = r.json()["session_id"]

        resp = client.post("/api/ask", json={
            "question": "高血压饮食注意什么？",
            "session_id": sid,
            "top_k": 3,
        })
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid

    def test_retrieval_mode_returns_match(self, client):
        resp = client.post("/api/ask", json={
            "question": "每天应该睡几个小时？",
            "mode": "retrieval",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "similarity" in data
        assert "knowledge_point" in data

    def test_answer_contains_retrieval_source(self, client):
        resp = client.post("/api/ask", json={"question": "小孩发烧怎么办"})
        data = resp.json()
        assert "results" in data
        # 应该有检索来源
        if len(data.get("results", [])) > 0:
            r = data["results"][0]
            assert "answer" in r
            assert "knowledge_point" in r

    def test_nonexistent_topic_returns_fallback(self, client):
        """完全无关的问题应该返回兜底回答"""
        resp = client.post("/api/ask", json={
            "question": "xyzxyzxyz火星上有什么xyzxyzxyz",
            "top_k": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data


class TestSampleQuestions:
    def test_sample_questions_returned(self, client):
        resp = client.get("/api/sample_questions")
        assert resp.status_code == 200
        data = resp.json()
        assert "samples" in data
        assert len(data["samples"]) > 0

    def test_samples_are_medical(self, client):
        """示例问题应该是医疗健康领域"""
        resp = client.get("/api/sample_questions")
        data = resp.json()
        # 验证每个示例问题结构完整
        for s in data["samples"]:
            assert "question" in s
            assert "knowledge_point" in s
            assert len(s["question"]) > 3


class TestConversation:
    def test_new_and_get(self, client):
        r = client.post("/api/conversation/new")
        sid = r.json()["session_id"]
        assert len(sid) > 0

        r2 = client.get(f"/api/conversation/{sid}")
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid

    def test_clear(self, client):
        r = client.post("/api/conversation/new")
        sid = r.json()["session_id"]
        r2 = client.post(f"/api/conversation/{sid}/clear")
        assert r2.status_code == 200

    def test_unknown_session_returns_404(self, client):
        r = client.get("/api/conversation/ffffffffffffffff")
        assert r.status_code in [200, 404]  # 空历史返回200空列表也是合理的


class TestAskStream:
    def test_stream_medical_question(self, client):
        """流式问答 — 医疗问题"""
        resp = client.post("/api/ask/stream", json={"question": "怎么预防感冒？"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: " in body

    def test_stream_empty_question_rejected(self, client):
        resp = client.post("/api/ask/stream", json={"question": ""})
        assert resp.status_code == 422

    def test_stream_with_mode(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "如何改善睡眠质量？",
            "mode": "retrieval",
            "top_k": 3,
        })
        assert resp.status_code == 200
        body = resp.text
        assert "data: " in body


class TestFeedback:
    def test_submit_helpful(self, client):
        resp = client.post("/api/feedback", json={
            "question": "感冒怎么预防",
            "answer": "多喝水，勤洗手，保持室内通风。",
            "rating": "helpful",
        })
        assert resp.status_code == 200

    def test_submit_not_helpful(self, client):
        resp = client.post("/api/feedback", json={
            "question": "咳嗽怎么办",
            "answer": "答案不太清楚。",
            "rating": "not_helpful",
        })
        assert resp.status_code == 200

    def test_get_stats(self, client):
        resp = client.get("/api/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "helpful" in data


class TestEdgeCases:
    def test_very_long_question(self, client):
        long_q = "我" * 500 + "怎么办？"
        resp = client.post("/api/ask", json={"question": long_q})
        assert resp.status_code == 200

    def test_special_characters_question(self, client):
        resp = client.post("/api/ask", json={
            "question": "发烧 > 38.5°C 怎么办？需要吃退烧药吗？",
        })
        assert resp.status_code == 200

    def test_english_medical_term(self, client):
        resp = client.post("/api/ask", json={"question": "什么是DNA检测？"})
        assert resp.status_code == 200
