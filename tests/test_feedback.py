"""反馈收集单元测试"""
import json
import os
import tempfile
import pytest
from feedback import save_feedback, get_feedback_stats


class TestFeedback:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl", encoding="utf-8"
        )
        self.tmpfile.close()

    def teardown_method(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_save_and_stats(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)

        save_feedback("问题1", "答案1", "helpful", "rag", "session1")
        save_feedback("问题2", "答案2", "not_helpful", "retrieval", "session2")

        stats = get_feedback_stats()
        assert stats["total"] == 2
        assert stats["helpful"] == 1
        assert stats["not_helpful"] == 1
        assert stats["helpful_rate"] == 0.5

    def test_save_creates_valid_jsonl(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        save_feedback("头痛怎么办？", "建议休息，必要时就医。", "helpful", "rag", "sid1")
        with open(self.tmpfile.name, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["question"] == "头痛怎么办？"
        assert record["rating"] == "helpful"
        assert "timestamp" in record
        assert "session_id" in record

    def test_empty_stats(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        stats = get_feedback_stats()
        assert stats["total"] == 0
        assert stats["helpful"] == 0
        assert stats["not_helpful"] == 0

    def test_answer_truncated_to_2000(self, monkeypatch):
        """修复验证：截断从 500 改为 2000"""
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        long_answer = "A" * 3000
        save_feedback("问", long_answer, "helpful", "rag")
        with open(self.tmpfile.name, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert len(record["answer"]) <= 2000

    def test_multiple_feedbacks_append(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        for i in range(10):
            save_feedback(f"问题{i}", f"答案{i}", "helpful", "rag")
        stats = get_feedback_stats()
        assert stats["total"] == 10

    def test_corrupt_line_skipped(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        # 写入一条合法记录 + 一条损坏记录
        save_feedback("好问题", "好答案", "helpful", "rag")
        with open(self.tmpfile.name, "a", encoding="utf-8") as f:
            f.write("这不是合法的 JSON\n")
        stats = get_feedback_stats()
        assert stats["total"] == 1  # 损坏行被跳过

    def test_all_ratings_types(self, monkeypatch):
        monkeypatch.setattr("feedback.FEEDBACK_PATH", self.tmpfile.name)
        save_feedback("q1", "a1", "helpful", "rag")
        save_feedback("q2", "a2", "not_helpful", "retrieval")
        stats = get_feedback_stats()
        assert stats["helpful"] == 1
        assert stats["not_helpful"] == 1
