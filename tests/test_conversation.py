"""会话管理单元测试"""
import time
import pytest
from conversation import ConversationManager


class TestConversationManager:
    def setup_method(self):
        self.cm = ConversationManager()

    def test_create_session_returns_valid_id(self):
        sid = self.cm.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_create_session_unique_ids(self):
        sids = {self.cm.create_session() for _ in range(10)}
        assert len(sids) == 10

    def test_add_turn_stores_user_and_assistant(self):
        sid = self.cm.create_session()
        self.cm.add_turn(sid, "感冒了怎么办？", "多喝水多休息，必要时就医。")
        history = self.cm.get_history_messages(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "感冒了怎么办？"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "多喝水多休息，必要时就医。"

    def test_add_turn_preserves_full_answer(self):
        """修复验证：答案不再被截断到300字"""
        sid = self.cm.create_session()
        long_answer = "A" * 800
        self.cm.add_turn(sid, "测试问题", long_answer)
        history = self.cm.get_history_messages(sid)
        assert len(history[1]["content"]) == 800  # 完整保留

    def test_get_history_empty_for_unknown_session(self):
        assert self.cm.get_history("nonexistent") == ""

    def test_get_history_messages_unknown_session(self):
        assert self.cm.get_history_messages("nonexistent") == []

    def test_clear_session_removes_all_data(self):
        sid = self.cm.create_session()
        self.cm.add_turn(sid, "问题", "答案")
        self.cm.clear(sid)
        assert self.cm.get_history(sid) == ""
        assert self.cm.get_history_messages(sid) == []

    def test_add_turn_unknown_session_no_error(self):
        # 不应抛异常
        self.cm.add_turn("nonexistent", "问", "答")

    def test_get_history_format(self):
        sid = self.cm.create_session()
        self.cm.add_turn(sid, "头痛怎么办", "建议休息并观察，持续不缓解请就医。")
        text = self.cm.get_history(sid)
        assert "用户:" in text
        assert "助教:" in text
        assert "头痛怎么办" in text
        assert "建议休息" in text

    def test_multiple_turns_accumulate(self):
        sid = self.cm.create_session()
        for i in range(5):
            self.cm.add_turn(sid, f"问题{i}", f"答案{i}")
        history = self.cm.get_history_messages(sid)
        assert len(history) == 10  # 5 turns * 2 messages

    def test_max_conversation_turns_enforced(self):
        """修复验证：MAX_CONVERSATION_TURNS 从 5 改为 10"""
        sid = self.cm.create_session()
        for i in range(15):  # 超过 10 轮限制
            self.cm.add_turn(sid, f"问题{i}", f"答案{i}")
        history = self.cm.get_history_messages(sid)
        # 最多保留 10 轮 * 2 = 20 条消息
        assert len(history) <= 20
        # 应该保留的是最后的消息
        assert "问题14" in history[-2]["content"]

    def test_concurrent_session_isolation(self):
        sid1 = self.cm.create_session()
        sid2 = self.cm.create_session()
        self.cm.add_turn(sid1, "问题A", "答案A")
        self.cm.add_turn(sid2, "问题B", "答案B")
        assert "问题A" in self.cm.get_history(sid1)
        assert "问题B" in self.cm.get_history(sid2)
        assert "问题A" not in self.cm.get_history(sid2)

    def test_gc_cleans_expired_sessions(self, monkeypatch):
        """验证过期会话被清理（缩短 TTL 测试）"""
        monkeypatch.setattr("conversation.CONVERSATION_TTL_SECONDS", 0)
        sid = self.cm.create_session()
        self.cm.add_turn(sid, "问题", "答案")
        time.sleep(0.1)
        # 创建新会话触发 GC
        self.cm.create_session()
        # 过期会话应该被清理
        assert self.cm.get_history(sid) == ""
