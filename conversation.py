"""多轮对话管理 — 会话状态 + 对话历史"""
import time
import uuid
import threading
from collections import defaultdict
from config import MAX_CONVERSATION_TURNS, CONVERSATION_TTL_SECONDS, CONVERSATION_BACKEND
from logger_config import logger


class ConversationManager:
    """会话管理器，支持 memory 和 chroma 后端"""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)
        self._last_active: dict[str, float] = {}
        self._lock = threading.Lock()
        logger.info("对话管理器初始化，后端: {}", CONVERSATION_BACKEND)

    def _gc(self):
        """清理过期会话"""
        now = time.time()
        expired = [
            sid for sid, ts in self._last_active.items()
            if now - ts > CONVERSATION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
            del self._last_active[sid]
        if expired:
            logger.debug(f"清理 {len(expired)} 个过期会话")

    def create_session(self) -> str:
        with self._lock:
            self._gc()
            session_id = uuid.uuid4().hex[:12]
            self._sessions[session_id] = []
            self._last_active[session_id] = time.time()
            return session_id

    def add_turn(self, session_id: str, question: str, answer: str):
        with self._lock:
            if session_id not in self._sessions:
                return
            history = self._sessions[session_id]
            history.append({"role": "user", "content": question})
            # 保留完整答案用于多轮上下文（不做截断，由LLM自行处理长上下文）
            history.append({"role": "assistant", "content": answer})
            max_messages = MAX_CONVERSATION_TURNS * 2
            if len(history) > max_messages:
                self._sessions[session_id] = history[-max_messages:]
            self._last_active[session_id] = time.time()

    def get_history(self, session_id: str) -> str:
        with self._lock:
            if session_id not in self._sessions:
                return ""
            history = list(self._sessions[session_id])
        if not history:
            return ""
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助教"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_history_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_active.pop(session_id, None)


# 全局单例
conv_manager = ConversationManager()
