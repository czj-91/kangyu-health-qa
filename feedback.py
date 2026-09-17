"""用户反馈收集 — JSONL 文件持久化"""
import json
import os
import time
from typing import Any
from config import FEEDBACK_PATH
from logger_config import logger


def save_feedback(
    question: str,
    answer: str,
    rating: str,       # "helpful" | "not_helpful"
    mode: str,
    session_id: str = "",
    metadata: dict[str, Any] | None = None,
):
    """追加一条反馈到 JSONL 文件"""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer[:2000],
        "rating": rating,
        "mode": mode,
        "session_id": session_id,
        "metadata": metadata or {},
    }

    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"反馈已记录: rating={rating}, question={question[:50]}...")


def get_feedback_stats() -> dict:
    """读取反馈统计"""
    if not os.path.exists(FEEDBACK_PATH):
        return {"total": 0, "helpful": 0, "not_helpful": 0}

    total = 0
    helpful = 0
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                total += 1
                if record.get("rating") == "helpful":
                    helpful += 1
            except json.JSONDecodeError:
                logger.warning("反馈文件第 {} 行 JSON 解析失败，已跳过", total + 1)

    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": total - helpful,
        "helpful_rate": round(helpful / total, 2) if total > 0 else 0,
    }
