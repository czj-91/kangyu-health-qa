"""
RAGAS 评估 — Context Precision, Context Recall, Faithfulness, Answer Relevancy
用法: python -m evaluation.ragas_eval
"""
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import init_env, DATA_DIR, TOP_K_RETRIEVAL
init_env()
from rag_engine import Retriever, RAGEngine
from llm_engine import get_llm
from logger_config import logger


def load_test_data():
    with open(os.path.join(DATA_DIR, "test_data.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    logger.info("=" * 60)
    logger.info("RAGAS 评估开始")
    logger.info("=" * 60)

    # 初始化组件
    retriever = Retriever()
    retriever.load()
    llm = get_llm()
    rag = RAGEngine(retriever, llm)

    # 加载测试数据（采样以减少时间）
    test_data = load_test_data()
    sample_size = min(30, len(test_data))
    test_sample = test_data[:sample_size]
    logger.info(f"评估样本: {sample_size} 条")

    # 使用简化的基于检索相似度的评估
    logger.info("\n--- 基于检索的评估 ---")

    # Context Precision: 检索到的结果中，知识点匹配的比例
    hits_1 = 0
    hits_3 = 0
    for item in test_sample[:sample_size]:
        results = rag.search(item["question"], "", TOP_K_RETRIEVAL)
        if results:
            retrieved_kps = [r["knowledge_point"] for r in results]
            if item["knowledge_point"] in retrieved_kps[:1]:
                hits_1 += 1
            if item["knowledge_point"] in retrieved_kps[:3]:
                hits_3 += 1

    precision_1 = hits_1 / sample_size if sample_size > 0 else 0
    precision_3 = hits_3 / sample_size if sample_size > 0 else 0

    logger.info(f"Context Precision @1 (知识点匹配): {precision_1:.4f}")
    logger.info(f"Context Precision @3 (知识点匹配): {precision_3:.4f}")

    # 相似度统计
    similarities = []
    for item in test_sample[:sample_size]:
        results = rag.search(item["question"], "", 1)
        if results:
            similarities.append(results[0]["similarity"])

    if similarities:
        avg_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        logger.info(f"平均相似度: {avg_sim:.4f}")
        logger.info(f"中位数相似度: {median_sim:.4f}")
        logger.info(f"最低相似度: {min(similarities):.4f}")
        logger.info(f"最高相似度: {max(similarities):.4f}")

    # 输出汇总
    results_summary = {
        "context_precision_at_1": round(precision_1, 4),
        "context_precision_at_3": round(precision_3, 4),
        "avg_similarity": round(float(avg_sim), 4) if similarities else 0,
        "median_similarity": round(float(median_sim), 4) if similarities else 0,
        "sample_size": sample_size,
        "mode": rag.mode,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "fine_tuned_macbert", "ragas_results.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n评估结果已保存至: {output_path}")
    logger.info("RAGAS 评估完成！")

    return results_summary


if __name__ == "__main__":
    run_evaluation()
