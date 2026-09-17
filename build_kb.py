"""
医疗健康知识库构建脚本 (康语 KangYu)
─────────────────────────────────────────
读取健康问答数据集，使用基础 chinese-macbert-base 生成问题嵌入向量，
并导出 qa_library.json + qa_embeddings.npy 到 model/health_macbert/。

说明：
  - 本脚本**不**写入微调权重（config.json），因此 rag_engine.Retriever
    会自动回退到基础 MacBERT 嵌入，可直接用于语义检索。
  - 若希望进一步提升医疗领域检索效果，可运行 train.py 在健康数据集上
    微调 MacBERT（会生成带 config.json 的微调模型）。

用法：
  python build_kb.py            # 默认使用 config 中的路径
  python build_kb.py --device cpu
"""
import os
import json
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from config import (
    init_env, RAW_DATA_DIR, FINE_TUNED_MODEL_DIR,
    EMBED_MODEL_NAME, MAX_SEQ_LENGTH,
)
from logger_config import logger

init_env()


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 清洗：确保三字段齐全且长度合理
    cleaned = [
        item for item in data
        if item.get("question") and item.get("answer")
        and len(item["question"]) >= 3 and len(item["answer"]) >= 10
    ]
    return cleaned


def build_embeddings(model, tokenizer, questions: list[str], device: str, batch_size: int = 16) -> np.ndarray:
    model.eval()
    model = model.to(device)
    all_embeddings = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="康语 健康知识库构建")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="编码设备 cuda / cpu")
    parser.add_argument("--dataset", default=os.path.join(RAW_DATA_DIR, "health_qa.json"),
                        help="健康问答数据集 JSON 路径")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("康语 · 医疗健康知识库构建")
    logger.info("=" * 60)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"未找到数据集: {args.dataset}")

    data = load_dataset(args.dataset)
    logger.info("加载数据集: {} 条有效问答", len(data))

    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    knowledge_points = [item.get("knowledge_point", "健康科普") for item in data]

    logger.info("加载嵌入模型: {}", EMBED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

    logger.info("正在为 {} 条问题生成嵌入向量 (device={})...", len(questions), args.device)
    embeddings = build_embeddings(model, tokenizer, questions, args.device)
    logger.info("嵌入向量维度: {}", embeddings.shape)

    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
    np.save(os.path.join(FINE_TUNED_MODEL_DIR, "qa_embeddings.npy"), embeddings)
    with open(os.path.join(FINE_TUNED_MODEL_DIR, "qa_library.json"), "w", encoding="utf-8") as f:
        json.dump({
            "questions": questions,
            "answers": answers,
            "knowledge_points": knowledge_points,
        }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("知识库构建完成！")
    logger.info("  qa_library.json : {}", os.path.join(FINE_TUNED_MODEL_DIR, "qa_library.json"))
    logger.info("  qa_embeddings.npy: {}", os.path.join(FINE_TUNED_MODEL_DIR, "qa_embeddings.npy"))
    logger.info("  共 {} 条问答 / {} 个主题", len(questions), len(set(knowledge_points)))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
