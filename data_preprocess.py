"""
QACP 数据集预处理脚本
将 QACP_Q.json 和 QACP_A.csv 合并为统一的训练数据格式
"""
import json
import csv
import os
from sklearn.model_selection import train_test_split
from config import RAW_DATA_DIR, DATA_DIR
from logger_config import logger


def load_and_merge_data():
    with open(os.path.join(RAW_DATA_DIR, "QACP_Q.json"), "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    answers = {}
    with open(os.path.join(RAW_DATA_DIR, "QACP_A.csv"), "r", encoding="gbk") as f:
        for row in csv.DictReader(f):
            answers[int(row["id"])] = row["Answer"]

    merged = []
    for q in questions_data:
        if q["id"] in answers:
            merged.append({
                "id": q["id"],
                "question": q["Question"].strip(),
                "answer": answers[q["id"]].strip(),
                "knowledge_point": q["Knowledge Point"].strip(),
                "question_type": q["Question Type"].strip(),
                "answer_type": q["Answer Type"].strip(),
            })
    return merged


def preprocess_data(data):
    return [
        item for item in data
        if item["question"] and item["answer"]
        and len(item["question"]) >= 3 and len(item["answer"]) >= 10
    ]


def main():
    logger.info("正在加载和合并数据集...")
    data = load_and_merge_data()
    logger.info("合并完成，共 {} 条问答对", len(data))

    data = preprocess_data(data)
    logger.info("清洗后共 {} 条有效问答对", len(data))

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "full_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    with open(os.path.join(DATA_DIR, "train_data.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "test_data.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info("训练集: {} 条, 测试集: {} 条", len(train_data), len(test_data))
    logger.info("数据已保存至: {}", DATA_DIR)

    kp_stats: dict[str, int] = {}
    for item in data:
        kp = item["knowledge_point"]
        kp_stats[kp] = kp_stats.get(kp, 0) + 1
    logger.info("知识点分布:")
    for kp, count in sorted(kp_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info("  {}: {} 条", kp, count)

    logger.info("数据集: https://github.com/NTAIX/Chinese-Python-QA-Dataset")
    logger.info("论文:   https://arxiv.org/abs/2402.07913")


if __name__ == "__main__":
    main()
