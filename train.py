"""
MacBERT 模型微调脚本
使用 hfl/chinese-macbert-base 在 QACP 数据集上进行微调
实现知识问答检索系统
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

from config import (
    init_env, DATA_DIR, FINE_TUNED_MODEL_DIR, EMBED_MODEL_NAME, MAX_SEQ_LENGTH,
)
from logger_config import logger

init_env()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("使用设备: {}", device)


class QADataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length=MAX_SEQ_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.label_encoder.transform([item["knowledge_point"]])[0]
        encoding = self.tokenizer(
            item["question"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data():
    with open(os.path.join(DATA_DIR, "train_data.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(DATA_DIR, "test_data.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(os.path.join(DATA_DIR, "full_dataset.json"), "r", encoding="utf-8") as f:
        full_data = json.load(f)
    return train_data, test_data, full_data


def train_model():
    logger.info("=" * 60)
    logger.info("开始 MacBERT 模型微调")
    logger.info("=" * 60)

    train_data, test_data, full_data = load_data()
    logger.info("数据加载完成 - 训练集: {}, 测试集: {}", len(train_data), len(test_data))

    all_kps = list(set(item["knowledge_point"] for item in full_data))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_kps)
    num_labels = len(all_kps)
    logger.info("知识点类别数: {}", num_labels)

    label_map = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    with open(os.path.join(FINE_TUNED_MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    logger.info("加载预训练模型: {}", EMBED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        EMBED_MODEL_NAME, num_labels=num_labels,
    )

    train_dataset = QADataset(train_data, tokenizer, label_encoder)
    test_dataset = QADataset(test_data, tokenizer, label_encoder)

    training_args = TrainingArguments(
        output_dir=os.path.join(FINE_TUNED_MODEL_DIR, "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("开始微调训练...")
    trainer.train()

    logger.info("保存微调后的模型到: {}", FINE_TUNED_MODEL_DIR)
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

    logger.info("模型评估结果:")
    eval_results = trainer.evaluate()
    logger.info("  测试集准确率: {:.4f}", eval_results.get('eval_accuracy', float('nan')))
    logger.info("  测试集损失:   {:.4f}", eval_results.get('eval_loss', float('nan')))

    return model, tokenizer, label_encoder, train_data, test_data, full_data


def build_question_embeddings(model, tokenizer, questions, batch_size=16):
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
            outputs = model.bert(**inputs)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)


def evaluate_topk_retrieval(model, tokenizer, train_data, test_data, k_values=(1, 3, 5)):
    logger.info("=" * 60)
    logger.info("Top-K 检索命中率评估")
    logger.info("=" * 60)

    train_questions = [item["question"] for item in train_data]
    train_kp = [item["knowledge_point"] for item in train_data]
    test_questions = [item["question"] for item in test_data]

    logger.info("正在为训练集 {} 个问题构建嵌入...", len(train_questions))
    train_emb = build_question_embeddings(model, tokenizer, train_questions)
    logger.info("正在为测试集 {} 个问题构建嵌入...", len(test_questions))
    test_emb = build_question_embeddings(model, tokenizer, test_questions)

    train_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    test_norm = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    sim_matrix = np.dot(test_norm, train_norm.T)

    results = {}
    for k in k_values:
        hits = 0
        for i, test_item in enumerate(test_data):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
            retrieved_kps = [train_kp[idx] for idx in top_k_indices]
            if test_item["knowledge_point"] in retrieved_kps:
                hits += 1
        hit_rate = hits / len(test_data)
        results[k] = hit_rate
        logger.info("  Top-{} 命中率: {:.4f} ({}/{})", k, hit_rate, hits, len(test_data))

    with open(os.path.join(FINE_TUNED_MODEL_DIR, "topk_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "k_values": list(k_values),
            "hit_rates": {str(k): float(v) for k, v in results.items()},
            "total_test_samples": len(test_data),
        }, f, ensure_ascii=False, indent=2)

    return results


def save_qa_library(train_data, train_embeddings, train_questions, train_answers, train_kp):
    np.save(os.path.join(FINE_TUNED_MODEL_DIR, "qa_embeddings.npy"), train_embeddings)
    with open(os.path.join(FINE_TUNED_MODEL_DIR, "qa_library.json"), "w", encoding="utf-8") as f:
        json.dump({
            "questions": train_questions,
            "answers": train_answers,
            "knowledge_points": train_kp,
        }, f, ensure_ascii=False, indent=2)
    logger.info("问答库已保存至: {}", FINE_TUNED_MODEL_DIR)
    logger.info("  嵌入向量: qa_embeddings.npy ({})", train_embeddings.shape)
    logger.info("  问答数据: qa_library.json ({} 条)", len(train_questions))


def main():
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)

    model, tokenizer, label_encoder, train_data, test_data, full_data = train_model()

    logger.info("=" * 60)
    logger.info("构建问答库嵌入向量")
    logger.info("=" * 60)

    train_questions = [item["question"] for item in train_data]
    train_answers = [item["answer"] for item in train_data]
    train_kp = [item["knowledge_point"] for item in train_data]

    train_embeddings = build_question_embeddings(model, tokenizer, train_questions)
    logger.info("嵌入向量维度: {}", train_embeddings.shape)

    evaluate_topk_retrieval(model, tokenizer, train_data, test_data)
    save_qa_library(train_data, train_embeddings, train_questions, train_answers, train_kp)

    logger.info("=" * 60)
    logger.info("模型训练与评估完成！")
    logger.info("模型保存路径: {}", FINE_TUNED_MODEL_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
