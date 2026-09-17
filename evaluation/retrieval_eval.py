# -*- coding: utf-8 -*-
"""
检索质量评估 — 康语医疗健康知识库
─────────────────────────────────────────
两类查询模式 × 三种检索配置的对照实验:
  full_q    完整问题检索(问答路由准确率)
  partial_q 截断问题检索(前 15 字, 模拟真实用户的部分信息输入)

  semantic    MacBERT 语义向量 Top-K
  hybrid      语义 + BM25 → RRF 融合
  hybrid_rr   混合检索 + BGE Reranker 精排

指标: Hit@1 / Recall@5 / MRR@10。Gold 文档保留在库内,
衡量"给定问题能否把标准答案排到最前面"。

用法:
  python evaluation/retrieval_eval.py --sample 200 --seed 42
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from config import init_env, TOP_K_SEMANTIC, TOP_K_BM25, TOP_K_RETRIEVAL, RERANKER_MODEL, RERANK_SKIP_CONFIDENCE
init_env()

from logger_config import logger
from retriever.bm25_retriever import BM25Retriever
from retriever.hybrid import HybridRetriever

PARTIAL_LEN = 15
METRIC_K = 10


def batch_encode(retriever, texts, batch_size=8):
    """批量编码, 与 rag_engine.Retriever.encode 相同的 CLS 池化"""
    embs = []
    for i in range(0, len(texts), batch_size):
        inputs = retriever.tokenizer(
            texts[i:i + batch_size], truncation=True, padding=True,
            max_length=256, return_tensors="pt",
        ).to(retriever.device)
        with torch.inference_mode():
            out = retriever.model(**inputs)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)


def semantic_rank(query_emb, corpus_norm, k):
    scores = corpus_norm @ query_emb
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [{"id": str(int(i))} for i in idx]


def metrics(ids, gold):
    ids = ids[:METRIC_K]
    return (
        int(bool(ids) and ids[0] == gold),
        int(gold in ids[:5]),
        1.0 / (ids.index(gold) + 1) if gold in ids else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="康语检索质量评估")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    from rag_engine import Retriever
    retriever = Retriever()
    retriever.load()
    qa = retriever.qa_library
    emb = retriever.qa_embeddings
    n_docs = len(qa["questions"])
    assert emb is not None and emb.shape[0] == n_docs, "向量与问答库不一致, 请先运行 build_kb.py"

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_docs, size=min(args.sample, n_docs), replace=False)
    sample_idx = np.sort(sample_idx)
    queries = [qa["questions"][int(i)] for i in sample_idx]

    # 全量批量编码一次, 显著提速
    logger.info("批量编码 {} 条评测问题...", len(queries))
    q_embs = batch_encode(retriever, queries)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    bm25 = BM25Retriever()
    bm25.build_index(qa)

    reranker = None
    if not args.no_rerank:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("加载 Reranker: {}", RERANKER_MODEL)
            reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        except Exception as e:
            logger.warning("Reranker 不可用, 跳过精排对照: {}", e)

    acc = defaultdict(lambda: {"hit1": 0, "rec5": 0, "rr": 0.0, "n": 0})

    def record(mode, cfg, ids, gold):
        h, r, m = metrics(ids, gold)
        a = acc[(mode, cfg)]
        a["hit1"] += h
        a["rec5"] += r
        a["rr"] += m
        a["n"] += 1

    for pos, gold in enumerate(str(int(i)) for i in sample_idx):
        full_q = queries[pos]
        for mode, query in (("full_q", full_q), ("partial_q", full_q[:PARTIAL_LEN])):
            qemb = q_embs[pos] if mode == "full_q" else batch_encode(retriever, [query])[0]
            if mode == "partial_q":
                qemb = qemb / (np.linalg.norm(qemb) + 1e-9)
            sem = semantic_rank(qemb, emb_norm, TOP_K_SEMANTIC)
            bm = bm25.search(query, TOP_K_BM25)
            merged = HybridRetriever._rrf_fusion(None, sem, bm)[:TOP_K_RETRIEVAL]
            record(mode, "semantic", [d["id"] for d in sem], gold)
            record(mode, "hybrid", [d["id"] for d in merged], gold)
            if reranker is not None:
                cands = merged[:TOP_K_RETRIEVAL]
                top_conf = max((d.get("similarity", 0.0) for d in sem[:1]), default=0.0)
                if top_conf >= RERANK_SKIP_CONFIDENCE:
                    reranked = cands  # 与生产一致: 精确命中直通
                else:
                    pairs = [(query, qa["questions"][int(d["id"])]) for d in cands]
                    scores = reranker.predict(pairs)
                    order = np.argsort(scores)[::-1]
                    reranked = [cands[int(i)] for i in order]
                record(mode, "hybrid_rr", [d["id"] for d in reranked], gold)

    rows = [
        {"mode": m, "config": c, "n": a["n"],
         "hit@1": round(a["hit1"] / a["n"], 4),
         "recall@5": round(a["rec5"] / a["n"], 4),
         "mrr@10": round(a["rr"] / a["n"], 4)}
        for (m, c), a in sorted(acc.items())
    ]

    print("\n===== 康语 · 检索质量评估 =====")
    print(f"知识库: {n_docs} 条 | 抽样: {len(sample_idx)} 条 | seed={args.seed}")
    print(f"{'mode':<10}{'config':<12}{'Hit@1':>8}{'Recall@5':>10}{'MRR@10':>9}")
    for r in rows:
        print(f"{r['mode']:<10}{r['config']:<12}{r['hit@1']:>8.3f}{r['recall@5']:>10.3f}{r['mrr@10']:>9.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "kb_size": n_docs, "sample": len(sample_idx), "seed": args.seed,
        "embed_model": "chinese-macbert-base (CLS pooling)",
        "reranker": RERANKER_MODEL if reranker is not None else "skipped",
        "results": rows,
    }
    out_path = os.path.join(out_dir, "retrieval_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
