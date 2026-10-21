# -*- coding: utf-8 -*-
"""Agentic RAG 对比评测 — 固定管线 vs Function Calling 决策循环。

与 retrieval_eval.py 同一抽样口径 (seed 可对齐既有检索指标)：
  full_q    完整问题
  partial_q 截断问题(前 N 字，模拟部分信息输入 —— Agent 可通过改写重检自救)

指标:
  pipeline: Hit@1 / Recall@5 + 运行模式分布
  agent:    gold 进入本轮检索命中集比例 (gold_exposed)
            gold 被最终引用比例 (gold_cited)
            回答率 / 澄清率 / 降级率 + 平均步数/tokens/延迟
Agent 评测不带管线兜底(fallback=None)，如实暴露失败率。

断点续跑: 每题每配置一行 JSONL 即时 flush；重跑同一 --tag 自动跳过已完成项。
用法:
  python evaluation/agent_eval.py --tag p15 --sample 40 --seed 42
  python evaluation/agent_eval.py --tag p8  --sample 40 --seed 42 --modes partial --partial-len 8
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import init_env
init_env()

from logger_config import logger

PARTIAL_LEN = 15
TOP_K = 5
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_done(details_path):
    """读取已有明细 → (done_set, rows)"""
    done, rows = set(), []
    if os.path.exists(details_path):
        with open(details_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["cfg"], r["mode"], r["idx"]))
                rows.append(r)
    return done, rows


def summarize(rows):
    acc = {}
    for r in rows:
        d = acc.setdefault((r["cfg"], r["mode"]), {})
        for k, v in r.items():
            if k in ("cfg", "mode", "idx", "gold", "query"):
                continue
            d.setdefault(k, []).append(v)
    out = {}
    for (cfg, mode), d in sorted(acc.items()):
        summary = {}
        for k, vals in d.items():
            if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
                summary[k] = round(float(np.mean(vals)), 4)
            else:
                summary[k] = {}
                for v in vals:
                    summary[k][str(v)] = summary[k].get(str(v), 0) + 1
        out[f"{cfg}/{mode}"] = summary
    return out


async def amain(args):
    from agent.cli import MCPBridge, _make_client
    from agent.loop import AgentRunner
    from agent.tools import build_real_kb_tools
    from api_llm import APILLMEngine
    from config import AGENT_API_BASE_URL, AGENT_API_KEY, AGENT_API_MODEL

    if not args.dry_retrieval and not AGENT_API_KEY and not os.environ.get("API_KEY"):
        print("错误: 需要配置 AGENT_API_KEY / API_KEY（Agent 走远程 Function Calling 模型）。",
              file=sys.stderr)
        return 2

    # ── 知识库与抽样 ────────────────────────────────────
    kb_tools, hybrid, retriever = build_real_kb_tools()
    qa = retriever.qa_library
    n_docs = len(qa["questions"])
    rng = np.random.default_rng(args.seed)
    sample_idx = np.sort(rng.choice(n_docs, size=min(args.sample, n_docs), replace=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    details_path = os.path.join(RESULTS_DIR, f"agent_eval_{args.tag}_details.jsonl")
    done, all_rows = load_done(details_path)
    if done:
        print(f"断点续跑: 已有 {len(done)} 条记录，将跳过")

    llm = APILLMEngine(base_url=AGENT_API_BASE_URL, api_key=AGENT_API_KEY,
                       model=AGENT_API_MODEL)  # Agent 与管线共用同一模型, 对比才公平
    engine = None
    if "pipeline" in args.configs:
        from rag_engine import RAGEngine
        engine = RAGEngine(retriever, llm=llm)

    details_f = open(details_path, "a", encoding="utf-8")

    def emit(row):
        details_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        details_f.flush()
        all_rows.append(row)

    async with _make_client(args.transport, kb_tools) as client:
        bridge = MCPBridge(client)
        schemas = await bridge.list_kb_schemas()
        runner = AgentRunner(llm, bridge.invoke, max_steps=args.max_steps,
                             fallback=None, trace_dir=None)

        for pos, idx in enumerate(sample_idx):
            gold = str(int(idx))
            full_q = qa["questions"][int(idx)]
            modes = []
            if "full" in args.modes:
                modes.append(("full_q", full_q))
            if "partial" in args.modes:
                modes.append(("partial_q", full_q[:args.partial_len]))

            todo = [(m, q) for m, q in modes
                    if all((cfg, m, int(idx)) not in done for cfg in args.configs)]
            if not todo:
                continue

            for mode, query in todo:
                if args.dry_retrieval:
                    hits = kb_tools.hybrid_search(query, TOP_K)
                    ids = [r["id"] for r in hits.get("results", [])]
                    emit({"cfg": "dry", "mode": mode, "idx": int(idx), "gold": gold,
                          "hit1": int(bool(ids) and ids[0] == gold), "rec5": int(gold in ids)})
                    continue

                if engine and "pipeline" in args.configs and ("pipeline", mode, int(idx)) not in done:
                    try:
                        out = engine.ask(query, top_k=TOP_K)
                        ids = [r["id"] for r in out.get("results", [])]
                        emit({"cfg": "pipeline", "mode": mode, "idx": int(idx), "gold": gold,
                              "hit1": int(bool(ids) and ids[0] == gold), "rec5": int(gold in ids),
                              "mode_run": out.get("mode", "?")})
                    except Exception as e:
                        logger.warning(f"pipeline 失败 #{idx}: {e}")

                if "agent" in args.configs and ("agent", mode, int(idx)) not in done:
                    kb_tools.reset_session()
                    res = await runner.run_async(query, schemas)
                    emit({"cfg": "agent", "mode": mode, "idx": int(idx), "gold": gold,
                          "answered": int(res.final_type == "answer"),
                          "clarified": int(res.final_type == "clarification"),
                          "degraded": int(res.final_type == "degraded"),
                          "gold_exposed": int(gold in res.exposed_ids),
                          "gold_cited": int(gold in res.evidence_ids),
                          "steps": res.steps, "tokens": res.tokens_used,
                          "latency_ms": res.latency_ms,
                          "final_type": res.final_type,
                          "cited": res.evidence_ids})

            print(f"  进度 {pos + 1}/{len(sample_idx)}", flush=True)

    details_f.close()

    # ── 汇总（含断点前的历史行）──────────────────────────
    results = summarize(all_rows)
    report = {"sample": len(sample_idx), "seed": args.seed, "top_k": TOP_K,
              "partial_len": args.partial_len, "configs": args.configs,
              "n_rows": len(all_rows), "results": results}
    print(f"\n===== 康语 · Agentic RAG 对比评测 [{args.tag}] (n={len(sample_idx)}, seed={args.seed}) =====")
    for key, summary in results.items():
        print(f"\n[{key.replace('/', ' / ')}]")
        for k, v in summary.items():
            print(f"  {k:<14} {v}")

    out_path = os.path.join(RESULTS_DIR, f"agent_eval_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="康语 Agentic RAG 对比评测")
    p.add_argument("--tag", required=True, help="本轮评测的标识（决定输出文件名，同名可续跑）")
    p.add_argument("--sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", nargs="+", default=["pipeline", "agent"],
                   choices=["pipeline", "agent"])
    p.add_argument("--modes", nargs="+", default=["full", "partial"], choices=["full", "partial"])
    p.add_argument("--partial-len", type=int, default=PARTIAL_LEN)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--transport", choices=["inproc", "stdio"], default="inproc")
    p.add_argument("--dry-retrieval", action="store_true", help="不调 LLM，仅验证抽样+检索链路")
    args = p.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
