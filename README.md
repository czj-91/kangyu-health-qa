# 康语 · 医疗健康知识智能问答系统

![CI](https://github.com/czj-91/kangyu-health-qa/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

基于 **RAG（检索增强生成）** 架构的医疗健康与养生知识问答系统。面向普通大众，帮助大家在日常生活中更好地了解健康养生与基础医学常识，支持混合检索、流式输出、多轮对话。

> ⚠️ **健康声明**：本系统提供的内容仅为**健康科普与养生知识参考**，**不构成医疗诊断、治疗建议或处方**，也**不能替代执业医师的面诊**。涉及具体病情、用药、剂量、孕期/婴幼儿/慢病等个体情况，请前往正规医疗机构就诊或咨询医生、药师。如出现急症（剧烈胸痛、呼吸困难、意识障碍、大出血、疑似中风/心梗等），请立即拨打 **120** 急救。

## 架构概览

```mermaid
flowchart LR
    Q[用户问题] --> RW[查询改写<br/>可选]
    RW --> SEM[MacBERT 语义向量<br/>ChromaDB HNSW]
    RW --> BM[字符级 BM25]
    SEM --> RRF[RRF 融合 k=60]
    BM --> RRF
    RRF --> RR[BGE Reranker 精排]
    RR --> P[Prompt 组装<br/>医疗安全护栏]
    P --> LLM{LLM 后端}
    LLM -->|local| H[Qwen2.5 INT8]
    LLM -->|api| S[OpenAI 兼容 API]
    LLM -->|不可用自动降级| RET[返回检索原文]
    H --> SSE[SSE 流式输出]
    S --> SSE
```

```
用户问题 → 查询改写(可选) → 混合检索(语义+BM25) → RRF融合 → BGE Reranker精排 → LLM生成 → 流式输出
```

## 核心特性

| 特性 | 说明 |
|---|---|
| 混合检索 | MacBERT 语义向量 + 字符级 BM25 双路召回，RRF(k=60) 融合，BGE CrossEncoder 门控精排（高置信精确命中直通） |
| 双 LLM 后端 | 本地 Qwen2.5-1.5B INT8 量化推理 / 远程 OpenAI 兼容 API（SiliconFlow、DeepSeek、百炼等），运行时热切换 |
| 三级降级 | RAG 生成 → 检索返回原文 → 兜底话术，LLM 不可用时服务不中断 |
| 输入守卫 | 规则级识别非健康领域/无意义输入，即时返回提问引导（不消耗检索与 LLM 推理）；含健康信号则一律放行，多轮追问豁免，实测零误伤 |
| 多轮对话 | 会话 TTL 过期回收 + 线程锁；查询改写消除指代歧义（可选） |
| 流式输出 | SSE token 级推送，类 ChatGPT 打字机体验 |
| 医疗安全护栏 | Prompt 层不诊断、不替代就医、急症引导 120、用药遵医嘱；前端常驻免责声明 |
| 知识库 | 5,915 条医疗问答 / 23 个主题（内科、外科、妇产、儿科、肿瘤、心理、中医养生等） |
| Agentic RAG | 检索/读全文工具封装为标准 **MCP Server**（stdio），Agent 经 Function Calling 自主决策重检索与改写；步数/token/墙钟三重预算 + 证据归属校验，超限自动降级回固定管线 |
| 可观测 | Loguru 结构化日志、用户反馈 JSONL 持久化与统计、Agent 决策轨迹 JSONL |
| 工程质量 | pytest 185 条测试（136 单测 + 49 集成）、GitHub Actions CI、Docker 多阶段非 root 构建 + 健康检查 |

## 检索质量评测

`evaluation/retrieval_eval.py` 对知识库做抽样 holdout 评测（gold 保留在库内，衡量"给定问题能否把标准答案排进最前列"）：

- **full_q**：用完整问题检索（问答路由准确率）
- **partial_q**：仅用问题前 15 字检索（模拟用户不完整输入，更接近真实场景）

| 查询模式 | 检索配置 | Hit@1 | Recall@5 | MRR@10 |
|---|---|---:|---:|---:|
| full_q（完整问题） | 仅语义向量 | 0.990 | 1.000 | 0.994 |
| full_q | 混合检索 + RRF | 0.995 | 1.000 | 0.998 |
| full_q | 混合 + 门控精排（当前线上配置） | **0.995** | 1.000 | **0.998** |
| partial_q（前 15 字） | 仅语义向量 | 0.220 | 0.335 | 0.270 |
| partial_q | 混合检索 + RRF | 0.370 | 0.970 | 0.645 |
| partial_q | 混合 + 门控精排（当前线上配置） | **0.955** | **0.970** | **0.963** |

**评测驱动的两个关键结论**：
1. 真实用户常输入不完整问题（partial_q），纯语义检索在此场景几乎不可用（Hit@1 仅 0.22），混合召回 + 精排是必需的；
2. 初版精排以"长答案原文"打分且无条件执行，导致精确匹配场景 Hit@1 从 0.995 劣化到 0.740 —— 修复为**问题文本对齐打分 + 语义置信度 ≥0.98 直通**后，两种场景同时达到最优。

> 复现命令：`python evaluation/retrieval_eval.py --sample 200 --seed 42`，报告输出至 `evaluation/results/retrieval_eval.json`。

## Agentic RAG（MCP 工具化 + Function Calling）

在固定管线之外，提供一层**模型自主决策**的检索 Agent，与线上管线共存、互为降级：

- **MCP Server**（`agent/mcp_server.py`）：把 `hybrid_search` / `get_document` / `kb_stats` 与知识库 resource 封装为标准 MCP 工具，stdio 传输；除本项目 Agent 外，任意 MCP 客户端（Claude Desktop / Cursor 等）可直接接入。
- **决策循环**（`agent/loop.py`）：LLM 通过 Function Calling 自主决定检索次数、改写查询、读全文取证，以 `finalize_answer`（带证据引用）或 `ask_clarification`（澄清反问）终止。
- **确定性护栏**：步数/token/墙钟三重预算，超限先"强制终止"再降级；`finalize_answer` 的 `evidence_ids` 必须属于本会话真实检索命中集（Pydantic 契约 + 证据归属校验，编造 id 会被拒绝回喂修正）；`get_document` 受服务端**检索白名单**约束，防任意文档读取；任何失败路径统一降级到既有三级降级管线，用户永远有答案。
- **决策轨迹**：每轮问答的 assistant 决策 / 工具调用 / 拒绝事件保存为 JSONL（`logs/agent_traces/`），可复盘。
- 注意：Agent 需要支持 Function Calling 的 API 模型（本地 1.5B INT8 不具备，会自动走降级链，属预期行为）。

```bash
# MCP 工具链路自检（不调 LLM）
python -m agent.cli --no-llm --transport stdio

# 单次提问 / 交互模式（需 API_KEY 或 AGENT_API_KEY）
python -m agent.cli -q "感冒了怎么办"
python -m agent.cli -i
```

对比评测（与检索评测同口径抽样，截断输入正是 Agent 改写重检的收益场景，支持断点续跑）：

```bash
python evaluation/agent_eval.py --tag p15 --sample 40 --seed 42                       # 管线 vs Agent
python evaluation/agent_eval.py --tag p8  --sample 40 --seed 42 --modes partial --partial-len 8
```

### 验证状态（2026-10-20，真实模型：deepseek-chat）

端到端演示（CLI，轨迹保存于 `logs/agent_traces/`）：

| 场景 | 行为 | 结果 |
|---|---|---|
| 正常题「感冒流鼻涕低烧」 | 5 步 8 次工具调用：并行改写双检索 → 读 4 篇全文 → 追加检索就医指征 | 引用 4 条证据，不诊断、建议药师/医生 |
| 急症题「剧烈胸痛+呼吸困难」 | 2 步快速收敛 | 首句引导拨打 120，未给自行处理方案 |
| 截断题「宝宝发烧38度」 | 4 步，含改写重检 | 给出体温监测记录与就诊建议 |

对比评测初步结果（n=40 抽样、full+partial 双模式；因 API 账户余额中断，Agent 侧为**中断前有效的 44 条运行**，数字口径如实标注，完整重跑待补）：

| 配置 | 指标 | 数值 |
|---|---|---|
| pipeline（固定管线） | Hit@1 / Recall@5（纯检索指标，全样本有效） | 1.00 / 1.00 |
| agent（有效 44 条内） | 自主回答率 | 0.91 |
| agent | gold 进入本轮检索命中集 | 0.91 |
| agent | gold 被最终引用（证据归属通过） | 0.86 |
| agent | 平均步数 / tokens | ≈3 / ≈13k |

> ⚠️ 诚实声明：上表 agent 行样本量小（44）且截断难例（partial8）批次未跑成，**尚不足以宣称 Agent 相对管线的量化增益**；当前能确证的是行为正确性与证据引用质量。充值 API 账户后重跑 `--tag` 即可补齐（断点续跑自动跳过已完成项）。

## 项目结构

```
├── app.py                 # FastAPI 主入口 + API 路由
├── config.py              # 集中配置管理（环境变量可覆盖）
├── rag_engine.py          # RAG 引擎（检索 + 生成编排, Protocol 解耦 LLM 后端）
├── llm_engine.py          # 本地 LLM 推理引擎（transformers, INT8）
├── api_llm.py             # 远程 API LLM 引擎（OpenAI 兼容）
├── conversation.py        # 多轮对话会话管理（TTL + 线程锁）
├── feedback.py            # 用户反馈收集（JSONL）
├── prompts.py             # 健康助手提示词 + 医疗安全护栏
├── train.py               # 历史实验：早期 QACP 知识点问答的分类头微调脚本
├── build_kb.py            # 健康知识库构建脚本（生成向量与问答库）
├── retriever/             # 混合检索: 向量存储 / BM25 / RRF+Rerank
├── agent/                 # Agentic RAG: MCP Server / Function Calling 循环 / CLI
├── evaluation/            # RAGAS 端到端评估 + 检索质量评测 + Agent 对比评测
├── tests/                 # pytest: 单测(快) + integration(加载模型, 慢)
├── templates/             # 前端 SPA（青绿健康主题 + 免责横幅）
├── .github/workflows/     # CI
└── Dockerfile / docker-compose.yml
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 配置文件
```

### 3. 准备健康知识库

知识库源数据为 `data/Health-QA-Dataset/health_qa.json`（内置 5,915 条、23 个主题的医疗问答）。生成向量与问答库：

```bash
python build_kb.py
```

> 说明：线上检索使用基础 MacBERT（`hfl/chinese-macbert-base`）语义嵌入，知识库构建不依赖微调权重。仓库中的 `train.py` / `data_preprocess.py` 为项目早期版本（QACP 知识点问答）的分类头微调实验脚本，需自备 QACP 原始数据，与当前医疗健康知识库无关，仅作训练流程参考。

生成产物位于 `model/health_macbert/`：`qa_library.json` 与 `qa_embeddings.npy`。

### 4. 启动服务

```bash
python app.py
# 访问 http://localhost:5000
# API 文档 http://localhost:5000/docs
```

或使用 Docker：

```bash
docker compose up --build
```

### 5. 运行测试

```bash
pytest -q -m "not integration"   # 快速单测（CI 使用, ~12s）
pytest -q -m integration         # 全链路集成测试（需本地模型权重, 较慢）
```

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LLM_BACKEND` | `local` | LLM 后端：`local` / `api` |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | 本地模型名 |
| `API_BASE_URL` | `https://api.siliconflow.cn/v1` | API 地址 |
| `API_KEY` | - | API 密钥 |
| `API_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | API 模型名 |
| `RAG_MODE` | `auto` | RAG 模式：`rag` / `retrieval` / `auto` |
| `EMBED_MODEL_NAME` | `hfl/chinese-macbert-base` | 嵌入模型 |
| `FINE_TUNED_MODEL_DIR` | `model/health_macbert` | 健康知识库目录（含 qa_library.json） |
| `TOP_K_RETRIEVAL` | `5` | 最终返回数量 |
| `ENABLE_RERANKER` | `true` | 启用 BGE Reranker |
| `AGENT_API_KEY` | 同 `API_KEY` | Agent 专用密钥（需支持 Function Calling 的模型） |
| `AGENT_API_MODEL` / `AGENT_API_BASE_URL` | 同 `API_*` | Agent 专用模型端点 |
| `AGENT_MAX_STEPS` / `AGENT_TOKEN_BUDGET` / `AGENT_WALL_TIMEOUT` | `8` / `12000` / `180` | Agent 三重预算 |
| `DEBUG` | `false` | 调试模式 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查（含模式/设备/知识库规模） |
| POST | `/api/ask` | 同步问答 |
| POST | `/api/ask/stream` | 流式问答（SSE） |
| GET | `/api/sample_questions` | 示例问题 |
| POST | `/api/conversation/new` | 创建会话 |
| POST | `/api/conversation/{id}/clear` | 清除会话 |
| GET | `/api/conversation/{id}` | 获取会话历史 |
| POST | `/api/feedback` | 提交反馈 |
| GET | `/api/feedback/stats` | 反馈统计 |

## 知识库主题分布（5,915 条）

内科常见病 800 ｜ 外科与创伤 800 ｜ 妇产科健康 800 ｜ 皮肤与性病健康 712 ｜ 儿科与儿童保健 657 ｜ 眼耳鼻喉口腔 543 ｜ 肿瘤防治 351 ｜ 心理健康 226 ｜ 中医养生 198 ｜ 传染病预防 193 ｜ 其余 13 类共 635 条

> 内置数据集为**演示用种子知识库**，内容力求准确保守，但**正式上线前应由具备资质的医疗专业人员审核**，并可替换为更权威、更大规模的数据源（如公开医学指南、科普知识库等）。

## 支持的 API 服务商

所有兼容 OpenAI Chat Completions 接口的服务均可使用：

- 硅基流动 `https://api.siliconflow.cn/v1`
- DeepSeek `https://api.deepseek.com/v1`
- 阿里百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 智谱 AI `https://open.bigmodel.cn/api/paas/v4`
- Groq `https://api.groq.com/openai/v1`

## 关于 AI 辅助开发

本项目采用 **AI 结对编程**工作流（Claude / Codex 等编码助手 + 人工审查）：架构设计、关键路径取舍（RRF vs 加权融合、降级策略、接口协议解耦）、安全护栏规则、测试用例评审与全部缺陷修复由人工主导；样板代码与部分单测由 AI 生成后经人工逐段审查。仓库保留完整提交历史，可追溯每次 AI 产出的人工修正记录。

## 许可证

MIT

