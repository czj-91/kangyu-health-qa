**【康语 · 医疗健康知识智能问答系统】** ｜ 基于混合检索与精排门控的 RAG 健康科普问答平台
`PyTorch` `FastAPI` `ChromaDB` `Qwen2.5` `MacBERT` `BGE-Reranker` `MCP` `Docker` `GitHub Actions`

独立完成整体开发：双 LLM 后端热切换（本地 Qwen2.5-1.5B INT8 量化推理 / 远端 OpenAI 兼容协议，支持 SiliconFlow、DeepSeek 等 5 家 API）、三级降级策略（RAG 生成→检索返回原文→兜底话术）、多轮会话管理（TTL 过期回收 + 线程锁）、SSE 流式 token 级推送、用户反馈 JSONL 持久化等核心模块。

架构：runtime_checkable Protocol（LLMLike）接口解耦 LLM 策略，支持运行时类型检查与可插拔替换；混合检索采用 MacBERT 语义嵌入 + ChromaDB HNSW + 字符级 BM25 双路召回，RRF（k=60）融合后送入 BGE CrossEncoder 精排；设计**精排门控**（语义最高分≥0.98 直通、否则以问题文本精排），经评测驱动发现并修复"精排破坏精确匹配"的负优化问题；延迟加载 + 嵌入缓存 + NumPy 类型清洗。

将 5,915 条、23 主题（内科/外科/妇产/儿科/肿瘤/心理/中医养生等）的医疗问答数据集构建为 ChromaDB 知识库；自建检索评测基准（200 条抽样、完整/截断双查询模式、三配置对照），量化验证：不完整输入场景 Hit@1 从 0.37 提升至 0.955（MRR 0.963），同时消除初版精排对精确匹配的负优化（0.74→0.995）；提示词层内置医疗安全护栏（不诊断、急症引导 120、用药遵医嘱），前端常驻免责声明；pytest 185 条测试全绿（136 单测 + 49 集成），GitHub Actions CI，Docker 多阶段非 root 构建 + 健康检查。

Agentic RAG（MCP 工具化 + Function Calling）：将混合检索/原文读取封装为标准 MCP Server（stdio，任意 MCP 客户端可接入），Agent 循环经 Function Calling 自主决定改写重检、读文取证与终止方式；工程约束含步数/token/墙钟三重预算、强制终止阶段仅开放控制工具、finalize_answer 证据归属校验（引用 id 须为本会话真实检索命中，Pydantic 契约拒绝回喂自纠）、服务端检索白名单防任意文档读取；任何失败路径统一降级固定管线。真实模型（deepseek-chat）n=40 对比评测：120 次运行零降级；8 字截断难例下固定管线 Hit@1 0.825 仍强行作答，Agent 20% 主动澄清避免答错；评测同时揭示简单查询下 agentic 化属负收益（改写引入检索漂移 + 5~10 倍 token），故生产定位为非歧义输入默认管线、Agent 模式仅用于低置信场景（详见 README 评测章节）。

