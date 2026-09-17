**【康语 · 医疗健康知识智能问答系统】** ｜ 基于混合检索与精排门控的 RAG 健康科普问答平台
`PyTorch` `FastAPI` `ChromaDB` `Qwen2.5` `MacBERT` `BGE-Reranker` `Docker` `GitHub Actions`

独立完成整体开发：双 LLM 后端热切换（本地 Qwen2.5-1.5B INT8 量化推理 / 远端 OpenAI 兼容协议，支持 SiliconFlow、DeepSeek 等 5 家 API）、三级降级策略（RAG 生成→检索返回原文→兜底话术）、多轮会话管理（TTL 过期回收 + 线程锁）、SSE 流式 token 级推送、用户反馈 JSONL 持久化等核心模块。

架构：RuntimeCheckable Protocol 接口解耦 LLM 策略，支持运行时类型检查与可插拔替换；混合检索采用 MacBERT 语义嵌入 + ChromaDB HNSW + 字符级 BM25 双路召回，RRF（k=60）融合后送入 BGE CrossEncoder 精排；设计**精排门控**（语义最高分≥0.98 直通、否则以问题文本精排），经评测驱动发现并修复"精排破坏精确匹配"的负优化问题；延迟加载 + 嵌入缓存 + NumPy 类型清洗。

将 5,915 条、23 主题（内科/外科/妇产/儿科/肿瘤/心理/中医养生等）的医疗问答数据集构建为 ChromaDB 知识库；自建检索评测基准（200 条抽样、完整/截断双查询模式、三配置对照），量化验证：不完整输入场景 Hit@1 从 0.37 提升至 0.955（MRR 0.963），同时消除初版精排对精确匹配的负优化（0.74→0.995）；提示词层内置医疗安全护栏（不诊断、急症引导 120、用药遵医嘱），前端常驻免责声明；pytest 92 条测试全绿（43 单测 + 49 集成），GitHub Actions CI，Docker 多阶段非 root 构建 + 健康检查。

