"""
集中配置管理 — 所有路径和参数均可通过环境变量覆盖
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def init_env():
    """初始化环境变量 — 在项目入口处调用一次即可"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 不再强制限制线程数，让 PyTorch 自动管理
    # 如需手动限制，设置 OMP_NUM_THREADS / MKL_NUM_THREADS 环境变量
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 离线模式，避免 HuggingFace 连接超时

    # 加载 .env 文件（如果 python-dotenv 可用）
    try:
        from dotenv import load_dotenv
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

# ── 嵌入模型 ─────────────────────────────────────────────
# 医疗健康领域默认使用基础 chinese-macbert-base 作为语义编码器。
# 如需更优检索效果，可在健康数据集上微调（见 train.py），微调模型输出到下方目录。
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "hfl/chinese-macbert-base")
FINE_TUNED_MODEL_DIR = os.getenv("FINE_TUNED_MODEL_DIR", str(ROOT_DIR / "model" / "health_macbert"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))

# ── LLM 生成模型 ─────────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "800"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_LOAD_INT8 = os.getenv("LLM_LOAD_INT8", "true").lower() == "true"

# ── LLM 后端选择 ─────────────────────────────────────────
# "local": 本地 transformers 模型 (Qwen2.5-1.5B)
# "api":   远程 API (OpenAI 兼容接口 — 支持硅基流动、DeepSeek、OpenAI 等)
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")

# API 后端配置 (LLM_BACKEND="api" 时生效)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("API_KEY", "")
API_MODEL = os.getenv("API_MODEL", "Qwen/Qwen2.5-7B-Instruct")
API_MAX_TOKENS = int(os.getenv("API_MAX_TOKENS", "512"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))

# RAG 模式: "rag" | "retrieval" | "auto"
RAG_MODE = os.getenv("RAG_MODE", "auto")

# ── 混合检索 ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "data" / "chroma_db"))
TOP_K_SEMANTIC = int(os.getenv("TOP_K_SEMANTIC", "10"))      # 语义检索召回数
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))              # BM25 召回数
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))     # 最终返回数
RRF_K = int(os.getenv("RRF_K", "60"))                        # RRF 融合常数

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
# 语义最高分 >= 该阈值时视为精确命中, 跳过精排(精排对长答案反而会引入噪声)
RERANK_SKIP_CONFIDENCE = float(os.getenv("RERANK_SKIP_CONFIDENCE", "0.98"))

# ── 查询重写 ─────────────────────────────────────────────
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"

# ── 多轮对话 ─────────────────────────────────────────────
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "7200"))
CONVERSATION_BACKEND = os.getenv("CONVERSATION_BACKEND", "memory")  # "memory" | "chroma"

# ── 数据 ─────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", str(ROOT_DIR / "data" / "processed"))
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", str(ROOT_DIR / "data" / "Health-QA-Dataset"))
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", str(ROOT_DIR / "data" / "feedback.jsonl"))

# ── 服务 ─────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ── 日志 ─────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "structured")  # "structured" | "pretty"
