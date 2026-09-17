"""
FastAPI Web 后端 — 康语 · 医疗健康知识智能问答系统 (RAG 增强版 v2)
自带 Swagger 文档: http://localhost:5000/docs
"""
import json as json_module
import threading
from contextlib import asynccontextmanager

from config import init_env
init_env()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from typing import Any
from pydantic import BaseModel, Field

from config import HOST, PORT, DEBUG, TOP_K_RETRIEVAL, RAG_MODE, ROOT_DIR, LLM_BACKEND
from rag_engine import Retriever, RAGEngine, LLMLike
from llm_engine import get_llm
from api_llm import get_api_llm
from conversation import conv_manager
from feedback import save_feedback, get_feedback_stats
from logger_config import logger

# ── 全局组件 ────────────────────────────────────────────
_init_lock = threading.Lock()
retriever: Retriever | None = None
rag_engine: RAGEngine | None = None
llm: LLMLike | None = None


def _assert_ready():
    """确保全局组件已初始化，未就绪时抛出 500"""
    if retriever is None or rag_engine is None:
        raise HTTPException(500, "系统组件尚未初始化，请稍后重试")


def init_components():
    global retriever, rag_engine, llm
    # 双监听(IPv4 主进程 + IPv6 回环线程)会并发触发两次 lifespan,
    # 用锁序列化, 避免模型/ChromaDB 重复初始化竞争
    with _init_lock:
        if rag_engine is not None:
            logger.info("组件已初始化，跳过重复加载")
            return
        logger.info("=" * 60)
        logger.info("初始化 康语 · 医疗健康知识智能问答系统 v2")
        logger.info(f"LLM 后端: {LLM_BACKEND.upper()}")
        logger.info("=" * 60)

        retriever = Retriever()
        assert retriever is not None
        retriever.load()

        # LLM 加载：支持本地模型和远程 API 两种后端
        if RAG_MODE == "rag":
            logger.info("RAG 模式，正在加载 LLM...")
            llm = _load_llm()
        else:
            logger.info("LLM 将在首次 RAG 请求时延迟加载 (当前模式: {})", RAG_MODE)
            if LLM_BACKEND == "api":
                llm = _load_llm()

        rag_engine = RAGEngine(retriever, llm)
        logger.info("初始化完成！")


def _load_llm() -> LLMLike | None:
    """根据 LLM_BACKEND 配置加载对应的引擎"""
    if LLM_BACKEND == "api":
        logger.info("使用远程 API 后端")
        return get_api_llm()
    else:
        logger.info("使用本地 transformers 模型")
        return get_llm()


def ensure_llm() -> LLMLike | None:
    """延迟加载 LLM — 在首次 RAG 请求时调用"""
    global llm, rag_engine
    if llm is not None and llm.loaded:
        return llm
    logger.info("延迟加载 LLM...")
    llm = _load_llm()
    if llm is not None and rag_engine is not None:
        rag_engine.llm = llm
        rag_engine.resolve_mode()
    return llm


def _resolve_request_mode(req_mode: str, engine) -> None:
    """请求级模式解析 — 处理延迟加载和模式切换"""
    if req_mode:
        ensure_llm()
        if engine.llm is not None and engine.llm.loaded:
            engine.effective_mode = req_mode
    elif RAG_MODE == "auto" and (llm is None or not llm.loaded):
        ensure_llm()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_components()
    yield


# ── FastAPI 应用 ────────────────────────────────────────
app = FastAPI(
    title="康语 · 医疗健康知识智能问答系统",
    description="基于 MacBERT + Qwen2.5 的 RAG 检索增强生成问答系统，覆盖日常养生与基础医学科普，支持混合检索、流式输出、多轮对话。内容为健康科普，不替代专业医疗诊断。",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

MAX_BODY_SIZE = 10 * 1024 * 1024


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": "请求体过大，最大允许 10MB"},
        )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic 模型 ───────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=2)
    session_id: str = Field(default="", description="会话ID，空则创建新会话")
    top_k: int = Field(default=TOP_K_RETRIEVAL, description="检索数量", ge=1, le=20)
    mode: str = Field(
        default="",
        description="本次请求运行模式：空=跟随服务端配置 | rag | retrieval",
        pattern="^(|rag|retrieval)$",
    )


class AskResponse(BaseModel):
    answer: str
    mode: str
    session_id: str = ""
    similarity: float | None = None
    knowledge_point: str = ""
    matched_question: str = ""
    sources: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str = Field(..., description="helpful | not_helpful")
    session_id: str = ""
    mode: str = ""


class HealthResponse(BaseModel):
    status: str
    mode: str
    retriever_loaded: bool
    llm_loaded: bool
    hybrid_indexed: bool
    qa_count: int
    device: str
    llm_backend: str = ""


# ── 页面路由 ────────────────────────────────────────────

@app.get("/")
async def index():
    html_path = ROOT_DIR / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "前端页面未找到，请检查 templates/index.html 是否存在")
    content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    logger.error(f"500 Error on {request.url}: {exc}\n{tb}")
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "traceback": tb},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"},
    )


# ── API 路由 ────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "mode": rag_engine.mode if rag_engine else "unknown",
        "retriever_loaded": retriever is not None and retriever.model is not None,
        "llm_loaded": llm is not None and llm.loaded if llm else False,
        "hybrid_indexed": rag_engine.hybrid.vector_store.exists() if rag_engine else False,
        "qa_count": len(retriever.qa_library["questions"]) if retriever is not None and retriever.qa_library else 0,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "llm_backend": LLM_BACKEND,
    }


@app.get("/api/sample_questions")
async def sample_questions():
    if retriever is None:
        return {"samples": []}
    return {"samples": retriever.get_sample_questions()}


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """同步问答接口（非流式）"""
    _assert_ready()
    assert rag_engine is not None

    if not req.question.strip():
        raise HTTPException(400, "请输入问题")
    if len(req.question.strip()) < 2:
        raise HTTPException(400, "问题太短")

    # 会话管理
    session_id = req.session_id or conv_manager.create_session()
    chat_history = conv_manager.get_history(session_id)

    _resolve_request_mode(req.mode, rag_engine)

    result = rag_engine.ask(req.question, chat_history, req.top_k)

    conv_manager.add_turn(session_id, req.question, result.get("answer", ""))

    result["session_id"] = session_id
    return result


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    """流式问答接口 — SSE"""
    _assert_ready()
    assert rag_engine is not None

    if not req.question.strip() or len(req.question.strip()) < 2:
        raise HTTPException(400, "问题太短")

    session_id = req.session_id or conv_manager.create_session()
    chat_history = conv_manager.get_history(session_id)

    _resolve_request_mode(req.mode, rag_engine)

    results = rag_engine.search(req.question, chat_history, req.top_k)

    engine = rag_engine

    def generate():
        if not results:
            yield f"data: {json_module.dumps({'type': 'answer', 'text': '抱歉，未找到相关答案。'})}\n\n"
            yield f"data: {json_module.dumps({'type': 'done'})}\n\n"
            return

        best = results[0]

        # 检索模式
        if engine.effective_mode == "retrieval":
            full_answer = best["answer"]
            chunk_size = 30
            for i in range(0, len(full_answer), chunk_size):
                chunk = full_answer[i:i + chunk_size]
                yield f"data: {json_module.dumps({'type': 'token', 'text': chunk})}\n\n"
            yield f"data: {json_module.dumps({'type': 'done', 'answer': full_answer, 'mode': 'retrieval', 'similarity': best['similarity'], 'knowledge_point': best['knowledge_point'], 'sources': [{'question': r['question'], 'knowledge_point': r['knowledge_point'], 'similarity': r['similarity']} for r in results], 'results': results}, ensure_ascii=False)}\n\n"
            return

        # RAG 流式生成
        from prompts import build_rag_prompt, sanitize_answer, StreamEchoFilter
        prompt = build_rag_prompt(req.question, results, chat_history)
        full_text = ""

        if engine.llm is None:
            yield f"data: {json_module.dumps({'type': 'error', 'text': 'LLM 未加载'})}\n\n"
            return

        echo_filter = StreamEchoFilter()
        try:
            for token in engine.llm.generate_stream(prompt):
                full_text += token
                for safe in echo_filter.feed(token):
                    yield f"data: {json_module.dumps({'type': 'token', 'text': safe})}\n\n"
            for safe in echo_filter.flush():
                yield f"data: {json_module.dumps({'type': 'token', 'text': safe})}\n\n"
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"data: {json_module.dumps({'type': 'error', 'text': str(e)})}\n\n"

        clean_answer = sanitize_answer(full_text)
        yield f"data: {json_module.dumps({'type': 'done', 'answer': clean_answer, 'mode': 'rag', 'similarity': best['similarity'], 'knowledge_point': best['knowledge_point'], 'sources': [{'question': r['question'], 'knowledge_point': r['knowledge_point'], 'similarity': r['similarity']} for r in results], 'results': results}, ensure_ascii=False)}\n\n"

    response = StreamingResponse(generate(), media_type="text/event-stream")
    response.headers["X-Session-Id"] = session_id
    return response


@app.post("/api/conversation/new")
async def new_conversation():
    """创建新会话"""
    sid = conv_manager.create_session()
    return {"session_id": sid}


@app.post("/api/conversation/{session_id}/clear")
async def clear_conversation(session_id: str):
    """清除会话历史"""
    conv_manager.clear(session_id)
    return {"status": "ok"}


@app.get("/api/conversation/{session_id}")
async def get_conversation(session_id: str):
    """获取会话历史"""
    history = conv_manager.get_history_messages(session_id)
    return {"session_id": session_id, "history": history, "turns": len(history) // 2}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """提交用户反馈"""
    save_feedback(
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        mode=req.mode,
        session_id=req.session_id,
    )
    return {"status": "ok"}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """反馈统计"""
    return get_feedback_stats()


# ── 启动 ────────────────────────────────────────────────

def _serve_ipv6_loopback() -> None:
    """额外监听 IPv6 回环 (::1)。

    Windows 上浏览器解析 localhost 时优先尝试 ::1，若服务只绑定
    0.0.0.0 会出现"拒绝连接"。本线程让 http://localhost:PORT 可用，
    绑定失败不影响主 IPv4 服务。
    """
    try:
        cfg = uvicorn.Config(app, host="::1", port=PORT, log_level="warning")
        uvicorn.Server(cfg).run()
    except OSError as e:
        logger.warning("IPv6 回环端口绑定失败（不影响 IPv4 访问）: {}", e)


if __name__ == "__main__":
    import threading
    import uvicorn
    logger.info(f"启动服务: http://localhost:{PORT}  (或 http://127.0.0.1:{PORT})")
    logger.info(f"Swagger:  http://localhost:{PORT}/docs")
    threading.Thread(target=_serve_ipv6_loopback, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info" if not DEBUG else "debug")
