# ── 康语 · 医疗健康知识智能问答系统 (RAG v2 生产级镜像) ──────
# 多阶段构建 + 非 root 用户 + 健康检查
#
# 构建: docker build -t kangyu-rag:v2 .
# GPU:  docker run --gpus all -p 5000:5000 kangyu-rag:v2
# CPU:  docker run -p 5000:5000 -e RAG_MODE=retrieval kangyu-rag:v2

# ── Stage 1: Builder ───────────────────────────────────
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# 从 builder 复制 Python 包
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# 复制项目文件
COPY --chown=appuser:appuser config.py .
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser rag_engine.py .
COPY --chown=appuser:appuser llm_engine.py .
COPY --chown=appuser:appuser logger_config.py .
COPY --chown=appuser:appuser prompts.py .
COPY --chown=appuser:appuser conversation.py .
COPY --chown=appuser:appuser feedback.py .
COPY --chown=appuser:appuser train.py .
COPY --chown=appuser:appuser data_preprocess.py .
COPY --chown=appuser:appuser retriever/ retriever/
COPY --chown=appuser:appuser templates/ templates/
COPY --chown=appuser:appuser static/ static/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser model/ model/

# 数据目录权限
RUN mkdir -p /app/data/chroma_db /app/logs && chown -R appuser:appuser /app

EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

COPY --chown=appuser:appuser docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
ENTRYPOINT ["/entrypoint.sh"]
