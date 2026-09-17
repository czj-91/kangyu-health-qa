#!/bin/bash
set -e

echo "============================================"
echo "康语 · 医疗健康知识智能问答系统 (RAG 增强版)"
echo "============================================"
echo "HF_ENDPOINT: ${HF_ENDPOINT:-https://hf-mirror.com}"
echo "RAG_MODE:    ${RAG_MODE:-auto}"
echo "LLM_MODEL:   ${LLM_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "DEVICE:      $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "============================================"

# 如果健康问答库不存在且数据已就绪，自动构建知识库
if [ ! -f /app/model/health_macbert/qa_library.json ]; then
    if [ -f /app/data/Health-QA-Dataset/health_qa.json ]; then
        echo ""
        echo "[entrypoint] 检测到健康数据集但无问答库，自动运行 build_kb.py 构建..."
        python build_kb.py
        echo "[entrypoint] 知识库构建完成。"
    else
        echo "[entrypoint] 警告: 未找到健康数据集，跳过构建。"
    fi
fi

echo ""
echo "[entrypoint] 启动 FastAPI 服务..."
exec uvicorn app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-5000}" --log-level "${LOG_LEVEL:-info}"
