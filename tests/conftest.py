"""pytest 全局配置 — 强制离线模式 + 编码设置"""
import os

# 必须在所有 import 之前设置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"
