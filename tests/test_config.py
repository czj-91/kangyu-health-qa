"""配置管理单元测试"""
import os
import pytest


class TestConfigDefaults:
    def test_default_values(self, monkeypatch):
        """验证核心配置项的默认值符合预期"""
        # 清除本地 .env / 环境变量影响，保证读取的是代码内默认值
        for key in (
            "LLM_MAX_NEW_TOKENS", "TOP_K_RETRIEVAL", "MAX_SEQ_LENGTH",
            "API_TIMEOUT", "MAX_CONVERSATION_TURNS", "CONVERSATION_TTL_SECONDS",
            "ENABLE_QUERY_REWRITE",
        ):
            monkeypatch.delenv(key, raising=False)
        # 屏蔽 .env 加载，避免污染默认值断言
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

        # 重新导入以获取默认值
        import importlib
        import config
        importlib.reload(config)

        assert config.LLM_MAX_NEW_TOKENS == 800, "LLM token 限制应为 800"
        assert config.TOP_K_RETRIEVAL == 5, "检索结果数应为 5"
        assert config.MAX_SEQ_LENGTH == 256, "序列长度应为 256"
        assert config.API_TIMEOUT == 120, "API 超时应为 120"
        assert config.MAX_CONVERSATION_TURNS == 10, "最大对话轮次应为 10"
        assert config.CONVERSATION_TTL_SECONDS == 7200, "会话 TTL 应为 7200"
        assert config.ENABLE_QUERY_REWRITE is True, "查询改写应默认开启"

    def test_top_k_limits(self):
        import config
        assert 1 <= config.TOP_K_RETRIEVAL <= 20
        assert 1 <= config.TOP_K_SEMANTIC <= 100
        assert 1 <= config.TOP_K_BM25 <= 100

    def test_port_range(self):
        import config
        assert 1024 <= config.PORT <= 65535

    def test_rag_mode_valid(self):
        import config
        assert config.RAG_MODE in ("auto", "rag", "retrieval")

    def test_llm_backend_valid(self):
        import config
        assert config.LLM_BACKEND in ("local", "api")

    def test_temperature_range(self):
        import config
        assert 0.0 <= config.LLM_TEMPERATURE <= 2.0


class TestConfigEnvOverrides:
    def test_env_override_max_tokens(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_NEW_TOKENS", "1024")
        import importlib
        import config
        importlib.reload(config)
        assert config.LLM_MAX_NEW_TOKENS == 1024

    def test_env_override_top_k(self, monkeypatch):
        monkeypatch.setenv("TOP_K_RETRIEVAL", "8")
        import importlib
        import config
        importlib.reload(config)
        assert config.TOP_K_RETRIEVAL == 8

    def test_env_override_host_port(self, monkeypatch):
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "8080")
        import importlib
        import config
        importlib.reload(config)
        assert config.HOST == "127.0.0.1"
        assert config.PORT == 8080

    def test_env_override_llm_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "api")
        import importlib
        import config
        importlib.reload(config)
        assert config.LLM_BACKEND == "api"

    def test_env_override_boolean(self, monkeypatch):
        monkeypatch.setenv("ENABLE_RERANKER", "false")
        monkeypatch.setenv("ENABLE_QUERY_REWRITE", "false")
        import importlib
        import config
        importlib.reload(config)
        assert config.ENABLE_RERANKER is False
        assert config.ENABLE_QUERY_REWRITE is False


class TestInitEnv:
    def test_hf_endpoint_set(self, monkeypatch):
        # 清除已有环境变量
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        import importlib
        import config
        importlib.reload(config)
        config.init_env()
        assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"

    def test_transformers_offline_set(self, monkeypatch):
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        import importlib
        import config
        importlib.reload(config)
        config.init_env()
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
