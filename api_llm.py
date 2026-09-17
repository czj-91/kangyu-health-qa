"""
远程 API LLM 引擎 — 支持所有 OpenAI 兼容接口（硅基流动、DeepSeek、Groq、OpenAI 等）

用法:
  export LLM_BACKEND=api
  export API_BASE_URL=https://api.siliconflow.cn/v1
  export API_KEY=sk-xxxxxxxx
  export API_MODEL=Qwen/Qwen2.5-7B-Instruct

兼容的国内服务商:
  - 硅基流动:  https://api.siliconflow.cn/v1
  - DeepSeek:   https://api.deepseek.com/v1
  - 阿里百炼:   https://dashscope.aliyuncs.com/compatible-mode/v1
  - 智谱:       https://open.bigmodel.cn/api/paas/v4
"""
import json
import requests
from config import (
    API_BASE_URL, API_KEY, API_MODEL, API_MAX_TOKENS, API_TIMEOUT,
    LLM_TEMPERATURE, LLM_TOP_P,
)
from logger_config import logger


class APIGenerator:
    """封装流式响应的迭代器"""

    def __init__(self, response: requests.Response):
        self.response = response
        self._iter = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            try:
                line = next(self._iter)
            except StopIteration:
                raise

            if not line or not line.startswith(b"data: "):
                continue

            data_str = line[6:].decode("utf-8")
            if data_str.strip() == "[DONE]":
                raise StopIteration

            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
            except json.JSONDecodeError:
                continue

            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                return content

            # 非流式回退：有些服务会在第一个 chunk 直接返回完整 content
            message = choices[0].get("message", {})
            fallback = message.get("content", "")
            if fallback:
                return fallback

    def close(self):
        self.response.close()


class APILLMEngine:
    """远程 API LLM — 与本地 LLMEngine 接口完全一致"""

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.model = model or API_MODEL
        self._loaded = bool(self.api_key)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_messages(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    def _build_body(self, messages: list[dict], stream: bool = False) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": API_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "stream": stream,
        }
        if LLM_TEMPERATURE == 0.0:
            body["temperature"] = 0.0
            body["do_sample"] = False
        return body

    def generate(self, prompt: str) -> str:
        """同步生成"""
        messages = self._build_messages(prompt)
        body = self._build_body(messages, stream=False)

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.error(f"API 请求超时 ({API_TIMEOUT}s)")
            raise RuntimeError(f"API 请求超时，请检查网络或增加 API_TIMEOUT")
        except requests.exceptions.RequestException as e:
            logger.error(f"API 请求失败: {e}")
            raise RuntimeError(f"API 请求失败: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"API 响应格式异常: {e}")
            raise RuntimeError(f"API 响应格式异常: {e}")

    def generate_stream(self, prompt: str):
        """流式生成"""
        messages = self._build_messages(prompt)
        body = self._build_body(messages, stream=True)

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=API_TIMEOUT,
                stream=True,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"API 流式请求超时 ({API_TIMEOUT}s)")
            yield f"[错误] API 请求超时"
            return
        except requests.exceptions.RequestException as e:
            logger.error(f"API 流式请求失败: {e}")
            yield f"[错误] API 请求失败: {e}"
            return

        gen = APIGenerator(resp)
        try:
            for token in gen:
                yield token
        except Exception as e:
            logger.error(f"流式读取异常: {e}")
        finally:
            gen.close()

    def load(self):
        """API 模式无需加载模型，仅验证配置"""
        if not self.api_key:
            logger.warning(
                "未设置 API_KEY。请在 .env 或环境变量中配置:\n"
                "  export API_KEY=sk-your-key\n"
                "  export LLM_BACKEND=api"
            )
            self._loaded = False
            return

        # 快速连通性测试
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"API 连接成功: {self.base_url} (模型: {self.model})")
            else:
                logger.warning(f"API 端点返回 {resp.status_code}，但仍将尝试使用")
        except Exception:
            logger.info(f"API 端点连通性测试跳过，将在首次请求时验证")

        self._loaded = True
        logger.info(f"API LLM 引擎就绪 — {self.model} @ {self.base_url}")


# ── 全局单例 ────────────────────────────────────────────
_api_llm_instance: APILLMEngine | None = None


def get_api_llm() -> APILLMEngine:
    global _api_llm_instance
    if _api_llm_instance is not None:
        return _api_llm_instance
    _api_llm_instance = APILLMEngine()
    _api_llm_instance.load()
    return _api_llm_instance
