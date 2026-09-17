"""
LLM 推理引擎 — 加载 Qwen2.5 等开源模型，支持 GPU/CPU 推理 + 流式输出
"""
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import init_env, LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, LLM_LOAD_INT8
from logger_config import logger

init_env()
# 线程数由 PyTorch 自动管理，不再硬编码限制


class LLMEngine:
    """封装开源 LLM 的加载与推理"""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or LLM_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self):
        use_int8 = False
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
            if LLM_LOAD_INT8:
                try:
                    import accelerate  # noqa: F401
                    import bitsandbytes  # noqa: F401
                    use_int8 = True
                    logger.info("GPU 可用，启用 INT8 量化加载（显存减少约50%）")
                except ImportError:
                    logger.warning("INT8 量化需要 pip install accelerate bitsandbytes，回退 float16")
            if not use_int8:
                logger.info("GPU 可用，以 float16 加载模型到 CUDA")
        else:
            self.device = "cpu"
            dtype = torch.float32
            logger.info("无 GPU，以 float32 在 CPU 上运行")

        logger.info(f"加载模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

        if use_int8:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        vram = ""
        if self.device == "cuda":
            vram = f"，显存: {torch.cuda.memory_allocated() / (1024**3):.1f} GB"
        logger.info(f"模型加载完成，设备: {self.device}{vram}")

    def _build_generation_kwargs(self) -> dict:
        """统一的生成参数，供 generate() 和 generate_stream() 共用"""
        return {
            "max_new_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "do_sample": LLM_TEMPERATURE > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def generate(self, prompt: str) -> str:
        """同步推理"""
        if not self._loaded:
            raise RuntimeError("LLM 尚未加载")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **self._build_generation_kwargs(),
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def generate_stream(self, prompt: str):
        """流式推理 — 生成器，逐 token 产出"""
        if not self._loaded:
            raise RuntimeError("LLM 尚未加载")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120
        )

        gen_kwargs = {
            **inputs,
            **self._build_generation_kwargs(),
            "streamer": streamer,
            "use_cache": True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join(timeout=30)


# ── 全局单例 ────────────────────────────────────────────
_llm_instance: LLMEngine | None = None


def get_llm() -> LLMEngine | None:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance if _llm_instance.loaded else None
    try:
        _llm_instance = LLMEngine()
        _llm_instance.load()
        return _llm_instance
    except Exception as e:
        logger.error(f"LLM 加载失败: {e}")
        _llm_instance = None
        return None
