from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_EVAL_MODEL_PROVIDER = "deepseek"
DEFAULT_EVAL_MODEL_NAME = "deepseek-chat"
DEFAULT_EVAL_MODEL_BASE_URL = "https://api.deepseek.com"
DEFAULT_EVAL_TEMPERATURE = 0.0
DEFAULT_EVAL_TIMEOUT_SECONDS = 90.0
DEFAULT_EVAL_MAX_RETRIES = 2


@dataclass(frozen=True)
class EvalJudgeConfig:
    provider: str = DEFAULT_EVAL_MODEL_PROVIDER
    model_name: str = DEFAULT_EVAL_MODEL_NAME
    base_url: str = DEFAULT_EVAL_MODEL_BASE_URL
    api_key: str | None = None
    temperature: float = DEFAULT_EVAL_TEMPERATURE
    timeout_seconds: float = DEFAULT_EVAL_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_EVAL_MAX_RETRIES

    @property
    def enabled(self) -> bool:
        return bool((self.api_key or "").strip())

    @property
    def safe_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "enabled": self.enabled,
        }


class DeepSeekDeepEvalLLM:
    def __new__(cls, config: EvalJudgeConfig | None = None):
        try:
            from deepeval.models.base_model import DeepEvalBaseLLM
        except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent.
            raise RuntimeError("deepeval is required to create a DeepEval judge model.") from exc

        class _DeepSeekDeepEvalLLM(DeepEvalBaseLLM):
            def __init__(self, resolved_config: EvalJudgeConfig):
                self.config = resolved_config
                super().__init__(model=resolved_config.model_name)

            def load_model(self, *args: Any, **kwargs: Any) -> ChatOpenAI:
                return create_langchain_chat_model(self.config)

            def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
                response = self.model.invoke(prompt)
                return _response_text(response)

            async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
                response = await self.model.ainvoke(prompt)
                return _response_text(response)

            def get_model_name(self, *args: Any, **kwargs: Any) -> str:
                return self.config.model_name

        return _DeepSeekDeepEvalLLM(config or eval_judge_config_from_env(require_key=True))


def eval_judge_config_from_env(*, require_key: bool = False) -> EvalJudgeConfig:
    api_key = _first_env_value("EVAL_MODEL_API_KEY", "DEEPSEEK_API_KEY")
    if require_key and not api_key:
        raise RuntimeError(
            "No eval model API key is configured. Set DEEPSEEK_API_KEY or EVAL_MODEL_API_KEY."
        )

    return EvalJudgeConfig(
        provider=os.environ.get("EVAL_MODEL_PROVIDER", DEFAULT_EVAL_MODEL_PROVIDER).strip()
        or DEFAULT_EVAL_MODEL_PROVIDER,
        model_name=os.environ.get("EVAL_MODEL_NAME", DEFAULT_EVAL_MODEL_NAME).strip()
        or DEFAULT_EVAL_MODEL_NAME,
        base_url=os.environ.get("EVAL_MODEL_BASE_URL", DEFAULT_EVAL_MODEL_BASE_URL).strip()
        or DEFAULT_EVAL_MODEL_BASE_URL,
        api_key=api_key,
        temperature=_float_env("EVAL_MODEL_TEMPERATURE", DEFAULT_EVAL_TEMPERATURE),
        timeout_seconds=_float_env("EVAL_MODEL_TIMEOUT_SECONDS", DEFAULT_EVAL_TIMEOUT_SECONDS),
        max_retries=_int_env("EVAL_MODEL_MAX_RETRIES", DEFAULT_EVAL_MAX_RETRIES),
    )


def has_eval_judge_key() -> bool:
    return eval_judge_config_from_env().enabled


def create_langchain_chat_model(config: EvalJudgeConfig | None = None) -> ChatOpenAI:
    resolved = config or eval_judge_config_from_env(require_key=True)
    if not resolved.enabled:
        raise RuntimeError(
            "No eval model API key is configured. Set DEEPSEEK_API_KEY or EVAL_MODEL_API_KEY."
        )
    return ChatOpenAI(
        model=resolved.model_name,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        temperature=resolved.temperature,
        timeout=resolved.timeout_seconds,
        max_retries=resolved.max_retries,
    )


def create_deepeval_judge(config: EvalJudgeConfig | None = None) -> Any:
    return DeepSeekDeepEvalLLM(config)


def create_ragas_llm(config: EvalJudgeConfig | None = None) -> Any:
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent.
        raise RuntimeError("ragas is required to create a Ragas judge model.") from exc

    return LangchainLLMWrapper(create_langchain_chat_model(config))


def _first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)
