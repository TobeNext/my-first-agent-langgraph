from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings, get_settings


class ChatModelLike(Protocol):
    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any: ...


class StructuredChatModelLike(Protocol):
    def with_structured_output(
        self, schema: type[Any], *args: Any, **kwargs: Any
    ) -> ChatModelLike: ...


@dataclass(frozen=True)
class MockChatModel:
    response: str | dict[str, Any] | None = None

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> str:
        if self.response is None:
            return json.dumps({"followUpQuestion": None}, ensure_ascii=False)
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)

    def with_structured_output(self, schema: type[Any], *args: Any, **kwargs: Any) -> ChatModelLike:
        model = self

        class _StructuredMockChatModel:
            def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
                raw = json.loads(model.invoke(input))
                return schema.model_validate(raw)

        return _StructuredMockChatModel()


ChatModelFactory = Callable[[Settings], ChatModelLike | StructuredChatModelLike]


def create_chat_model(
    *,
    settings: Settings | None = None,
    factory: ChatModelFactory | None = None,
) -> ChatModelLike | StructuredChatModelLike:
    resolved_settings = settings or get_settings()
    if factory:
        return factory(resolved_settings)

    provider = resolved_settings.model_provider.strip().lower()
    if provider in {"", "mock", "none"}:
        return MockChatModel()

    if provider in {"openai", "openai-compatible", "zhipu", "zhipuai", "deepseek"}:
        api_key = _resolve_api_key(resolved_settings, provider)
        if not api_key:
            return MockChatModel()
        return _create_openai_compatible_model(resolved_settings, api_key)

    raise ValueError(f"Unsupported MODEL_PROVIDER: {resolved_settings.model_provider}")


def _resolve_api_key(settings: Settings, provider: str) -> str | None:
    if settings.model_api_key:
        return settings.model_api_key
    if provider in {"zhipu", "zhipuai"}:
        return os.getenv("ZHIPU_API_KEY")
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _create_openai_compatible_model(settings: Settings, api_key: str) -> Any:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise RuntimeError(
            "LangChain is not installed. Install project dependencies before using a real model."
        ) from exc

    kwargs: dict[str, Any] = {
        "model": _resolve_model_name(settings),
        "model_provider": "openai",
        "api_key": api_key,
        "temperature": settings.model_temperature,
        "timeout": settings.model_timeout_seconds,
        "max_retries": settings.model_max_retries,
    }
    base_url = _resolve_base_url(settings)
    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(**kwargs)


def _resolve_model_name(settings: Settings) -> str:
    if settings.model_provider.strip().lower() == "deepseek" and settings.model_name.startswith(
        "mock/"
    ):
        return "deepseek-chat"
    return settings.model_name


def _resolve_base_url(settings: Settings) -> str | None:
    if settings.model_base_url:
        return settings.model_base_url
    if settings.model_provider.strip().lower() == "deepseek":
        return "https://api.deepseek.com"
    return None
