from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

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
        with _chat_completion_span(
            provider="mock",
            model="mock/interview-runtime",
            timeout_seconds=None,
            max_retries=None,
        ) as span:
            try:
                if self.response is None:
                    response = json.dumps({"followUpQuestion": None}, ensure_ascii=False)
                elif isinstance(self.response, str):
                    response = self.response
                else:
                    response = json.dumps(self.response, ensure_ascii=False)
                span.set_attribute("llm.response_type", type(response).__name__)
                return response
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    def with_structured_output(self, schema: type[Any], *args: Any, **kwargs: Any) -> ChatModelLike:
        model = self

        class _StructuredMockChatModel:
            def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
                raw = json.loads(model.invoke(input))
                return schema.model_validate(raw)

        return _StructuredMockChatModel()


@dataclass(frozen=True)
class TracedChatModel:
    model: Any
    settings: Settings

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        with _chat_completion_span(
            provider=self.settings.model_provider,
            model=_resolve_model_name(self.settings),
            timeout_seconds=self.settings.model_timeout_seconds,
            max_retries=self.settings.model_max_retries,
        ) as span:
            try:
                response = self.model.invoke(input, *args, **kwargs)
                span.set_attribute("llm.response_type", type(response).__name__)
                return response
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    def with_structured_output(
        self, schema: type[Any], *args: Any, **kwargs: Any
    ) -> TracedChatModel:
        return TracedChatModel(
            model=self.model.with_structured_output(schema, *args, **kwargs),
            settings=self.settings,
        )


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
        return _trace_chat_model_if_supported(
            _create_openai_compatible_model(resolved_settings, api_key),
            settings=resolved_settings,
        )

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


def _trace_chat_model_if_supported(model: Any, *, settings: Settings) -> Any:
    if hasattr(model, "invoke"):
        return TracedChatModel(model=model, settings=settings)
    return model


def _chat_completion_span(
    *,
    provider: str,
    model: str,
    timeout_seconds: float | None,
    max_retries: int | None,
) -> trace.Span:
    attributes: dict[str, str | int | float] = {
        "llm.provider": provider,
        "llm.model": model,
    }
    if timeout_seconds is not None:
        attributes["llm.timeout_seconds"] = timeout_seconds
    if max_retries is not None:
        attributes["llm.max_retries"] = max_retries
    return trace.get_tracer("interview-python-agent").start_as_current_span(
        "llm.chat_completion",
        attributes=attributes,
    )
