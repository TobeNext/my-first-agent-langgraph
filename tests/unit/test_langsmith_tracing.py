import os
import warnings

from app.config import Settings
from app.langsmith_tracing import (
    build_langsmith_metadata,
    langsmith_graph_context,
    langsmith_tracing_enabled,
)


def _settings(**overrides: object) -> Settings:
    values = {
        "APP_ENV": "test",
        "MODEL_PROVIDER": "mock",
        "MODEL_NAME": "mock/interview-runtime",
        "LANGSMITH_TRACING": False,
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_PROJECT": "my-first-agent-test",
        "LANGSMITH_DATA_MODE": "standard",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_langsmith_tracing_stays_disabled_by_default() -> None:
    assert langsmith_tracing_enabled(_settings()) is False
    assert langsmith_tracing_enabled(_settings(LANGSMITH_TRACING=True)) is False


def test_langsmith_tracing_requires_explicit_enable_and_api_key() -> None:
    assert (
        langsmith_tracing_enabled(
            _settings(
                LANGSMITH_TRACING=True,
                LANGSMITH_API_KEY="test-key",
                LANGSMITH_DATA_MODE="standard",
            )
        )
        is True
    )


def test_langsmith_metadata_contains_only_safe_keys() -> None:
    metadata = build_langsmith_metadata(
        settings=_settings(MODEL_PROVIDER="deepseek", MODEL_NAME="deepseek-chat"),
        thread_id="thread-1",
    )

    assert metadata == {
        "thread_id": "thread-1",
        "runtime_provider": "python-langgraph",
        "app_env": "test",
        "model_provider": "deepseek",
        "model_name": "deepseek-chat",
        "otel.trace_id": "",
    }


def test_langsmith_context_passes_settings_api_key_to_client(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = _settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="test-key")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with langsmith_graph_context(settings=settings, thread_id="thread-1"):
            pass

    assert not [
        warning
        for warning in caught
        if type(warning.message).__name__ == "LangSmithMissingAPIKeyWarning"
    ]
    assert os.environ.get("LANGSMITH_API_KEY") == "test-key"
