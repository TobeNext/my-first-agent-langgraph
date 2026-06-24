import pytest

from app.config import Settings
from app.integrations.models import (
    MockChatModel,
    TracedChatModel,
    create_chat_model,
    invoke_json_output_model,
    should_use_json_object_response_format,
    should_use_native_structured_output,
)


def _settings(**overrides: object) -> Settings:
    values = {
        "APP_ENV": "test",
        "MODEL_PROVIDER": "mock",
        "MODEL_NAME": "mock/interview-runtime",
        "MODEL_BASE_URL": None,
        "MODEL_STRUCTURED_OUTPUT_MODE": "auto",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_create_chat_model_defaults_to_noop_mock_without_api_key() -> None:
    model = create_chat_model(settings=_settings())

    assert isinstance(model, MockChatModel)
    assert model.invoke("prompt") == '{"followUpQuestion": null}'


def test_create_chat_model_returns_mock_when_real_provider_has_no_key() -> None:
    model = create_chat_model(settings=_settings(MODEL_PROVIDER="openai", MODEL_API_KEY=None))

    assert isinstance(model, MockChatModel)


def test_create_chat_model_configures_deepseek_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_chat_deepseek(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("langchain_deepseek.ChatDeepSeek", fake_chat_deepseek)

    model = create_chat_model(
        settings=_settings(
            MODEL_PROVIDER="deepseek",
            MODEL_NAME="mock/interview-runtime",
            MODEL_API_KEY="test-key",
        )
    )

    assert model is sentinel
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 90
    assert captured["max_retries"] == 2
    assert captured["model_kwargs"] == {"response_format": {"type": "json_object"}}


def test_create_chat_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
        create_chat_model(settings=_settings(MODEL_PROVIDER="unknown"))


class _StructuredModel:
    def __init__(self) -> None:
        self.invoke_kwargs: dict[str, object] | None = None
        self.structured_output_kwargs: dict[str, object] | None = None

    def invoke(self, prompt: str, **kwargs: object) -> str:
        self.invoke_kwargs = kwargs
        return "{}"

    def with_structured_output(self, schema, **kwargs):
        self.structured_output_kwargs = kwargs
        return self


def test_deepseek_chat_defaults_to_native_structured_output_mode() -> None:
    model = TracedChatModel(
        model=_StructuredModel(),
        settings=_settings(MODEL_PROVIDER="deepseek", MODEL_API_KEY="test-key"),
    )

    assert should_use_native_structured_output(model) is True


def test_deepseek_structured_output_defaults_to_json_mode() -> None:
    inner = _StructuredModel()
    model = TracedChatModel(
        model=inner,
        settings=_settings(MODEL_PROVIDER="deepseek", MODEL_API_KEY="test-key"),
    )

    model.with_structured_output(dict)

    assert inner.structured_output_kwargs == {"method": "json_mode"}


def test_deepseek_reasoner_defaults_to_raw_json_structured_output_mode() -> None:
    model = TracedChatModel(
        model=_StructuredModel(),
        settings=_settings(
            MODEL_PROVIDER="deepseek",
            MODEL_NAME="deepseek-reasoner",
            MODEL_API_KEY="test-key",
        ),
    )

    assert should_use_native_structured_output(model) is False


def test_deepseek_raw_json_uses_json_object_response_format() -> None:
    inner = _StructuredModel()
    model = TracedChatModel(
        model=inner,
        settings=_settings(MODEL_PROVIDER="deepseek", MODEL_API_KEY="test-key"),
    )

    response = invoke_json_output_model(model, "Return JSON only.")

    assert response == "{}"
    assert should_use_json_object_response_format(model) is True
    assert inner.invoke_kwargs == {"response_format": {"type": "json_object"}}


def test_openai_raw_json_does_not_force_json_object_response_format() -> None:
    inner = _StructuredModel()
    model = TracedChatModel(
        model=inner,
        settings=_settings(MODEL_PROVIDER="openai", MODEL_API_KEY="test-key"),
    )

    response = invoke_json_output_model(model, "Return JSON only.")

    assert response == "{}"
    assert should_use_json_object_response_format(model) is False
    assert inner.invoke_kwargs == {}


def test_structured_output_mode_can_force_native_for_deepseek() -> None:
    model = TracedChatModel(
        model=_StructuredModel(),
        settings=_settings(
            MODEL_PROVIDER="deepseek",
            MODEL_API_KEY="test-key",
            MODEL_STRUCTURED_OUTPUT_MODE="native",
        ),
    )

    assert should_use_native_structured_output(model) is True


def test_openai_defaults_to_native_structured_output_mode() -> None:
    model = TracedChatModel(
        model=_StructuredModel(),
        settings=_settings(MODEL_PROVIDER="openai", MODEL_API_KEY="test-key"),
    )

    assert should_use_native_structured_output(model) is True
