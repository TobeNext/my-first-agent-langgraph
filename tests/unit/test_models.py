import pytest

from app.config import Settings
from app.integrations.models import MockChatModel, create_chat_model


def _settings(**overrides: object) -> Settings:
    values = {
        "APP_ENV": "test",
        "MODEL_PROVIDER": "mock",
        "MODEL_NAME": "mock/interview-runtime",
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


def test_create_chat_model_configures_deepseek_openai_compatible_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)

    model = create_chat_model(
        settings=_settings(
            MODEL_PROVIDER="deepseek",
            MODEL_NAME="mock/interview-runtime",
            MODEL_API_KEY="test-key",
        )
    )

    assert model is sentinel
    assert captured["model"] == "deepseek-chat"
    assert captured["model_provider"] == "openai"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.deepseek.com"


def test_create_chat_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
        create_chat_model(settings=_settings(MODEL_PROVIDER="unknown"))
