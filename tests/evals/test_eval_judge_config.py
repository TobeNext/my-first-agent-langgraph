import pytest

from tests.evals.evaluators.deepseek_judge import (
    DEFAULT_EVAL_MODEL_BASE_URL,
    DEFAULT_EVAL_MODEL_NAME,
    EvalJudgeConfig,
    create_langchain_chat_model,
    eval_judge_config_from_env,
    has_eval_judge_key,
)


def test_eval_judge_config_defaults_to_deepseek_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("EVAL_MODEL_NAME", raising=False)
    monkeypatch.delenv("EVAL_MODEL_BASE_URL", raising=False)

    config = eval_judge_config_from_env()

    assert config.provider == "deepseek"
    assert config.model_name == DEFAULT_EVAL_MODEL_NAME
    assert config.base_url == DEFAULT_EVAL_MODEL_BASE_URL
    assert config.api_key is None
    assert config.enabled is False
    assert has_eval_judge_key() is False


def test_eval_judge_config_reads_deepseek_key_and_does_not_expose_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-key")

    config = eval_judge_config_from_env()

    assert config.api_key == "secret-test-key"
    assert config.enabled is True
    assert "secret-test-key" not in str(config.safe_metadata)
    assert config.safe_metadata == {
        "provider": "deepseek",
        "model_name": DEFAULT_EVAL_MODEL_NAME,
        "base_url": DEFAULT_EVAL_MODEL_BASE_URL,
        "enabled": True,
    }


def test_eval_model_api_key_overrides_deepseek_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_MODEL_API_KEY", "eval-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = eval_judge_config_from_env()

    assert config.api_key == "eval-key"


def test_require_key_raises_clear_error_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No eval model API key is configured"):
        eval_judge_config_from_env(require_key=True)


def test_langchain_chat_model_uses_openai_compatible_deepseek_config() -> None:
    config = EvalJudgeConfig(api_key="test-key", timeout_seconds=12.0, max_retries=1)

    model = create_langchain_chat_model(config)

    assert model.model_name == DEFAULT_EVAL_MODEL_NAME
    assert str(model.openai_api_base) == DEFAULT_EVAL_MODEL_BASE_URL
    assert model.temperature == 0.0
    assert model.request_timeout == 12.0
    assert model.max_retries == 1
