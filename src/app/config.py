from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="development", alias="APP_ENV")
    model_provider: str = Field(default="mock", alias="MODEL_PROVIDER")
    model_name: str = Field(default="mock/interview-runtime", alias="MODEL_NAME")
    model_api_key: str | None = Field(default=None, alias="MODEL_API_KEY")
    model_base_url: str | None = Field(default=None, alias="MODEL_BASE_URL")
    model_timeout_seconds: float = Field(default=20, alias="MODEL_TIMEOUT_SECONDS", gt=0)
    model_max_retries: int = Field(default=2, alias="MODEL_MAX_RETRIES", ge=0)
    model_temperature: float = Field(default=0.2, alias="MODEL_TEMPERATURE", ge=0, le=2)
    embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION", gt=0)
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    redis_connect_timeout_seconds: float = Field(
        default=0.2,
        alias="REDIS_CONNECT_TIMEOUT_SECONDS",
        gt=0,
    )
    redis_socket_timeout_seconds: float = Field(
        default=0.2,
        alias="REDIS_SOCKET_TIMEOUT_SECONDS",
        gt=0,
    )
    milvus_address: str = Field(default="http://localhost:19530", alias="MILVUS_ADDRESS")
    checkpoint_url: str = Field(default="sqlite:///./checkpoints.db", alias="CHECKPOINT_URL")
    outcome_root: str = Field(default="../my-first-agent/Interview outcome", alias="OUTCOME_ROOT")
    rag_log_root: str = Field(default="../my-first-agent/RAG LOG INFO", alias="RAG_LOG_ROOT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
