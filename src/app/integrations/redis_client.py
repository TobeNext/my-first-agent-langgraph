from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.integrations.redis_evaluation_store import RedisAnswerEvaluationStore
from app.integrations.redis_report_generation_store import RedisReportGenerationStore


class RedisEvaluationClient:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def get(self, key: str) -> str | None:
        value = await self.client.get(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    async def set(self, key: str, value: str) -> object:
        return await self.client.set(key, value)

    async def rpush(self, key: str, value: str) -> object:
        return await self.client.rpush(key, value)

    async def lpop(self, key: str) -> str | None:
        value = await self.client.lpop(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    async def sadd(self, key: str, value: str) -> object:
        return await self.client.sadd(key, value)

    async def smembers(self, key: str) -> list[str]:
        values = await self.client.smembers(key)
        return [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ]

    async def disconnect(self) -> None:
        await self.client.aclose()


def create_redis_evaluation_client(
    settings: Settings | None = None,
) -> RedisEvaluationClient:
    resolved_settings = settings or get_settings()
    try:
        from redis.asyncio import Redis
    except ImportError as exc:
        raise RuntimeError(
            "redis is not installed. Install project dependencies before using Redis evaluation."
        ) from exc

    return RedisEvaluationClient(
        Redis.from_url(
            resolved_settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=resolved_settings.redis_connect_timeout_seconds,
            socket_timeout=resolved_settings.redis_socket_timeout_seconds,
        )
    )


def create_redis_answer_evaluation_store(
    client: RedisEvaluationClient | None = None,
    *,
    settings: Settings | None = None,
) -> RedisAnswerEvaluationStore:
    return RedisAnswerEvaluationStore(client or create_redis_evaluation_client(settings))


def create_redis_report_generation_store(
    client: RedisEvaluationClient | None = None,
    *,
    settings: Settings | None = None,
) -> RedisReportGenerationStore:
    return RedisReportGenerationStore(client or create_redis_evaluation_client(settings))
