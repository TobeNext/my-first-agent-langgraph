from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from app.config import get_settings

EMBEDDING_DIMENSION = 384


class EmbeddingProvider(Protocol):
    def embed_query(self, query_text: str) -> list[float]: ...


class HashEmbeddingProvider:
    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def embed_query(self, query_text: str) -> list[float]:
        return _hash_embed_query_text(query_text, dimension=self.dimension)


def embed_query_text(query_text: str, *, provider: EmbeddingProvider | None = None) -> list[float]:
    return (provider or build_embedding_provider()).embed_query(query_text)


def build_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider.strip().lower() in {"hash", "mock", "deterministic"}:
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)

    api_key = (
        settings.embedding_api_key
        or settings.model_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ZHIPU_API_KEY")
    )
    if not api_key:
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)

    try:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, object] = {
            "model": settings.embedding_model,
            "api_key": api_key,
        }
        base_url = settings.embedding_base_url or settings.model_base_url
        if base_url:
            kwargs["base_url"] = base_url
        if settings.embedding_dimension and "text-embedding-3" in settings.embedding_model:
            kwargs["dimensions"] = settings.embedding_dimension
        return OpenAIEmbeddings(**kwargs)
    except Exception:
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)


def _hash_embed_query_text(query_text: str, *, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for token in query_text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
