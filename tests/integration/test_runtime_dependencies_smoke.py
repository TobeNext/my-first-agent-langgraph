import json
import os
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config import get_settings
from app.integrations.embeddings import embed_query_text
from app.integrations.milvus_store import MilvusQuestionStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RUNTIME_DEPENDENCY_SMOKE") != "1",
    reason="Set RUN_RUNTIME_DEPENDENCY_SMOKE=1 to run Docker Redis/Milvus smoke tests.",
)


def _redis_command(host: str, port: int, *parts: str) -> bytes:
    payload = f"*{len(parts)}\r\n" + "".join(
        f"${len(part.encode())}\r\n{part}\r\n" for part in parts
    )
    with socket.create_connection((host, port), timeout=5) as client:
        client.sendall(payload.encode())
        return client.recv(4096)


def _host_and_port(address: str) -> tuple[str, str]:
    parsed = urlparse(address if "://" in address else f"http://{address}")
    return parsed.hostname or "localhost", str(parsed.port or 19530)


def test_milvus_and_redis_accept_real_runtime_smoke_writes() -> None:
    settings = get_settings()
    host, port = _host_and_port(settings.milvus_address)
    alias = f"smoke_{uuid4().hex}"
    collection_name = f"runtime_smoke_{uuid4().hex}"

    connections.connect(alias=alias, host=host, port=port, timeout=5)
    try:
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("questionText", DataType.VARCHAR, max_length=256),
            ]
        )
        collection = Collection(collection_name, schema=schema, using=alias)
        collection.insert(
            [
                ["smoke-question-1"],
                [[0.1, 0.2, 0.3, 0.4]],
                ["Explain how you would trace a RAG candidate recall."],
            ]
        )
        collection.flush()
        collection.create_index(
            "embedding",
            {
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {},
            },
        )
        collection.load()
        hits = collection.search(
            [[0.1, 0.2, 0.3, 0.4]],
            "embedding",
            {"metric_type": "COSINE", "params": {}},
            limit=1,
            output_fields=["questionText"],
        )
    finally:
        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)
        connections.disconnect(alias)

    assert hits
    assert hits[0][0].id == "smoke-question-1"
    assert "RAG candidate recall" in hits[0][0].entity.get("questionText")

    redis_host = settings.redis_url.removeprefix("redis://").split(":", 1)[0]
    redis_port = int(settings.redis_url.rsplit(":", 1)[1].split("/", 1)[0])
    task_key = f"my-first-agent-langgraph:smoke:{uuid4().hex}"
    task_payload = json.dumps(
        {
            "type": "answer-evaluation-smoke",
            "threadId": "dependency-smoke",
            "status": "queued",
        }
    )

    set_response = _redis_command(redis_host, redis_port, "SET", task_key, task_payload, "EX", "30")
    get_response = _redis_command(redis_host, redis_port, "GET", task_key)
    _redis_command(redis_host, redis_port, "DEL", task_key)

    assert set_response.startswith(b"+OK")
    assert b"answer-evaluation-smoke" in get_response


def test_existing_interview_questions_collection_can_be_read() -> None:
    store = MilvusQuestionStore()
    if not store.collection_exists():
        pytest.skip("Milvus collection interview_questions does not exist in this environment.")

    result = store.search(
        vector=embed_query_text("Target role: AI Engineer\nPrimary skill: RAG 检索"),
        top_k=3,
        round_type="professional-skills",
    )

    assert result.questions
    assert all(question.id for question in result.questions)
    assert all(question.text for question in result.questions)
