import json
from collections import defaultdict

import pytest

from app.integrations.redis_evaluation_store import (
    ANSWER_EVALUATION_PENDING_QUEUE_KEY,
    RedisAnswerEvaluationStore,
    manifest_key,
    result_key,
    status_key,
    task_interview_index_key,
    task_key,
    tasks_key,
)
from app.schemas.answer_evaluation import AnswerEvaluationTask, LlmAnswerEvaluationResult

NOW = "2026-06-07T00:00:00.000Z"


class FakeRedisClient:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.lists: defaultdict[str, list[str]] = defaultdict(list)
        self.sets: defaultdict[str, set[str]] = defaultdict(set)

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def set(self, key: str, value: str) -> str:
        self.strings[key] = value
        return "OK"

    async def rpush(self, key: str, value: str) -> int:
        self.lists[key].append(value)
        return len(self.lists[key])

    async def lpop(self, key: str) -> str | None:
        if not self.lists[key]:
            return None
        return self.lists[key].pop(0)

    async def sadd(self, key: str, value: str) -> int:
        self.sets[key].add(value)
        return len(self.sets[key])

    async def smembers(self, key: str) -> list[str]:
        return list(self.sets[key])


def build_task(overrides: dict | None = None) -> AnswerEvaluationTask:
    payload = {
        "schemaVersion": 1,
        "taskId": "task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "nodeId": "node-1",
        "roundId": "round-1",
        "roundType": "professional-skills",
        "attemptId": "attempt-1",
        "targetType": "main-question",
        "targetId": "node-1",
        "targetRole": "Backend Engineer",
        "responseLanguage": "zh",
        "question": "请说明 Spring 事务传播机制。",
        "mainQuestion": "请说明 Spring 事务传播机制。",
        "referenceAnswer": "说明 REQUIRED 和 REQUIRES_NEW；补充异常回滚边界。",
        "evaluationPoints": ["说明 REQUIRED 和 REQUIRES_NEW", "补充异常回滚边界"],
        "candidateAnswer": "我会说明 REQUIRED 和 REQUIRES_NEW 的区别。",
        "nodeConversation": [],
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return AnswerEvaluationTask.model_validate(payload)


def build_result(overrides: dict | None = None) -> LlmAnswerEvaluationResult:
    payload = {
        "schemaVersion": 1,
        "taskId": "task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "nodeId": "node-1",
        "roundId": "round-1",
        "roundType": "professional-skills",
        "attemptId": "attempt-1",
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 8,
            "depth": 7,
            "specificity": 7,
            "clarity": 8,
            "weightedTotal": 7.65,
        },
        "strengths": ["覆盖了事务传播机制"],
        "missingPoints": ["异常回滚边界还不够完整"],
        "incorrectPoints": [],
        "shouldAskFollowUp": True,
        "followUpFocus": ["异常回滚边界"],
        "evaluatorModel": "test-model",
        "promptVersion": "answer-evaluation-v1",
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return LlmAnswerEvaluationResult.model_validate(payload)


@pytest.mark.asyncio
async def test_enqueue_task_creates_ts_compatible_keys_status_and_manifest() -> None:
    redis = FakeRedisClient()
    store = RedisAnswerEvaluationStore(redis, now=lambda: NOW)

    await store.enqueue_task(build_task())

    assert redis.lists[ANSWER_EVALUATION_PENDING_QUEUE_KEY] == ["task-1"]
    assert redis.strings[task_interview_index_key("task-1")] == "interview-1"
    assert json.loads(redis.strings[task_key("interview-1", "task-1")])["taskId"] == "task-1"
    assert json.loads(redis.strings[status_key("interview-1", "task-1")]) == {
        "schemaVersion": 1,
        "taskId": "task-1",
        "interviewId": "interview-1",
        "attemptId": "attempt-1",
        "status": "pending",
        "attempts": 0,
        "createdAt": NOW,
    }
    assert json.loads(redis.strings[manifest_key("interview-1")]) == {
        "schemaVersion": 1,
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "expectedTaskIds": ["task-1"],
        "completedTaskIds": [],
        "failedTaskIds": [],
        "sealed": False,
        "updatedAt": NOW,
    }
    assert redis.sets[tasks_key("interview-1")] == {"task-1"}


@pytest.mark.asyncio
async def test_claim_next_task_marks_it_running() -> None:
    store = RedisAnswerEvaluationStore(FakeRedisClient(), now=lambda: NOW)
    await store.enqueue_task(build_task())

    claimed_task = await store.claim_next_task()

    assert claimed_task and claimed_task.taskId == "task-1"
    status = await store.read_task_status("task-1")
    assert status
    assert status.status == "running"
    assert status.attempts == 1
    assert status.startedAt == NOW


@pytest.mark.asyncio
async def test_mark_succeeded_records_result_and_completed_manifest() -> None:
    store = RedisAnswerEvaluationStore(FakeRedisClient(), now=lambda: NOW)
    await store.enqueue_task(build_task())
    await store.mark_running("task-1")

    await store.mark_succeeded(build_result())

    status = await store.read_task_status("task-1")
    manifest = await store.read_manifest("interview-1")
    assert status and status.status == "succeeded"
    assert status.attempts == 1
    assert status.completedAt == NOW
    assert manifest and manifest.completedTaskIds == ["task-1"]
    assert manifest.failedTaskIds == []
    assert await store.read_results("interview-1") == [build_result()]


@pytest.mark.asyncio
async def test_seal_existing_interview_manifest() -> None:
    store = RedisAnswerEvaluationStore(FakeRedisClient(), now=lambda: NOW)
    await store.enqueue_task(build_task())

    await store.seal_interview("interview-1")

    manifest = await store.read_manifest("interview-1")
    assert manifest
    assert manifest.sealed is True
    assert manifest.sealedAt == NOW


@pytest.mark.asyncio
async def test_mark_failed_and_retry_task() -> None:
    redis = FakeRedisClient()
    store = RedisAnswerEvaluationStore(redis, now=lambda: NOW)
    await store.enqueue_task(build_task())

    await store.mark_failed("task-1", "model returned invalid JSON")

    status = await store.read_task_status("task-1")
    manifest = await store.read_manifest("interview-1")
    assert status and status.status == "failed"
    assert status.lastError == "model returned invalid JSON"
    assert manifest and manifest.failedTaskIds == ["task-1"]

    await store.retry_task("task-1")

    retry_status = await store.read_task_status("task-1")
    retry_manifest = await store.read_manifest("interview-1")
    assert retry_status and retry_status.status == "pending"
    assert retry_manifest and retry_manifest.failedTaskIds == []
    assert redis.lists[ANSWER_EVALUATION_PENDING_QUEUE_KEY] == ["task-1", "task-1"]


@pytest.mark.asyncio
async def test_read_results_falls_back_to_task_set_without_manifest() -> None:
    redis = FakeRedisClient()
    store = RedisAnswerEvaluationStore(redis, now=lambda: NOW)
    await redis.sadd(tasks_key("interview-1"), "task-1")
    await redis.set(result_key("interview-1", "task-1"), build_result().model_dump_json())

    assert await store.read_results("interview-1") == [build_result()]
