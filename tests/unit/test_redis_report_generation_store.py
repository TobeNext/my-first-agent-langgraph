import json
from collections import defaultdict

import pytest

from app.integrations.redis_report_generation_store import (
    REPORT_GENERATION_PENDING_QUEUE_KEY,
    RedisReportGenerationStore,
    report_manifest_key,
    report_status_key,
    report_task_interview_index_key,
    report_task_key,
)
from app.schemas.interview_report import ReportGenerationTask

NOW = "2026-06-19T00:00:00.000Z"


class FakeRedisClient:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.lists: defaultdict[str, list[str]] = defaultdict(list)

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


def build_task(overrides: dict | None = None) -> ReportGenerationTask:
    payload = {
        "schemaVersion": 1,
        "taskId": "report-task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "resourceId": "resource-1",
        "targetRole": "Backend Engineer",
        "responseLanguage": "zh",
        "evaluationManifestKey": "interview:interview-1:evaluation:manifest",
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return ReportGenerationTask.model_validate(payload)


@pytest.mark.asyncio
async def test_enqueue_task_creates_documented_keys_status_manifest_and_queue() -> None:
    redis = FakeRedisClient()
    store = RedisReportGenerationStore(redis, now=lambda: NOW)

    manifest = await store.enqueue_task(build_task())

    assert manifest.taskId == "report-task-1"
    assert redis.lists[REPORT_GENERATION_PENDING_QUEUE_KEY] == ["report-task-1"]
    assert redis.strings[report_task_interview_index_key("report-task-1")] == "interview-1"
    assert json.loads(redis.strings[report_task_key("interview-1", "report-task-1")])[
        "evaluationManifestKey"
    ] == "interview:interview-1:evaluation:manifest"
    assert json.loads(redis.strings[report_status_key("interview-1", "report-task-1")]) == {
        "schemaVersion": 1,
        "taskId": "report-task-1",
        "interviewId": "interview-1",
        "status": "pending",
        "attempts": 0,
        "createdAt": NOW,
    }
    assert json.loads(redis.strings[report_manifest_key("interview-1")]) == {
        "schemaVersion": 1,
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "taskId": "report-task-1",
        "status": "pending",
        "evaluationExpectedCount": 0,
        "evaluationCompletedCount": 0,
        "evaluationFailedCount": 0,
        "markdownAvailable": False,
        "attempts": 0,
        "createdAt": NOW,
        "updatedAt": NOW,
    }


@pytest.mark.asyncio
async def test_duplicate_enqueue_same_interview_does_not_create_multiple_valid_tasks() -> None:
    redis = FakeRedisClient()
    store = RedisReportGenerationStore(redis, now=lambda: NOW)

    first = await store.enqueue_task(build_task())
    second = await store.enqueue_task(
        build_task({"taskId": "report-task-2", "threadId": "thread-1"})
    )

    assert second == first
    assert redis.lists[REPORT_GENERATION_PENDING_QUEUE_KEY] == ["report-task-1"]
    assert report_task_interview_index_key("report-task-2") not in redis.strings


@pytest.mark.asyncio
async def test_claim_next_task_marks_running_and_updates_attempt_progress() -> None:
    store = RedisReportGenerationStore(FakeRedisClient(), now=lambda: NOW)
    await store.enqueue_task(build_task())

    claimed = await store.claim_next_task()

    assert claimed and claimed.taskId == "report-task-1"
    status = await store.read_task_status("report-task-1")
    manifest = await store.read_manifest("interview-1")
    assert status and status.status == "running"
    assert status.attempts == 1
    assert manifest and manifest.status == "running"
    assert manifest.attempts == 1
    assert manifest.startedAt == NOW


@pytest.mark.asyncio
async def test_mark_succeeded_records_report_id_markdown_and_evaluation_counts() -> None:
    store = RedisReportGenerationStore(FakeRedisClient(), now=lambda: NOW)
    await store.enqueue_task(build_task())
    await store.mark_running("report-task-1")

    await store.mark_succeeded(
        "report-task-1",
        report_id="report-1",
        evaluation_expected_count=6,
        evaluation_completed_count=6,
        evaluation_failed_count=0,
    )

    status = await store.read_task_status("report-task-1")
    manifest = await store.read_manifest("interview-1")
    assert status and status.status == "succeeded"
    assert status.completedAt == NOW
    assert manifest and manifest.status == "succeeded"
    assert manifest.reportId == "report-1"
    assert manifest.markdownAvailable is True
    assert manifest.evaluationExpectedCount == 6
    assert manifest.evaluationCompletedCount == 6
    assert manifest.evaluationFailedCount == 0


@pytest.mark.asyncio
async def test_mark_failed_and_retry_requeues_existing_task() -> None:
    redis = FakeRedisClient()
    store = RedisReportGenerationStore(redis, now=lambda: NOW)
    await store.enqueue_task(build_task())
    await store.claim_next_task()

    await store.mark_failed(
        "report-task-1",
        "evaluation failed",
        evaluation_expected_count=6,
        evaluation_completed_count=5,
        evaluation_failed_count=1,
    )

    failed_status = await store.read_task_status("report-task-1")
    failed_manifest = await store.read_manifest("interview-1")
    assert failed_status and failed_status.status == "failed"
    assert failed_status.lastError == "evaluation failed"
    assert failed_manifest and failed_manifest.status == "failed"
    assert failed_manifest.markdownAvailable is False
    assert failed_manifest.evaluationFailedCount == 1

    await store.retry_task("report-task-1", error="retry after transient failure")

    retry_status = await store.read_task_status("report-task-1")
    retry_manifest = await store.read_manifest("interview-1")
    assert retry_status and retry_status.status == "pending"
    assert retry_status.attempts == 1
    assert retry_manifest and retry_manifest.status == "pending"
    assert retry_manifest.lastError == "retry after transient failure"
    assert redis.lists[REPORT_GENERATION_PENDING_QUEUE_KEY] == ["report-task-1"]
