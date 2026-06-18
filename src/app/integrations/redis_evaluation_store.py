from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from app.schemas.answer_evaluation import (
    AnswerEvaluationTask,
    AnswerEvaluationTaskStatus,
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)

ANSWER_EVALUATION_PENDING_QUEUE_KEY = "answer-evaluation:pending"
TASK_INTERVIEW_INDEX_PREFIX = "answer-evaluation:task-interview:"

class EvaluationRedisClient(Protocol):
    def get(self, key: str) -> Awaitable[str | None]: ...

    def set(self, key: str, value: str) -> Awaitable[object]: ...

    def rpush(self, key: str, value: str) -> Awaitable[object]: ...

    def lpop(self, key: str) -> Awaitable[str | None]: ...

    def sadd(self, key: str, value: str) -> Awaitable[object]: ...

    def smembers(self, key: str) -> Awaitable[list[str] | set[str]]: ...


def task_interview_index_key(task_id: str) -> str:
    return f"{TASK_INTERVIEW_INDEX_PREFIX}{task_id}"


def manifest_key(interview_id: str) -> str:
    return f"interview:{interview_id}:evaluation:manifest"


def tasks_key(interview_id: str) -> str:
    return f"interview:{interview_id}:evaluation:tasks"


def task_key(interview_id: str, task_id: str) -> str:
    return f"interview:{interview_id}:evaluation:task:{task_id}"


def status_key(interview_id: str, task_id: str) -> str:
    return f"interview:{interview_id}:evaluation:status:{task_id}"


def result_key(interview_id: str, task_id: str) -> str:
    return f"interview:{interview_id}:evaluation:result:{task_id}"


class RedisAnswerEvaluationStore:
    def __init__(
        self,
        client: EvaluationRedisClient,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.client = client
        self.now = now or _utc_now

    async def enqueue_task(self, raw_task: AnswerEvaluationTask | dict) -> None:
        task = AnswerEvaluationTask.model_validate(raw_task)
        manifest = await self._read_or_create_manifest(task.interviewId, task.threadId)
        status = AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task.taskId,
                "interviewId": task.interviewId,
                "attemptId": task.attemptId,
                "status": "pending",
                "attempts": 0,
                "createdAt": self.now(),
            }
        )
        next_manifest = InterviewEvaluationManifest.model_validate(
            {
                **manifest.model_dump(),
                "expectedTaskIds": _unique_append(manifest.expectedTaskIds, task.taskId),
                "failedTaskIds": _remove_value(manifest.failedTaskIds, task.taskId),
                "updatedAt": self.now(),
            }
        )

        await self.client.set(task_interview_index_key(task.taskId), task.interviewId)
        await self.client.set(task_key(task.interviewId, task.taskId), _serialize(task))
        await self.client.set(status_key(task.interviewId, task.taskId), _serialize(status))
        await self.client.sadd(tasks_key(task.interviewId), task.taskId)
        await self._write_manifest(next_manifest)
        await self.client.rpush(ANSWER_EVALUATION_PENDING_QUEUE_KEY, task.taskId)

    async def claim_next_task(self) -> AnswerEvaluationTask | None:
        task_id = await self.client.lpop(ANSWER_EVALUATION_PENDING_QUEUE_KEY)
        if not task_id:
            return None

        task = await self.read_task(task_id)
        if not task:
            return None

        await self.mark_running(task_id)
        return task

    async def mark_running(self, task_id: str) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        status = AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "attemptId": task.attemptId,
                "status": "running",
                "attempts": (current_status.attempts if current_status else 0) + 1,
                "createdAt": current_status.createdAt if current_status else self.now(),
                "startedAt": self.now(),
                "lastError": current_status.lastError if current_status else None,
            }
        )

        await self.client.set(status_key(task.interviewId, task_id), _serialize(status))

    async def mark_succeeded(self, raw_result: LlmAnswerEvaluationResult | dict) -> None:
        result = LlmAnswerEvaluationResult.model_validate(raw_result)
        task = await self._require_task(result.taskId)
        current_status = await self.read_task_status(result.taskId)
        status = AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": result.taskId,
                "interviewId": result.interviewId,
                "attemptId": result.attemptId,
                "status": "succeeded",
                "attempts": current_status.attempts if current_status else 0,
                "createdAt": current_status.createdAt if current_status else self.now(),
                "startedAt": current_status.startedAt if current_status else None,
                "completedAt": self.now(),
            }
        )
        manifest = await self._read_or_create_manifest(task.interviewId, task.threadId)
        next_manifest = InterviewEvaluationManifest.model_validate(
            {
                **manifest.model_dump(),
                "completedTaskIds": _unique_append(manifest.completedTaskIds, result.taskId),
                "failedTaskIds": _remove_value(manifest.failedTaskIds, result.taskId),
                "updatedAt": self.now(),
            }
        )

        await self.client.set(result_key(result.interviewId, result.taskId), _serialize(result))
        await self.client.set(status_key(result.interviewId, result.taskId), _serialize(status))
        await self._write_manifest(next_manifest)

    async def mark_failed(self, task_id: str, error: str) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        status = AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "attemptId": task.attemptId,
                "status": "failed",
                "attempts": current_status.attempts if current_status else 0,
                "createdAt": current_status.createdAt if current_status else self.now(),
                "startedAt": current_status.startedAt if current_status else None,
                "completedAt": self.now(),
                "lastError": error,
            }
        )
        manifest = await self._read_or_create_manifest(task.interviewId, task.threadId)
        next_manifest = InterviewEvaluationManifest.model_validate(
            {
                **manifest.model_dump(),
                "failedTaskIds": _unique_append(manifest.failedTaskIds, task_id),
                "completedTaskIds": _remove_value(manifest.completedTaskIds, task_id),
                "updatedAt": self.now(),
            }
        )

        await self.client.set(status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)

    async def retry_task(self, task_id: str, error: str | None = None) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        status = AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "attemptId": task.attemptId,
                "status": "pending",
                "attempts": current_status.attempts if current_status else 0,
                "createdAt": current_status.createdAt if current_status else self.now(),
                "lastError": error or (current_status.lastError if current_status else None),
            }
        )
        manifest = await self._read_or_create_manifest(task.interviewId, task.threadId)
        next_manifest = InterviewEvaluationManifest.model_validate(
            {
                **manifest.model_dump(),
                "failedTaskIds": _remove_value(manifest.failedTaskIds, task_id),
                "updatedAt": self.now(),
            }
        )

        await self.client.set(status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)
        await self.client.rpush(ANSWER_EVALUATION_PENDING_QUEUE_KEY, task_id)

    async def seal_interview(self, interview_id: str) -> None:
        manifest = await self.read_manifest(interview_id)
        if not manifest:
            raise ValueError(
                f"Cannot seal missing evaluation manifest for interview {interview_id}."
            )

        await self._write_manifest(
            InterviewEvaluationManifest.model_validate(
                {
                    **manifest.model_dump(),
                    "sealed": True,
                    "sealedAt": self.now(),
                    "updatedAt": self.now(),
                }
            )
        )

    async def read_task(self, task_id: str) -> AnswerEvaluationTask | None:
        interview_id = await self.client.get(task_interview_index_key(task_id))
        if not interview_id:
            return None
        return await _parse_json(
            await self.client.get(task_key(interview_id, task_id)),
            AnswerEvaluationTask,
        )

    async def read_task_status(self, task_id: str) -> AnswerEvaluationTaskStatus | None:
        interview_id = await self.client.get(task_interview_index_key(task_id))
        if not interview_id:
            return None
        return await _parse_json(
            await self.client.get(status_key(interview_id, task_id)),
            AnswerEvaluationTaskStatus,
        )

    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None:
        return await _parse_json(
            await self.client.get(manifest_key(interview_id)),
            InterviewEvaluationManifest,
        )

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]:
        manifest = await self.read_manifest(interview_id)
        if manifest:
            task_ids = manifest.expectedTaskIds
        else:
            members = await self.client.smembers(tasks_key(interview_id))
            task_ids = list(members)
        results = [
            await _parse_json(
                await self.client.get(result_key(interview_id, task_id)),
                LlmAnswerEvaluationResult,
            )
            for task_id in task_ids
        ]
        return [result for result in results if result is not None]

    async def _require_task(self, task_id: str) -> AnswerEvaluationTask:
        task = await self.read_task(task_id)
        if not task:
            raise ValueError(f"Answer evaluation task {task_id} was not found.")
        return task

    async def _read_or_create_manifest(
        self,
        interview_id: str,
        thread_id: str,
    ) -> InterviewEvaluationManifest:
        existing_manifest = await self.read_manifest(interview_id)
        if existing_manifest:
            return existing_manifest

        return InterviewEvaluationManifest.model_validate(
            {
                "schemaVersion": 1,
                "interviewId": interview_id,
                "threadId": thread_id,
                "expectedTaskIds": [],
                "completedTaskIds": [],
                "failedTaskIds": [],
                "sealed": False,
                "updatedAt": self.now(),
            }
        )

    async def _write_manifest(self, manifest: InterviewEvaluationManifest) -> None:
        await self.client.set(manifest_key(manifest.interviewId), _serialize(manifest))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _unique_append(values: list[str], value: str) -> list[str]:
    return [*values] if value in values else [*values, value]


def _remove_value(values: list[str], value: str) -> list[str]:
    return [item for item in values if item != value]


def _serialize(value: BaseModel) -> str:
    return value.model_dump_json(exclude_none=True)


async def _parse_json(raw: str | None, model: type[BaseModel]) -> Any:
    if not raw:
        return None
    return model.model_validate(json.loads(raw))
