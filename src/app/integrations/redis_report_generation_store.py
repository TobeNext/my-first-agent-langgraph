from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from app.schemas.interview_report import (
    InterviewReportManifest,
    ReportGenerationTask,
    ReportGenerationTaskStatus,
)

REPORT_GENERATION_PENDING_QUEUE_KEY = "report-generation:pending"
REPORT_TASK_INTERVIEW_INDEX_PREFIX = "report-generation:task-interview:"


class ReportGenerationRedisClient(Protocol):
    def get(self, key: str) -> Awaitable[str | None]: ...

    def set(self, key: str, value: str) -> Awaitable[object]: ...

    def rpush(self, key: str, value: str) -> Awaitable[object]: ...

    def lpop(self, key: str) -> Awaitable[str | None]: ...


def report_task_interview_index_key(task_id: str) -> str:
    return f"{REPORT_TASK_INTERVIEW_INDEX_PREFIX}{task_id}"


def report_manifest_key(interview_id: str) -> str:
    return f"interview:{interview_id}:report:manifest"


def report_task_key(interview_id: str, task_id: str) -> str:
    return f"interview:{interview_id}:report:task:{task_id}"


def report_status_key(interview_id: str, task_id: str) -> str:
    return f"interview:{interview_id}:report:status:{task_id}"


def report_read_key(interview_id: str) -> str:
    return f"interview:{interview_id}:report:read"


class RedisReportGenerationStore:
    def __init__(
        self,
        client: ReportGenerationRedisClient,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.client = client
        self.now = now or _utc_now

    async def enqueue_task(self, raw_task: ReportGenerationTask | dict) -> InterviewReportManifest:
        task = ReportGenerationTask.model_validate(raw_task)
        existing_manifest = await self.read_manifest(task.interviewId)
        if existing_manifest:
            return existing_manifest

        created_at = self.now()
        status = ReportGenerationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task.taskId,
                "interviewId": task.interviewId,
                "status": "pending",
                "attempts": 0,
                "createdAt": created_at,
            }
        )
        manifest = InterviewReportManifest.model_validate(
            {
                "schemaVersion": 1,
                "interviewId": task.interviewId,
                "threadId": task.threadId,
                "taskId": task.taskId,
                "status": "pending",
                "evaluationExpectedCount": 0,
                "evaluationCompletedCount": 0,
                "evaluationFailedCount": 0,
                "reportId": None,
                "markdownAvailable": False,
                "attempts": 0,
                "createdAt": created_at,
                "updatedAt": created_at,
            }
        )

        await self.client.set(report_task_interview_index_key(task.taskId), task.interviewId)
        await self.client.set(report_task_key(task.interviewId, task.taskId), _serialize(task))
        await self.client.set(report_status_key(task.interviewId, task.taskId), _serialize(status))
        await self._write_manifest(manifest)
        await self.client.rpush(REPORT_GENERATION_PENDING_QUEUE_KEY, task.taskId)
        return manifest

    async def claim_next_task(self) -> ReportGenerationTask | None:
        task_id = await self.client.lpop(REPORT_GENERATION_PENDING_QUEUE_KEY)
        if not task_id:
            return None

        task = await self.read_task(task_id)
        if not task:
            return None

        await self.mark_running(task_id)
        return task

    async def mark_running(
        self,
        task_id: str,
        *,
        evaluation_expected_count: int | None = None,
        evaluation_completed_count: int | None = None,
        evaluation_failed_count: int | None = None,
    ) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        manifest = await self._require_manifest(task.interviewId)
        attempts = (current_status.attempts if current_status else manifest.attempts) + 1
        now = self.now()
        status = ReportGenerationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "status": "running",
                "attempts": attempts,
                "createdAt": current_status.createdAt if current_status else manifest.createdAt,
                "startedAt": now,
                "lastError": current_status.lastError if current_status else manifest.lastError,
            }
        )
        next_manifest = InterviewReportManifest.model_validate(
            {
                **manifest.model_dump(),
                "status": "running",
                "evaluationExpectedCount": _coalesce(
                    evaluation_expected_count,
                    manifest.evaluationExpectedCount,
                ),
                "evaluationCompletedCount": _coalesce(
                    evaluation_completed_count,
                    manifest.evaluationCompletedCount,
                ),
                "evaluationFailedCount": _coalesce(
                    evaluation_failed_count,
                    manifest.evaluationFailedCount,
                ),
                "attempts": attempts,
                "startedAt": now,
                "updatedAt": now,
            }
        )

        await self.client.set(report_status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)

    async def mark_succeeded(
        self,
        task_id: str,
        *,
        report_id: str,
        evaluation_expected_count: int,
        evaluation_completed_count: int,
        evaluation_failed_count: int,
        markdown_available: bool = True,
    ) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        manifest = await self._require_manifest(task.interviewId)
        now = self.now()
        status = ReportGenerationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "status": "succeeded",
                "attempts": current_status.attempts if current_status else manifest.attempts,
                "createdAt": current_status.createdAt if current_status else manifest.createdAt,
                "startedAt": current_status.startedAt if current_status else manifest.startedAt,
                "completedAt": now,
            }
        )
        next_manifest = InterviewReportManifest.model_validate(
            {
                **manifest.model_dump(),
                "status": "succeeded",
                "evaluationExpectedCount": evaluation_expected_count,
                "evaluationCompletedCount": evaluation_completed_count,
                "evaluationFailedCount": evaluation_failed_count,
                "reportId": report_id,
                "markdownAvailable": markdown_available,
                "lastError": None,
                "completedAt": now,
                "updatedAt": now,
            }
        )

        await self.client.set(report_status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)

    async def mark_failed(
        self,
        task_id: str,
        error: str,
        *,
        evaluation_expected_count: int | None = None,
        evaluation_completed_count: int | None = None,
        evaluation_failed_count: int | None = None,
    ) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        manifest = await self._require_manifest(task.interviewId)
        now = self.now()
        status = ReportGenerationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "status": "failed",
                "attempts": current_status.attempts if current_status else manifest.attempts,
                "createdAt": current_status.createdAt if current_status else manifest.createdAt,
                "startedAt": current_status.startedAt if current_status else manifest.startedAt,
                "completedAt": now,
                "lastError": error,
            }
        )
        next_manifest = InterviewReportManifest.model_validate(
            {
                **manifest.model_dump(),
                "status": "failed",
                "evaluationExpectedCount": _coalesce(
                    evaluation_expected_count,
                    manifest.evaluationExpectedCount,
                ),
                "evaluationCompletedCount": _coalesce(
                    evaluation_completed_count,
                    manifest.evaluationCompletedCount,
                ),
                "evaluationFailedCount": _coalesce(
                    evaluation_failed_count,
                    manifest.evaluationFailedCount,
                ),
                "markdownAvailable": False,
                "lastError": error,
                "completedAt": now,
                "updatedAt": now,
            }
        )

        await self.client.set(report_status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)

    async def retry_task(self, task_id: str, error: str | None = None) -> None:
        task = await self._require_task(task_id)
        current_status = await self.read_task_status(task_id)
        manifest = await self._require_manifest(task.interviewId)
        now = self.now()
        status = ReportGenerationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": task_id,
                "interviewId": task.interviewId,
                "status": "pending",
                "attempts": current_status.attempts if current_status else manifest.attempts,
                "createdAt": current_status.createdAt if current_status else manifest.createdAt,
                "lastError": error or (current_status.lastError if current_status else None),
            }
        )
        next_manifest = InterviewReportManifest.model_validate(
            {
                **manifest.model_dump(),
                "status": "pending",
                "markdownAvailable": False,
                "lastError": error or manifest.lastError,
                "completedAt": None,
                "updatedAt": now,
            }
        )

        await self.client.set(report_status_key(task.interviewId, task_id), _serialize(status))
        await self._write_manifest(next_manifest)
        await self.client.rpush(REPORT_GENERATION_PENDING_QUEUE_KEY, task_id)

    async def read_task(self, task_id: str) -> ReportGenerationTask | None:
        interview_id = await self.client.get(report_task_interview_index_key(task_id))
        if not interview_id:
            return None
        return await _parse_json(
            await self.client.get(report_task_key(interview_id, task_id)),
            ReportGenerationTask,
        )

    async def read_task_status(self, task_id: str) -> ReportGenerationTaskStatus | None:
        interview_id = await self.client.get(report_task_interview_index_key(task_id))
        if not interview_id:
            return None
        return await _parse_json(
            await self.client.get(report_status_key(interview_id, task_id)),
            ReportGenerationTaskStatus,
        )

    async def read_manifest(self, interview_id: str) -> InterviewReportManifest | None:
        return await _parse_json(
            await self.client.get(report_manifest_key(interview_id)),
            InterviewReportManifest,
        )

    async def mark_read(self, interview_id: str, read_at: str) -> None:
        await self.client.set(
            report_read_key(interview_id),
            json.dumps({"readAt": read_at}, ensure_ascii=False),
        )

    async def _require_task(self, task_id: str) -> ReportGenerationTask:
        task = await self.read_task(task_id)
        if not task:
            raise ValueError(f"Report generation task {task_id} was not found.")
        return task

    async def _require_manifest(self, interview_id: str) -> InterviewReportManifest:
        manifest = await self.read_manifest(interview_id)
        if not manifest:
            raise ValueError(f"Report generation manifest {interview_id} was not found.")
        return manifest

    async def _write_manifest(self, manifest: InterviewReportManifest) -> None:
        await self.client.set(report_manifest_key(manifest.interviewId), _serialize(manifest))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _coalesce(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


def _serialize(value: BaseModel) -> str:
    return value.model_dump_json(exclude_none=True)


async def _parse_json(raw: str | None, model: type[BaseModel]) -> Any:
    if not raw:
        return None
    return model.model_validate(json.loads(raw))
