from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.integrations.redis_client import create_redis_report_generation_store
from app.integrations.redis_evaluation_store import manifest_key as evaluation_manifest_key
from app.integrations.redis_report_generation_store import RedisReportGenerationStore
from app.schemas.interview_report import ReportGenerationTask
from app.schemas.interview_state import InterviewSessionState

logger = logging.getLogger(__name__)


def build_report_generation_task(
    *,
    state: InterviewSessionState,
    resource_id: str | None = None,
    now: Callable[[], str] | None = None,
    create_task_id: Callable[[InterviewSessionState], str] | None = None,
) -> ReportGenerationTask:
    created_at = now() if now else _utc_now()
    return ReportGenerationTask.model_validate(
        {
            "schemaVersion": 1,
            "taskId": (
                create_task_id(state)
                if create_task_id
                else f"report-generation-{state.threadId}"
            ),
            "interviewId": state.threadId,
            "threadId": state.threadId,
            "resourceId": resource_id,
            "targetRole": state.targetRole,
            "responseLanguage": state.responseLanguage,
            "evaluationManifestKey": evaluation_manifest_key(state.threadId),
            "createdAt": created_at,
        }
    )


def enqueue_report_generation_task_best_effort(
    *,
    state: InterviewSessionState,
    resource_id: str | None = None,
    store: RedisReportGenerationStore | None = None,
    now: Callable[[], str] | None = None,
    create_task_id: Callable[[InterviewSessionState], str] | None = None,
) -> ReportGenerationTask:
    task = build_report_generation_task(
        state=state,
        resource_id=resource_id,
        now=now,
        create_task_id=create_task_id,
    )

    try:
        asyncio.run(_enqueue_task(task, store))
        logger.info(
            "Report generation task enqueued",
            extra={
                "event": "report_generation.task.enqueued",
                "interviewId": task.interviewId,
                "taskId": task.taskId,
            },
        )
    except Exception as exc:
        logger.warning(
            "Failed to enqueue report generation task",
            extra={
                "event": "report_generation.task.enqueue_failed",
                "interviewId": task.interviewId,
                "taskId": task.taskId,
                "err": str(exc),
            },
        )

    return task


async def _enqueue_task(
    task: ReportGenerationTask,
    store: RedisReportGenerationStore | None,
) -> None:
    resolved_store = store or create_redis_report_generation_store()
    await resolved_store.enqueue_task(task)
    client = getattr(resolved_store, "client", None)
    disconnect = getattr(client, "disconnect", None)
    if store is None and disconnect:
        await disconnect()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
