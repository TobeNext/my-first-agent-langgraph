from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.answer_evaluation import InterviewEvaluationManifest, LlmAnswerEvaluationResult

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_WAIT_SECONDS = 120.0
BlockingReason = Literal["manifest-missing", "not-sealed", "pending", "failed", "timeout"]


class EvaluationReportStoreLike(Protocol):
    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None: ...

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]: ...


class WaitAndReadInterviewEvaluationsOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ready: bool
    sealed: bool
    expectedCount: int = Field(ge=0)
    completedCount: int = Field(ge=0)
    failedCount: int = Field(ge=0)
    evaluations: list[LlmAnswerEvaluationResult]
    waitElapsedMs: int = Field(ge=0)
    blockingReason: BlockingReason | None


async def wait_and_read_interview_evaluations(
    *,
    interview_id: str,
    thread_id: str,
    store: EvaluationReportStoreLike,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> WaitAndReadInterviewEvaluationsOutput:
    now_fn = now or time.monotonic
    sleep_fn = sleep or asyncio.sleep
    started_at = now_fn()

    while True:
        wait_elapsed_ms = max(0, round((now_fn() - started_at) * 1000))
        manifest = await store.read_manifest(interview_id)

        if manifest:
            _validate_manifest_thread(manifest, thread_id)
            if manifest.failedTaskIds:
                return _blocked_output(manifest, wait_elapsed_ms, "failed")

            if manifest.sealed and _is_manifest_complete(manifest):
                evaluations = await store.read_results(interview_id)
                result_by_task_id = {result.taskId: result for result in evaluations}
                ordered = [
                    result_by_task_id[task_id]
                    for task_id in manifest.expectedTaskIds
                    if task_id in result_by_task_id
                ]
                if len(ordered) == len(manifest.expectedTaskIds):
                    return WaitAndReadInterviewEvaluationsOutput.model_validate(
                        {
                            "ready": True,
                            "sealed": True,
                            "expectedCount": len(manifest.expectedTaskIds),
                            "completedCount": len(manifest.completedTaskIds),
                            "failedCount": 0,
                            "evaluations": ordered,
                            "waitElapsedMs": wait_elapsed_ms,
                            "blockingReason": None,
                        }
                    )

        if wait_elapsed_ms >= round(max_wait_seconds * 1000):
            reason: BlockingReason = (
                "manifest-missing"
                if not manifest
                else "pending"
                if manifest.sealed
                else "not-sealed"
            )
            return _blocked_output(manifest, wait_elapsed_ms, reason)

        remaining_seconds = max(0, max_wait_seconds - (now_fn() - started_at))
        await sleep_fn(min(poll_interval_seconds, remaining_seconds))


def _blocked_output(
    manifest: InterviewEvaluationManifest | None,
    wait_elapsed_ms: int,
    blocking_reason: BlockingReason,
) -> WaitAndReadInterviewEvaluationsOutput:
    return WaitAndReadInterviewEvaluationsOutput.model_validate(
        {
            "ready": False,
            "sealed": manifest.sealed if manifest else False,
            "expectedCount": len(manifest.expectedTaskIds) if manifest else 0,
            "completedCount": len(manifest.completedTaskIds) if manifest else 0,
            "failedCount": len(manifest.failedTaskIds) if manifest else 0,
            "evaluations": [],
            "waitElapsedMs": wait_elapsed_ms,
            "blockingReason": blocking_reason,
        }
    )


def _is_manifest_complete(manifest: InterviewEvaluationManifest) -> bool:
    return len(manifest.completedTaskIds) == len(manifest.expectedTaskIds)


def _validate_manifest_thread(manifest: InterviewEvaluationManifest, thread_id: str) -> None:
    if manifest.threadId != thread_id:
        raise ValueError(
            f"Evaluation manifest thread mismatch for interview {manifest.interviewId}: "
            f"expected {thread_id}, found {manifest.threadId}."
        )
