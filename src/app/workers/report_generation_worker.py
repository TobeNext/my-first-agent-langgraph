from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from app.config import get_settings
from app.domain.report_generation import (
    REPORT_GENERATION_PROMPT_VERSION,
    build_report_generation_prompt,
    generate_report_with_model,
)
from app.integrations.redis_client import (
    create_redis_answer_evaluation_store,
    create_redis_report_generation_store,
)
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.answer_evaluation import (
    AnswerEvaluationTask,
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)
from app.schemas.interview_report import (
    InterviewReportItemWrite,
    InterviewReportRecord,
    InterviewReportWrite,
    ReportGenerationOutput,
    ReportGenerationTask,
    ReportGenerationTaskStatus,
)

DEFAULT_REPORT_GENERATION_MAX_ATTEMPTS = 3
DEFAULT_REPORT_GENERATION_POLL_INTERVAL_SECONDS = 1.0

ReportGenerationWorkerStatus = Literal["succeeded", "retrying", "failed"]


class ReportGenerationStoreLike(Protocol):
    async def claim_next_task(self) -> ReportGenerationTask | None: ...

    async def mark_succeeded(
        self,
        task_id: str,
        *,
        report_id: str,
        evaluation_expected_count: int,
        evaluation_completed_count: int,
        evaluation_failed_count: int,
        markdown_available: bool = True,
    ) -> None: ...

    async def mark_failed(
        self,
        task_id: str,
        error: str,
        *,
        evaluation_expected_count: int | None = None,
        evaluation_completed_count: int | None = None,
        evaluation_failed_count: int | None = None,
    ) -> None: ...

    async def retry_task(self, task_id: str, error: str | None = None) -> None: ...

    async def read_task_status(self, task_id: str) -> ReportGenerationTaskStatus | None: ...


class AnswerEvaluationStoreLike(Protocol):
    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None: ...

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]: ...

    async def read_task(self, task_id: str) -> AnswerEvaluationTask | None: ...


class InterviewReportRepositoryLike(Protocol):
    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None: ...

    def write_report(self, report: InterviewReportWrite) -> InterviewReportRecord: ...


ReportGenerationModelEvaluator = Callable[
    [str, ReportGenerationTask],
    Awaitable[ReportGenerationOutput | dict[str, Any]],
]


@dataclass(frozen=True)
class ReportGenerationWorkerTickResult:
    processed: bool
    taskId: str | None = None
    status: ReportGenerationWorkerStatus | None = None
    attempts: int | None = None
    error: str | None = None
    output: ReportGenerationOutput | None = None


class ReportGenerationRunner:
    def __init__(
        self,
        *,
        store: ReportGenerationStoreLike,
        evaluation_store: AnswerEvaluationStoreLike,
        repository: InterviewReportRepositoryLike,
        evaluator: ReportGenerationModelEvaluator | None = None,
        now: Callable[[], str] | None = None,
        max_attempts: int = DEFAULT_REPORT_GENERATION_MAX_ATTEMPTS,
        prompt_version: str = REPORT_GENERATION_PROMPT_VERSION,
    ) -> None:
        self.store = store
        self.evaluation_store = evaluation_store
        self.repository = repository
        self.evaluator = evaluator or _default_evaluator
        self.now = now or _utc_now
        self.max_attempts = max_attempts
        self.prompt_version = prompt_version
        self.model_name = get_settings().model_name

    async def run_once(self) -> ReportGenerationWorkerTickResult:
        task = await self.store.claim_next_task()
        if not task:
            return ReportGenerationWorkerTickResult(processed=False)

        try:
            manifest = await self.evaluation_store.read_manifest(task.interviewId)
            if not manifest:
                return await self._retry_task(task, "evaluation manifest missing")
            if manifest.failedTaskIds:
                message = f"evaluation failed tasks: {', '.join(manifest.failedTaskIds)}"
                await self.store.mark_failed(
                    task.taskId,
                    message,
                    evaluation_expected_count=len(manifest.expectedTaskIds),
                    evaluation_completed_count=len(manifest.completedTaskIds),
                    evaluation_failed_count=len(manifest.failedTaskIds),
                )
                return ReportGenerationWorkerTickResult(
                    processed=True,
                    taskId=task.taskId,
                    status="failed",
                    error=message,
                )
            if not manifest.sealed:
                return await self._retry_task(task, "evaluation manifest is not sealed")
            if len(manifest.completedTaskIds) < len(manifest.expectedTaskIds):
                return await self._retry_task(task, "evaluation results are still pending")

            existing_report = self.repository.get_report_by_interview_id(task.interviewId)
            if (
                existing_report
                and existing_report.status == "succeeded"
                and existing_report.markdown
            ):
                await self._mark_succeeded_from_manifest(task, manifest, existing_report.id)
                return ReportGenerationWorkerTickResult(
                    processed=True,
                    taskId=task.taskId,
                    status="succeeded",
                )

            results = await self.evaluation_store.read_results(task.interviewId)
            result_by_task_id = {result.taskId: result for result in results}
            if len(result_by_task_id) < len(manifest.expectedTaskIds):
                return await self._retry_task(task, "evaluation results are still pending")

            evaluation_tasks = [
                await self.evaluation_store.read_task(task_id)
                for task_id in manifest.expectedTaskIds
            ]
            tasks = [item for item in evaluation_tasks if item is not None]
            if len(tasks) < len(manifest.expectedTaskIds):
                return await self._retry_task(task, "evaluation task context is missing")

            prompt = self.build_prompt(
                task,
                manifest=manifest,
                evaluation_tasks=tasks,
                evaluation_results=results,
            )
            raw_output = await self.evaluator(prompt, task)
            output = ReportGenerationOutput.model_validate(raw_output)
            report = self._build_report_write(
                task=task,
                manifest=manifest,
                evaluation_tasks=tasks,
                evaluation_results=results,
                output=output,
            )
            stored_report = self.repository.write_report(report)
            await self._mark_succeeded_from_manifest(task, manifest, stored_report.id)
            return ReportGenerationWorkerTickResult(
                processed=True,
                taskId=task.taskId,
                status="succeeded",
                output=output,
            )
        except Exception as exc:
            return await self._retry_or_fail_task(task, str(exc))

    def build_prompt(
        self,
        task: ReportGenerationTask,
        *,
        manifest: InterviewEvaluationManifest | None = None,
        evaluation_tasks: list[AnswerEvaluationTask] | None = None,
        evaluation_results: list[LlmAnswerEvaluationResult] | None = None,
    ) -> str:
        return build_report_generation_prompt(
            task=task,
            interview_metadata={
                "interviewId": task.interviewId,
                "threadId": task.threadId,
                "targetRole": task.targetRole,
                "responseLanguage": task.responseLanguage,
                "promptVersion": self.prompt_version,
                "modelName": self.model_name,
                "evaluationExpectedCount": len(manifest.expectedTaskIds) if manifest else 0,
            },
            evaluation_results=[
                result.model_dump(mode="json") for result in (evaluation_results or [])
            ],
            question_answer_context=[
                _build_question_answer_context_item(item)
                for item in (evaluation_tasks or [])
            ],
        )

    async def _retry_task(
        self,
        task: ReportGenerationTask,
        message: str,
    ) -> ReportGenerationWorkerTickResult:
        await self.store.retry_task(task.taskId, message)
        status = await self.store.read_task_status(task.taskId)
        return ReportGenerationWorkerTickResult(
            processed=True,
            taskId=task.taskId,
            status="retrying",
            attempts=status.attempts if status else None,
            error=message,
        )

    async def _retry_or_fail_task(
        self,
        task: ReportGenerationTask,
        message: str,
    ) -> ReportGenerationWorkerTickResult:
        status = await self.store.read_task_status(task.taskId)
        attempts = status.attempts if status else 0

        if attempts >= self.max_attempts:
            await self.store.mark_failed(task.taskId, message)
            return ReportGenerationWorkerTickResult(
                processed=True,
                taskId=task.taskId,
                status="failed",
                attempts=attempts,
                error=message,
            )

        await self.store.retry_task(task.taskId, message)
        return ReportGenerationWorkerTickResult(
            processed=True,
            taskId=task.taskId,
            status="retrying",
            attempts=attempts,
            error=message,
        )

    async def _mark_succeeded_from_manifest(
        self,
        task: ReportGenerationTask,
        manifest: InterviewEvaluationManifest,
        report_id: str,
    ) -> None:
        await self.store.mark_succeeded(
            task.taskId,
            report_id=report_id,
            evaluation_expected_count=len(manifest.expectedTaskIds),
            evaluation_completed_count=len(manifest.completedTaskIds),
            evaluation_failed_count=len(manifest.failedTaskIds),
            markdown_available=True,
        )

    def _build_report_write(
        self,
        *,
        task: ReportGenerationTask,
        manifest: InterviewEvaluationManifest,
        evaluation_tasks: list[AnswerEvaluationTask],
        evaluation_results: list[LlmAnswerEvaluationResult],
        output: ReportGenerationOutput,
    ) -> InterviewReportWrite:
        now = self.now()
        report_id = f"report-{task.interviewId}"
        evaluation_task_by_attempt_id = {item.attemptId: item for item in evaluation_tasks}
        evaluation_result_by_attempt_id = {item.attemptId: item for item in evaluation_results}

        return InterviewReportWrite(
            id=report_id,
            interview_id=task.interviewId,
            thread_id=task.threadId,
            target_role=task.targetRole,
            response_language=task.responseLanguage,
            status="succeeded",
            overall_score=output.summary.overallScore,
            markdown=output.markdown,
            structured_json=output.model_dump_json(exclude_none=True),
            prompt_version=self.prompt_version,
            model_name=self.model_name,
            source_evaluation_manifest_json=manifest.model_dump_json(exclude_none=True),
            created_at=now,
            updated_at=now,
            completed_at=now,
            items=[
                _build_report_item_write(
                    report_id=report_id,
                    interview_id=task.interviewId,
                    review=review.model_dump(),
                    evaluation_task=evaluation_task_by_attempt_id.get(review.attemptId),
                    evaluation_result=evaluation_result_by_attempt_id.get(review.attemptId),
                )
                for review in output.questionReviews
            ],
        )

    async def run_forever(
        self,
        poll_interval_seconds: float = DEFAULT_REPORT_GENERATION_POLL_INTERVAL_SECONDS,
    ) -> None:
        while True:
            result = await self.run_once()
            if not result.processed:
                await asyncio.sleep(poll_interval_seconds)


async def _default_evaluator(
    prompt: str,
    task: ReportGenerationTask,
) -> ReportGenerationOutput:
    return await generate_report_with_model(prompt=prompt, task=task)


def create_default_report_generation_runner() -> ReportGenerationRunner:
    return ReportGenerationRunner(
        store=create_redis_report_generation_store(),
        evaluation_store=create_redis_answer_evaluation_store(),
        repository=InterviewReportRepository(),
    )


def _build_question_answer_context_item(task: AnswerEvaluationTask) -> dict[str, Any]:
    return {
        "taskId": task.taskId,
        "attemptId": task.attemptId,
        "nodeId": task.nodeId,
        "roundId": task.roundId,
        "roundType": task.roundType,
        "targetType": task.targetType,
        "question": task.question,
        "mainQuestion": task.mainQuestion,
        "followUpQuestion": task.followUpQuestion,
        "referenceAnswer": task.referenceAnswer,
        "evaluationPoints": task.evaluationPoints,
        "candidateAnswer": task.candidateAnswer,
        "nodeConversation": [
            item.model_dump(mode="json") for item in task.nodeConversation
        ],
    }


def _build_report_item_write(
    *,
    report_id: str,
    interview_id: str,
    review: dict[str, Any],
    evaluation_task: AnswerEvaluationTask | None,
    evaluation_result: LlmAnswerEvaluationResult | None,
) -> InterviewReportItemWrite:
    attempt_id = str(review["attemptId"])
    task_id = evaluation_result.taskId if evaluation_result else f"unknown-task-{attempt_id}"
    node_id = evaluation_task.nodeId if evaluation_task else str(review["questionId"])
    round_id = evaluation_task.roundId if evaluation_task else "unknown-round"
    round_type = evaluation_task.roundType if evaluation_task else "unknown"
    candidate_answer = evaluation_task.candidateAnswer if evaluation_task else ""
    return InterviewReportItemWrite(
        id=f"{report_id}-item-{attempt_id}",
        task_id=task_id,
        attempt_id=attempt_id,
        node_id=node_id,
        round_id=round_id,
        round_type=round_type,
        target_type=str(review["targetType"]),
        question=str(review["question"]),
        candidate_answer=candidate_answer,
        score=float(review["score"]),
        comment=str(review["comment"]),
        missing_points_json=json.dumps(review["missingPoints"], ensure_ascii=False),
        improvement_advice_json=json.dumps(
            review["improvementAdvice"],
            ensure_ascii=False,
        ),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
