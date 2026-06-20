from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.domain.answer_evaluation_runtime import AnswerEvaluationContext
from app.domain.report_generation import (
    REPORT_GENERATION_PROMPT_VERSION,
    build_report_generation_prompt,
    generate_report_with_model,
)
from app.schemas.answer_evaluation import LlmAnswerEvaluationResult
from app.schemas.interview_report import (
    InterviewReportItemWrite,
    InterviewReportWrite,
    ReportGenerationOutput,
)
from app.schemas.interview_state import InterviewSessionState, ResponseLanguage


class ReportGenerationContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schemaVersion: int = 1
    generationId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    resourceId: str | None = None
    targetRole: str = Field(min_length=1)
    responseLanguage: ResponseLanguage
    createdAt: str


ReportGenerationModelEvaluator = Callable[
    [str, ReportGenerationContext],
    ReportGenerationOutput | dict[str, Any] | Awaitable[ReportGenerationOutput | dict[str, Any]],
]


def build_report_generation_context_from_session(
    state: InterviewSessionState,
    resource_id: str | None = None,
    *,
    now: Callable[[], str] | None = None,
) -> ReportGenerationContext:
    created_at = now() if now else _utc_now()
    return ReportGenerationContext.model_validate(
        {
            "schemaVersion": 1,
            "generationId": f"inline-report-{state.threadId}",
            "interviewId": state.threadId,
            "threadId": state.threadId,
            "resourceId": resource_id,
            "targetRole": state.targetRole,
            "responseLanguage": state.responseLanguage,
            "createdAt": created_at,
        }
    )


def build_report_prompt_from_session(
    *,
    state: InterviewSessionState,
    evaluation_contexts: list[AnswerEvaluationContext],
    evaluation_results: list[LlmAnswerEvaluationResult],
    resource_id: str | None = None,
    context: ReportGenerationContext | None = None,
    prompt_version: str = REPORT_GENERATION_PROMPT_VERSION,
    model_name: str | None = None,
    now: Callable[[], str] | None = None,
) -> str:
    resolved_context = context or build_report_generation_context_from_session(
        state,
        resource_id,
        now=now,
    )
    return build_report_generation_prompt(
        task=resolved_context,
        interview_metadata={
            "interviewId": resolved_context.interviewId,
            "threadId": resolved_context.threadId,
            "targetRole": resolved_context.targetRole,
            "responseLanguage": resolved_context.responseLanguage,
            "promptVersion": prompt_version,
            "modelName": model_name or get_settings().model_name,
            "evaluationExpectedCount": len(evaluation_contexts),
            "source": "langgraph-inline",
        },
        evaluation_results=[
            result.model_dump(mode="json") for result in evaluation_results
        ],
        question_answer_context=[
            _build_question_answer_context_item(item) for item in evaluation_contexts
        ],
    )


async def generate_report_from_evaluations(
    *,
    state: InterviewSessionState,
    evaluation_contexts: list[AnswerEvaluationContext],
    evaluation_results: list[LlmAnswerEvaluationResult],
    resource_id: str | None = None,
    context: ReportGenerationContext | None = None,
    evaluator: ReportGenerationModelEvaluator | None = None,
    prompt_version: str = REPORT_GENERATION_PROMPT_VERSION,
    model_name: str | None = None,
    now: Callable[[], str] | None = None,
) -> ReportGenerationOutput:
    resolved_context = context or build_report_generation_context_from_session(
        state,
        resource_id,
        now=now,
    )
    prompt = build_report_prompt_from_session(
        state=state,
        evaluation_contexts=evaluation_contexts,
        evaluation_results=evaluation_results,
        context=resolved_context,
        prompt_version=prompt_version,
        model_name=model_name,
    )
    raw_output = (
        await _maybe_await(evaluator(prompt, resolved_context))
        if evaluator
        else await generate_report_with_model(
            prompt=prompt,
            task=resolved_context,
        )
    )
    return ReportGenerationOutput.model_validate(raw_output)


def build_report_write_from_output(
    *,
    state: InterviewSessionState,
    evaluation_contexts: list[AnswerEvaluationContext],
    evaluation_results: list[LlmAnswerEvaluationResult],
    output: ReportGenerationOutput,
    context: ReportGenerationContext | None = None,
    resource_id: str | None = None,
    now: Callable[[], str] | None = None,
    prompt_version: str = REPORT_GENERATION_PROMPT_VERSION,
    model_name: str | None = None,
) -> InterviewReportWrite:
    created_at = now() if now else _utc_now()
    resolved_context = context or build_report_generation_context_from_session(
        state,
        resource_id,
        now=lambda: created_at,
    )
    report_id = f"report-{resolved_context.interviewId}"
    evaluation_context_by_attempt_id = {
        item.attemptId: item for item in evaluation_contexts
    }
    evaluation_result_by_attempt_id = {
        item.attemptId: item for item in evaluation_results
    }
    return InterviewReportWrite(
        id=report_id,
        interview_id=resolved_context.interviewId,
        thread_id=resolved_context.threadId,
        target_role=resolved_context.targetRole,
        response_language=resolved_context.responseLanguage,
        status="succeeded",
        overall_score=output.summary.overallScore,
        markdown=output.markdown,
        structured_json=output.model_dump_json(exclude_none=True),
        prompt_version=prompt_version,
        model_name=model_name or get_settings().model_name,
        source_evaluation_manifest_json=json.dumps(
            {
                "schemaVersion": 1,
                "source": "langgraph-inline",
                "interviewId": resolved_context.interviewId,
                "threadId": resolved_context.threadId,
                "evaluationIds": [item.evaluationId for item in evaluation_contexts],
                "completedEvaluationIds": [
                    item.taskId for item in evaluation_results
                ],
            },
            ensure_ascii=False,
        ),
        created_at=created_at,
        updated_at=created_at,
        completed_at=created_at,
        items=[
            _build_report_item_write(
                report_id=report_id,
                interview_id=resolved_context.interviewId,
                review=review.model_dump(),
                evaluation_context=evaluation_context_by_attempt_id.get(review.attemptId),
                evaluation_result=evaluation_result_by_attempt_id.get(review.attemptId),
            )
            for review in output.questionReviews
        ],
    )


def build_failed_report_write(
    *,
    state: InterviewSessionState,
    error: str,
    resource_id: str | None = None,
    context: ReportGenerationContext | None = None,
    now: Callable[[], str] | None = None,
    prompt_version: str = REPORT_GENERATION_PROMPT_VERSION,
    model_name: str | None = None,
) -> InterviewReportWrite:
    created_at = now() if now else _utc_now()
    resolved_context = context or build_report_generation_context_from_session(
        state,
        resource_id,
        now=lambda: created_at,
    )
    report_id = f"report-{resolved_context.interviewId}"
    return InterviewReportWrite(
        id=report_id,
        interview_id=resolved_context.interviewId,
        thread_id=resolved_context.threadId,
        target_role=resolved_context.targetRole,
        response_language=resolved_context.responseLanguage,
        status="failed",
        overall_score=None,
        markdown="",
        structured_json=json.dumps(
            {
                "schemaVersion": 1,
                "status": "failed",
                "error": error,
            },
            ensure_ascii=False,
        ),
        prompt_version=prompt_version,
        model_name=model_name or get_settings().model_name,
        source_evaluation_manifest_json=json.dumps(
            {
                "schemaVersion": 1,
                "source": "langgraph-inline",
                "interviewId": resolved_context.interviewId,
                "threadId": resolved_context.threadId,
                "status": "failed",
            },
            ensure_ascii=False,
        ),
        created_at=created_at,
        updated_at=created_at,
        completed_at=None,
        items=[],
    )


def _build_question_answer_context_item(
    context: AnswerEvaluationContext,
) -> dict[str, Any]:
    return {
        "evaluationId": context.evaluationId,
        "attemptId": context.attemptId,
        "nodeId": context.nodeId,
        "roundId": context.roundId,
        "roundType": context.roundType,
        "targetType": context.targetType,
        "question": context.question,
        "mainQuestion": context.mainQuestion,
        "followUpQuestion": context.followUpQuestion,
        "referenceAnswer": context.referenceAnswer,
        "evaluationPoints": context.evaluationPoints,
        "candidateAnswer": context.candidateAnswer,
        "nodeConversation": [
            item.model_dump(mode="json") for item in context.nodeConversation
        ],
    }


def _build_report_item_write(
    *,
    report_id: str,
    interview_id: str,
    review: dict[str, Any],
    evaluation_context: AnswerEvaluationContext | None,
    evaluation_result: LlmAnswerEvaluationResult | None,
) -> InterviewReportItemWrite:
    attempt_id = str(review["attemptId"])
    task_id = evaluation_result.taskId if evaluation_result else f"unknown-evaluation-{attempt_id}"
    node_id = evaluation_context.nodeId if evaluation_context else str(review["questionId"])
    round_id = evaluation_context.roundId if evaluation_context else "unknown-round"
    round_type = evaluation_context.roundType if evaluation_context else "unknown"
    candidate_answer = evaluation_context.candidateAnswer if evaluation_context else ""
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


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
