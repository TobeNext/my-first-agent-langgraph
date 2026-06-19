from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.answer_evaluation import AnswerEvaluationTargetType
from app.schemas.interview_state import ResponseLanguage

INTERVIEW_REPORT_SCHEMA_VERSION = 1


class InterviewReportContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


ReportGenerationStatusValue = Literal["pending", "running", "succeeded", "failed"]
InterviewReportStateValue = Literal["not-started", "generating", "ready", "failed"]
InterviewReportBlockingReason = Literal[
    "manifest-missing",
    "not-sealed",
    "pending",
    "failed",
    "timeout",
]


class ReportGenerationTask(InterviewReportContractModel):
    schemaVersion: Literal[1]
    taskId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    resourceId: str | None = None
    targetRole: str = Field(min_length=1)
    responseLanguage: ResponseLanguage
    evaluationManifestKey: str = Field(min_length=1)
    createdAt: str


class ReportGenerationTaskStatus(InterviewReportContractModel):
    schemaVersion: Literal[1]
    taskId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    status: ReportGenerationStatusValue
    attempts: int = Field(ge=0)
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    lastError: str | None = None


class InterviewReportManifest(InterviewReportContractModel):
    schemaVersion: Literal[1]
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    taskId: str = Field(min_length=1)
    status: ReportGenerationStatusValue
    evaluationExpectedCount: int = Field(ge=0)
    evaluationCompletedCount: int = Field(ge=0)
    evaluationFailedCount: int = Field(ge=0)
    reportId: str | None = None
    markdownAvailable: bool
    attempts: int = Field(ge=0)
    lastError: str | None = None
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    updatedAt: str


class InterviewReportSummary(InterviewReportContractModel):
    overallScore: float = Field(ge=0, le=10)
    overallComment: str = Field(min_length=1)
    strengths: list[str]
    improvementPriorities: list[str]


class InterviewReportQuestionReview(InterviewReportContractModel):
    questionId: str = Field(min_length=1)
    attemptId: str = Field(min_length=1)
    targetType: AnswerEvaluationTargetType
    question: str = Field(min_length=1)
    score: float = Field(ge=0, le=10)
    comment: str = Field(min_length=1)
    missingPoints: list[str]
    improvementAdvice: list[str]


class ReportGenerationOutput(InterviewReportContractModel):
    summary: InterviewReportSummary
    questionReviews: list[InterviewReportQuestionReview]
    markdown: str = Field(min_length=1)


class InterviewReportStatus(InterviewReportContractModel):
    threadId: str = Field(min_length=1)
    reportState: InterviewReportStateValue
    sealed: bool
    expectedCount: int = Field(ge=0)
    completedCount: int = Field(ge=0)
    failedCount: int = Field(ge=0)
    unreadCount: int = Field(ge=0)
    markdownAvailable: bool
    reportId: str | None
    updatedAt: str | None
    blockingReason: InterviewReportBlockingReason | None = None


@dataclass(frozen=True)
class InterviewReportItemWrite:
    id: str
    task_id: str
    attempt_id: str
    node_id: str
    round_id: str
    round_type: str
    target_type: str
    question: str
    candidate_answer: str
    score: float
    comment: str
    missing_points_json: str
    improvement_advice_json: str


@dataclass(frozen=True)
class InterviewReportWrite:
    id: str
    interview_id: str
    thread_id: str
    target_role: str
    response_language: str
    status: str
    overall_score: float | None
    markdown: str
    structured_json: str
    prompt_version: str
    model_name: str
    source_evaluation_manifest_json: str
    created_at: str
    updated_at: str
    completed_at: str | None
    items: list[InterviewReportItemWrite]


@dataclass(frozen=True)
class InterviewReportRecord:
    id: str
    interview_id: str
    thread_id: str
    target_role: str
    response_language: str
    status: str
    overall_score: float | None
    markdown: str
    structured_json: str
    prompt_version: str
    model_name: str
    source_evaluation_manifest_json: str
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class InterviewReportItemRecord:
    id: str
    report_id: str
    interview_id: str
    task_id: str
    attempt_id: str
    node_id: str
    round_id: str
    round_type: str
    target_type: str
    question: str
    candidate_answer: str
    score: float
    comment: str
    missing_points_json: str
    improvement_advice_json: str
    created_at: str


@dataclass(frozen=True)
class InterviewReportReadReceipt:
    id: str
    interview_id: str
    thread_id: str
    read_at: str
