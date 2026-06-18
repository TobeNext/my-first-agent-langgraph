from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.interview_state import (
    AnswerClassification,
    AnswerScore,
    ResponseLanguage,
    RoundType,
)

ANSWER_EVALUATION_SCHEMA_VERSION = 1


class AnswerEvaluationContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


AnswerEvaluationTargetType = Literal["main-question", "follow-up"]
AnswerEvaluationTaskStatusValue = Literal["pending", "running", "succeeded", "failed"]
NodeConversationRole = Literal["interviewer", "candidate"]


class AnswerEvaluationConversationItem(AnswerEvaluationContractModel):
    role: NodeConversationRole
    targetType: AnswerEvaluationTargetType
    text: str
    createdAt: str


class AnswerEvaluationTask(AnswerEvaluationContractModel):
    schemaVersion: Literal[1]
    taskId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    resourceId: str | None = None
    nodeId: str = Field(min_length=1)
    roundId: str = Field(min_length=1)
    roundType: RoundType
    attemptId: str = Field(min_length=1)
    targetType: AnswerEvaluationTargetType
    targetId: str = Field(min_length=1)
    targetRole: str = Field(min_length=1)
    responseLanguage: ResponseLanguage
    question: str = Field(min_length=1)
    mainQuestion: str = Field(min_length=1)
    followUpQuestion: str | None = None
    referenceAnswer: str | None = None
    evaluationPoints: list[str] = Field(default_factory=list)
    candidateAnswer: str = Field(min_length=1)
    nodeConversation: list[AnswerEvaluationConversationItem] = Field(default_factory=list)
    createdAt: str


class AnswerEvaluationTaskStatus(AnswerEvaluationContractModel):
    schemaVersion: Literal[1]
    taskId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    attemptId: str = Field(min_length=1)
    status: AnswerEvaluationTaskStatusValue
    attempts: int = Field(ge=0)
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    lastError: str | None = None


class LlmAnswerEvaluationResult(AnswerEvaluationContractModel):
    schemaVersion: Literal[1]
    taskId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    nodeId: str = Field(min_length=1)
    roundId: str = Field(min_length=1)
    roundType: RoundType
    attemptId: str = Field(min_length=1)
    classification: AnswerClassification
    score: AnswerScore
    strengths: list[str]
    missingPoints: list[str]
    incorrectPoints: list[str]
    shouldAskFollowUp: bool
    followUpFocus: list[str]
    evaluatorModel: str = Field(min_length=1)
    promptVersion: str = Field(min_length=1)
    createdAt: str


class InterviewEvaluationManifest(AnswerEvaluationContractModel):
    schemaVersion: Literal[1]
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    expectedTaskIds: list[str]
    completedTaskIds: list[str]
    failedTaskIds: list[str]
    sealed: bool
    sealedAt: str | None = None
    updatedAt: str
