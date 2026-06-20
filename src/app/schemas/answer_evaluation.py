from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.interview_state import (
    AnswerClassification,
    AnswerScore,
    RoundType,
)

ANSWER_EVALUATION_SCHEMA_VERSION = 1


class AnswerEvaluationContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


AnswerEvaluationTargetType = Literal["main-question", "follow-up"]
NodeConversationRole = Literal["interviewer", "candidate"]


class AnswerEvaluationConversationItem(AnswerEvaluationContractModel):
    role: NodeConversationRole
    targetType: AnswerEvaluationTargetType
    text: str
    createdAt: str


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
