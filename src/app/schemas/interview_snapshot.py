from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RoundType = Literal["professional-skills", "project-experience"]
SessionPhase = Literal[
    "intro",
    "professional-skills-round",
    "project-experience-round",
    "wrap-up",
    "completed",
]
CurrentStage = Literal["main-question", "follow-up", "completed"]


class InterviewProgressSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    totalQuestionCount: int = Field(ge=0)
    completedQuestionCount: int = Field(ge=0)
    remainingQuestionCount: int = Field(ge=0)
    currentQuestionIndex: int | None = Field(default=None, ge=1)
    currentRoundType: RoundType | None = None
    currentRoundLabel: str | None = None
    currentStage: CurrentStage
    currentFollowUpIndex: int | None = Field(default=None, ge=0)
    currentQuestionText: str | None = None
    currentNodeTopic: str | None = None


class InterviewStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assistantReply: str
    flowTestMockUserReply: str | None
    phase: SessionPhase
    activeRoundType: RoundType | None
    activeNodeTopic: str | None
    finalReportReady: bool
    progress: InterviewProgressSummary
