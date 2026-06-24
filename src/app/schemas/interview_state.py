from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

INTERVIEW_STATE_VERSION = 1
PROFESSIONAL_NODE_COUNT = 6
PROJECT_NODE_COUNT = 2
DEFAULT_PROFESSIONAL_QUESTION_COUNT = PROFESSIONAL_NODE_COUNT
DEFAULT_PROJECT_QUESTION_COUNT = PROJECT_NODE_COUNT
MAX_TOTAL_QUESTION_COUNT = 10
PROFESSIONAL_MAX_FOLLOW_UPS = 3
PROJECT_MAX_FOLLOW_UPS = 2
MAX_DETOUR_RESPONSES = 2

ResponseLanguage = Literal["zh", "en"]
SessionPhase = Literal[
    "intro",
    "professional-skills-round",
    "project-experience-round",
    "wrap-up",
    "completed",
]
RoundType = Literal["professional-skills", "project-experience"]
RoundStatus = Literal["pending", "in-progress", "completed", "skipped"]
TopicNodeStatus = Literal[
    "pending",
    "asking-main-question",
    "awaiting-main-answer",
    "asking-follow-up",
    "awaiting-follow-up-answer",
    "detour-handling",
    "evaluating",
    "completed",
    "skipped",
]
FollowUpIntent = Literal["breadth", "depth", "accuracy", "experience"]
FollowUpStatus = Literal["pending", "asked", "answered", "abandoned"]
AnswerTargetType = Literal["main-question", "follow-up"]
AnswerClassification = Literal[
    "direct-answer",
    "partial-answer",
    "deep-answer",
    "off-topic",
    "clarification-request",
    "skip-request",
    "stop-request",
    "meta-question",
]
TopicSource = Literal["resume", "knowledge-base", "setup", "generated"]
ProfessionalQuestionMode = Literal["per-skill-default", "custom-count"]
DirectionSource = Literal["preset", "custom", "derived"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class InterviewQuestionCandidate(ContractModel):
    id: str
    text: str = Field(min_length=1)
    score: float = 0
    role: str | None = None
    company: str | None = None
    questionType: str | None = None
    difficulty: int | str | None = None
    skillArea: list[str] | None = None
    answer: str | None = None
    tags: str | None = None
    answerPoints: list[str] | None = None
    skills: list[str] | None = None
    level: str | None = None
    jobFamily: str | None = None
    jobDuties: list[str] | None = None
    language: str | None = None
    embeddingText: str | None = None
    source: str | None = None
    isActive: bool | None = None
    userId: str | None = None
    selectionScore: float | None = None


class InterviewSystemSettings(ContractModel):
    reviewIncorrectOrMissingPoints: bool
    skipProfessionalSkillsRound: bool
    skipProjectExperienceRound: bool
    enableFlowTestMode: bool
    enableHistoricalMemory: bool = True
    professionalQuestionMode: ProfessionalQuestionMode
    professionalQuestionCount: int = Field(ge=0, le=MAX_TOTAL_QUESTION_COUNT)
    projectQuestionCount: int = Field(ge=0, le=MAX_TOTAL_QUESTION_COUNT)


class AnswerScore(ContractModel):
    relevance: float = Field(ge=0, le=10)
    accuracy: float = Field(ge=0, le=10)
    depth: float = Field(ge=0, le=10)
    specificity: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    weightedTotal: float = Field(ge=0, le=10)


class AnswerAttemptState(ContractModel):
    id: str
    targetType: AnswerTargetType
    targetId: str
    userMessage: str
    classification: AnswerClassification
    score: AnswerScore | None
    strengths: list[str]
    missingPoints: list[str]
    incorrectPoints: list[str]
    isDetour: bool
    createdAt: str


class FollowUpState(ContractModel):
    id: str
    index: int = Field(ge=0)
    intent: FollowUpIntent
    question: str
    status: FollowUpStatus
    linkedAnswerId: str | None


class TopicSummary(ContractModel):
    strengths: list[str]
    weaknesses: list[str]
    missingPoints: list[str]
    improvementAdvice: list[str]
    evidence: list[str]


class InterviewTopicNodeState(ContractModel):
    id: str
    topic: str
    source: TopicSource
    mainQuestion: str
    referenceAnswer: str | None = None
    evaluationPoints: list[str] | None = None
    status: TopicNodeStatus
    currentTargetType: AnswerTargetType
    currentFollowUpId: str | None
    followUpCount: int = Field(ge=0)
    maxFollowUps: int = Field(gt=0)
    detourResponseCount: int = Field(ge=0)
    earlyCompletionReason: str | None
    followUps: list[FollowUpState]
    answerAttempts: list[AnswerAttemptState]
    aggregatedScore: float | None = Field(default=None, ge=0, le=10)
    summary: TopicSummary | None


class InterviewRoundState(ContractModel):
    id: str
    type: RoundType
    status: RoundStatus
    plannedNodeCount: int = Field(ge=0)
    completedNodeCount: int = Field(ge=0)
    activeNodeId: str | None
    nodeOrder: list[str]
    nodes: list[InterviewTopicNodeState]


class InterviewSetup(ContractModel):
    selectedDirection: str
    directionSource: DirectionSource
    settings: InterviewSystemSettings


class ResumeContext(ContractModel):
    professionalSkills: str
    projectExperience: str
    jobDescription: str
    resumeParsed: bool


class FollowUpMemoryState(ContractModel):
    askedQuestions: list[str] = Field(default_factory=list)
    resumeDigest: str = ""
    jobDescriptionDigest: str = ""
    updatedAt: str | None = None


class HistoricalInterviewMemoryProfileState(ContractModel):
    stableWeaknesses: list[str] = Field(default_factory=list)
    improvedAreas: list[str] = Field(default_factory=list)
    recurringMistakes: list[str] = Field(default_factory=list)
    updatedAt: str | None = None


class HistoricalInterviewMemoryState(ContractModel):
    hasMemory: bool = False
    sourceInterviewIds: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missingPoints: list[str] = Field(default_factory=list)
    improvementAdvice: list[str] = Field(default_factory=list)
    reinforcementQuestionHints: list[str] = Field(default_factory=list)
    profile: HistoricalInterviewMemoryProfileState = Field(
        default_factory=HistoricalInterviewMemoryProfileState
    )


class InterviewSessionState(ContractModel):
    version: Literal[1]
    threadId: str
    targetRole: str
    company: str | None
    responseLanguage: ResponseLanguage
    phase: SessionPhase
    activeRoundId: str | None
    finalReportReady: bool
    finalReport: str | None
    setup: InterviewSetup
    resumeContext: ResumeContext
    followUpMemory: FollowUpMemoryState = Field(default_factory=FollowUpMemoryState)
    historicalMemory: HistoricalInterviewMemoryState = Field(
        default_factory=HistoricalInterviewMemoryState
    )
    lastCorrectionSummary: str | None
    rounds: list[InterviewRoundState]
