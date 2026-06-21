from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from app.schemas.interview_state import InterviewSessionState, InterviewTopicNodeState

TEXT_SEGMENT_LIMIT = 1200
DUPLICATE_TOKEN_OVERLAP_THRESHOLD = 0.86


class ResumeMemorySummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    professionalSkills: str
    projectExperience: str
    jobDescription: str


class HistoricalReportMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reportExcerpts: list[str] = []
    weaknesses: list[str] = []
    missingPoints: list[str] = []
    improvementAdvice: list[str] = []
    reinforcementQuestionHints: list[str] = []


class FollowUpMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resumeSummary: ResumeMemorySummary
    askedFollowUpQuestions: list[str]
    currentMainQuestion: str
    historicalReportMemory: HistoricalReportMemory


def build_follow_up_memory_snapshot(
    state: InterviewSessionState,
    active_node: InterviewTopicNodeState,
) -> FollowUpMemorySnapshot:
    """Build the bounded memory context used by dedicated follow-up generation."""

    resume_context = state.resumeContext
    explicit_memory = state.followUpMemory
    return FollowUpMemorySnapshot(
        resumeSummary=ResumeMemorySummary(
            professionalSkills=_compact_text(
                explicit_memory.resumeDigest or resume_context.professionalSkills
            ),
            projectExperience=_compact_text(resume_context.projectExperience),
            jobDescription=_compact_text(
                explicit_memory.jobDescriptionDigest or resume_context.jobDescription,
                empty_value="not provided",
            ),
        ),
        askedFollowUpQuestions=explicit_memory.askedQuestions
        or _collect_asked_follow_up_questions(state),
        currentMainQuestion=_compact_text(active_node.mainQuestion),
        historicalReportMemory=_build_historical_report_memory(state),
    )


def normalize_question_text(value: str | None) -> str:
    normalized = " ".join((value or "").lower().split())
    normalized = normalized.replace("？", "?")
    normalized = re.sub(r"[\s?？！!。.,，;；:：'\"“”‘’（）()\[\]【】{}<>《》、]+", "", normalized)
    return normalized


def is_duplicate_follow_up_question(candidate: str | None, memory: FollowUpMemorySnapshot) -> bool:
    candidate_text = normalize_question_text(candidate)
    if not candidate_text:
        return False

    candidate_tokens = _question_tokens(candidate or "")
    for asked_question in memory.askedFollowUpQuestions:
        asked_text = normalize_question_text(asked_question)
        if asked_text and candidate_text == asked_text:
            return True
        if _token_overlap(candidate_tokens, _question_tokens(asked_question)) >= (
            DUPLICATE_TOKEN_OVERLAP_THRESHOLD
        ):
            return True
    return False


def _collect_asked_follow_up_questions(state: InterviewSessionState) -> list[str]:
    questions: list[str] = []
    for round_state in state.rounds:
        for node in round_state.nodes:
            for follow_up in node.followUps:
                question = _compact_text(follow_up.question, empty_value="")
                if follow_up.status in {"asked", "answered"} and question:
                    questions.append(question)
    return questions


def _build_historical_report_memory(state: InterviewSessionState) -> HistoricalReportMemory:
    memory = state.historicalMemory
    if not memory.hasMemory:
        return HistoricalReportMemory()
    return HistoricalReportMemory(
        weaknesses=_compact_list(memory.weaknesses),
        missingPoints=_compact_list(memory.missingPoints),
        improvementAdvice=_compact_list(memory.improvementAdvice),
        reinforcementQuestionHints=_compact_list(memory.reinforcementQuestionHints),
    )


def _question_tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.lower())
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", value)
    return set(words + cjk_chars)


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0
    return len(left & right) / len(left | right)


def _compact_text(
    value: str | None,
    *,
    limit: int = TEXT_SEGMENT_LIMIT,
    empty_value: str = "",
) -> str:
    normalized = " ".join((value or "").split())
    if not normalized:
        return empty_value
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip()


def _compact_list(values: list[str], *, limit: int = TEXT_SEGMENT_LIMIT) -> list[str]:
    return [_compact_text(value, limit=limit) for value in values if _compact_text(value)]
