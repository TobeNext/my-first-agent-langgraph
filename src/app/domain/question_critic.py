from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.question_generator import fallback_professional_question, fallback_project_question
from app.domain.question_planner import ProfessionalQuestionPlan
from app.schemas.interview_state import InterviewQuestionCandidate


@dataclass(frozen=True)
class QuestionJudgeRecord:
    roundType: str
    questionId: str
    originalQuestionText: str
    finalQuestionText: str
    verdict: str
    failureReasons: list[str]


@dataclass(frozen=True)
class JudgeInitializationQuestionSetResult:
    professionalQuestions: list[InterviewQuestionCandidate]
    projectQuestions: list[InterviewQuestionCandidate]
    judgeTrace: list[QuestionJudgeRecord]


def judge_initialization_question_set(
    *,
    professional_question_plan: list[ProfessionalQuestionPlan],
    professional_questions: list[InterviewQuestionCandidate],
    project_questions: list[InterviewQuestionCandidate],
    normalized_project_topics: list[str],
    target_role: str,
) -> JudgeInitializationQuestionSetResult:
    seen: set[str] = set()
    professional_results = [
        _judge_professional(
            question,
            professional_question_plan[index] if index < len(professional_question_plan) else None,
            seen,
            target_role,
        )
        for index, question in enumerate(professional_questions)
    ]
    project_results = [
        _judge_project(question, normalized_project_topics, seen) for question in project_questions
    ]
    return JudgeInitializationQuestionSetResult(
        professionalQuestions=[item[0] for item in professional_results],
        projectQuestions=[item[0] for item in project_results],
        judgeTrace=[
            *[item[1] for item in professional_results],
            *[item[1] for item in project_results],
        ],
    )


def _judge_professional(
    question: InterviewQuestionCandidate,
    plan: ProfessionalQuestionPlan | None,
    seen: set[str],
    target_role: str,
) -> tuple[InterviewQuestionCandidate, QuestionJudgeRecord]:
    reasons: list[str] = []
    normalized = _normalize(question.text)
    if len(normalized) < 8:
        reasons.append("question-too-short")
    if normalized and normalized in seen:
        reasons.append("duplicate-question")
    if (
        plan
        and plan.questionType == "scenario"
        and not _includes_any(question.text, ["如何", "场景", "设计", "取舍", "结合", "scenario"])
    ):
        reasons.append("scenario-shape-mismatch")
    final = question if not reasons else _fallback_professional_question(plan, target_role)
    seen.add(_normalize(final.text))
    return final, QuestionJudgeRecord(
        roundType="professional-skills",
        questionId=question.id,
        originalQuestionText=question.text,
        finalQuestionText=final.text,
        verdict="accepted" if not reasons else "fallback",
        failureReasons=reasons,
    )


def _judge_project(
    question: InterviewQuestionCandidate,
    topics: list[str],
    seen: set[str],
) -> tuple[InterviewQuestionCandidate, QuestionJudgeRecord]:
    reasons: list[str] = []
    normalized = _normalize(question.text)
    if len(normalized) < 8:
        reasons.append("question-too-short")
    if normalized and normalized in seen:
        reasons.append("duplicate-question")
    if not _includes_any(question.text, ["项目", "project", "经历", "负责"]):
        reasons.append("project-shape-mismatch")
    final = question if not reasons else fallback_project_question(topics[0] if topics else None)
    seen.add(_normalize(final.text))
    return final, QuestionJudgeRecord(
        roundType="project-experience",
        questionId=question.id,
        originalQuestionText=question.text,
        finalQuestionText=final.text,
        verdict="accepted" if not reasons else "fallback",
        failureReasons=reasons,
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _includes_any(text: str, signals: list[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(signal) in normalized for signal in signals)


def _fallback_professional_question(
    plan: ProfessionalQuestionPlan | None,
    target_role: str,
) -> InterviewQuestionCandidate:
    if plan:
        return fallback_professional_question(plan, target_role)
    return InterviewQuestionCandidate.model_validate(
        {
            "id": "fallback-professional-general",
            "text": (
                "请结合你真实做过的项目，详细说明你最熟悉的一项专业能力是如何落地、"
                "排障和优化的。"
            ),
            "score": 0,
            "role": "professional-skills",
            "questionType": "knowledge-check",
            "difficulty": "medium",
            "skillArea": None,
        }
    )


def _fallback_plan() -> ProfessionalQuestionPlan:
    from app.domain.question_planner import ProfessionalQuestionPlan

    return ProfessionalQuestionPlan(
        kind="broad-professional-scenario",
        primarySkill=None,
        relatedSkills=[],
        lens="implementation-depth",
        targetAbility="专业能力",
        questionType="knowledge-check",
        coverageIntent="implementation-depth",
        resumeSignals=[],
        jobDescriptionSignals=[],
        questionDriver="resume",
        expectedDifficulty="medium",
        selectionReason="Fallback professional plan.",
    )
