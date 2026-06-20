from __future__ import annotations

from dataclasses import dataclass

from app.domain.job_description_signals import (
    extract_job_description_signal_set,
    resolve_question_driver,
)
from app.domain.question_planner import ProfessionalQuestionPlan
from app.schemas.interview_state import InterviewQuestionCandidate


@dataclass(frozen=True)
class GeneratedQuestionRecord:
    roundType: str
    source: str
    targetAbility: str | None
    questionType: str
    coverageIntent: str
    questionDriver: str
    resumeSignals: list[str]
    jobDescriptionSignals: list[str]
    expectedDifficulty: str
    questionId: str
    questionText: str
    selectionReason: str


@dataclass(frozen=True)
class GenerateInitializationQuestionSetResult:
    professionalQuestions: list[InterviewQuestionCandidate]
    projectQuestions: list[InterviewQuestionCandidate]
    generationTrace: list[GeneratedQuestionRecord]


def generate_initialization_question_set(
    *,
    professional_question_plan: list[ProfessionalQuestionPlan],
    professional_questions: list[InterviewQuestionCandidate],
    project_questions: list[InterviewQuestionCandidate],
    job_description: str = "",
    normalized_project_topics: list[str] | None = None,
) -> GenerateInitializationQuestionSetResult:
    professional = [
        _normalize(question) for question in professional_questions if question.text.strip()
    ]
    project = [_normalize(question) for question in project_questions if question.text.strip()]
    return GenerateInitializationQuestionSetResult(
        professionalQuestions=professional,
        projectQuestions=project,
        generationTrace=[
            *_professional_trace(professional_question_plan, professional),
            *_project_trace(project, job_description, normalized_project_topics or []),
        ],
    )


def fallback_professional_question(
    plan: ProfessionalQuestionPlan, target_role: str
) -> InterviewQuestionCandidate:
    if plan.kind == "skill-focus" and plan.primarySkill:
        text = (
            f"请结合你真实做过的项目，详细说明你在{plan.primarySkill}上的"
            "实现思路、关键取舍与排障经验。"
        )
        skill_area = [plan.primarySkill]
    elif plan.kind == "jd-gap-scenario":
        text = (
            f"请结合{target_role}岗位，说明你会如何处理“{plan.targetAbility}”"
            "相关场景、风险和验证方式。"
        )
        skill_area = plan.relatedSkills
    else:
        target = "、".join(plan.relatedSkills) or plan.targetAbility
        text = (
            f"请结合你真实做过的项目，说明你如何围绕{target}处理一个复杂场景，"
            "并解释关键取舍、限制和结果。"
        )
        skill_area = plan.relatedSkills
    return InterviewQuestionCandidate.model_validate(
        {
            "id": f"fallback-professional-{abs(hash(text)) % 10_000_000}",
            "text": text,
            "score": 0,
            "role": "professional-skills",
            "questionType": plan.questionType,
            "difficulty": plan.expectedDifficulty,
            "skillArea": skill_area,
        }
    )


def fallback_project_question(topic: str | None) -> InterviewQuestionCandidate:
    text = (
        f"请结合项目“{topic}”，说明项目背景、你的职责、关键决策、遇到的挑战以及最终结果。"
        if topic
        else "请结合一个你负责过的项目，说明项目背景、你的职责、关键决策、遇到的挑战以及最终结果。"
    )
    return InterviewQuestionCandidate.model_validate(
        {
            "id": f"fallback-project-{abs(hash(text)) % 10_000_000}",
            "text": text,
            "score": 0,
            "role": "project-experience",
            "questionType": "project-deep-dive",
            "difficulty": "medium",
            "skillArea": [topic] if topic else None,
        }
    )


def _normalize(question: InterviewQuestionCandidate) -> InterviewQuestionCandidate:
    return question.model_copy(update={"text": question.text.strip()}, deep=True)


def _question_source(question: InterviewQuestionCandidate) -> str:
    return "fallback" if question.id.startswith("fallback-") else "retrieved"


def _professional_trace(
    plans: list[ProfessionalQuestionPlan],
    questions: list[InterviewQuestionCandidate],
) -> list[GeneratedQuestionRecord]:
    records: list[GeneratedQuestionRecord] = []
    for index, question in enumerate(questions):
        plan = plans[index] if index < len(plans) else None
        records.append(
            GeneratedQuestionRecord(
                roundType="professional-skills",
                source=_question_source(question),
                targetAbility=plan.targetAbility if plan else None,
                questionType=plan.questionType if plan else "knowledge-check",
                coverageIntent=plan.coverageIntent if plan else "professional-skills-context",
                questionDriver=plan.questionDriver if plan else "resume",
                resumeSignals=plan.resumeSignals if plan else [],
                jobDescriptionSignals=plan.jobDescriptionSignals if plan else [],
                expectedDifficulty=plan.expectedDifficulty if plan else "medium",
                questionId=question.id,
                questionText=question.text,
                selectionReason=plan.selectionReason if plan else "Adapted a retrieved question.",
            )
        )
    return records


def _project_trace(
    questions: list[InterviewQuestionCandidate],
    job_description: str,
    project_topics: list[str],
) -> list[GeneratedQuestionRecord]:
    signal_set = extract_job_description_signal_set(
        job_description=job_description,
        project_topics=project_topics,
    )
    jd_signals = (signal_set.alignedSignals or signal_set.topSignals)[:3]
    driver = resolve_question_driver(
        has_resume_signals=bool(project_topics),
        has_job_description_signals=bool(jd_signals),
    )
    return [
        GeneratedQuestionRecord(
            roundType="project-experience",
            source=_question_source(question),
            targetAbility=None,
            questionType="project-deep-dive",
            coverageIntent="project-experience-context",
            questionDriver=driver,
            resumeSignals=project_topics[:3],
            jobDescriptionSignals=jd_signals,
            expectedDifficulty="medium",
            questionId=question.id,
            questionText=question.text,
            selectionReason=(
                "Adapted a project-experience candidate into the final main-question set."
            ),
        )
        for question in questions
    ]
