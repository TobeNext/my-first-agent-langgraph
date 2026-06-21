from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from app.domain.job_description_signals import (
    QuestionDriver,
    extract_job_description_signal_set,
    resolve_question_driver,
)

ProfessionalQuestionMode = Literal["per-skill-default", "custom-count"]
PlannedQuestionType = Literal["knowledge-check", "scenario"]
PlannedQuestionDifficulty = Literal["medium", "hard"]
ReinforcementIntent = Literal["none", "review-weakness", "verify-improvement"]
ProfessionalQuestionLens = Literal[
    "implementation-depth",
    "trade-off-analysis",
    "failure-recovery",
    "scalability",
    "cross-skill-integration",
    "delivery-prioritization",
]
ProfessionalQuestionKind = Literal[
    "skill-focus",
    "cross-skill-scenario",
    "broad-professional-scenario",
    "jd-gap-scenario",
]

OVERFLOW_LENSES: list[ProfessionalQuestionLens] = [
    "trade-off-analysis",
    "failure-recovery",
    "scalability",
    "cross-skill-integration",
    "delivery-prioritization",
]


@dataclass(frozen=True)
class ProfessionalQuestionPlan:
    kind: ProfessionalQuestionKind
    primarySkill: str | None
    relatedSkills: list[str]
    lens: ProfessionalQuestionLens
    targetAbility: str
    questionType: PlannedQuestionType
    coverageIntent: ProfessionalQuestionLens
    resumeSignals: list[str]
    jobDescriptionSignals: list[str]
    questionDriver: QuestionDriver
    expectedDifficulty: PlannedQuestionDifficulty
    selectionReason: str
    historicalWeaknessSignals: list[str] = field(default_factory=list)
    reinforcementIntent: ReinforcementIntent = "none"


def plan_professional_question_queries(
    *,
    mode: ProfessionalQuestionMode,
    professional_skills: list[str],
    desired_question_count: int,
    job_description: str = "",
    project_topics: list[str] | None = None,
    historical_weakness_signals: list[str] | None = None,
) -> list[ProfessionalQuestionPlan]:
    skills = _unique_skills(professional_skills)
    if not skills or desired_question_count <= 0:
        return []

    signal_set = extract_job_description_signal_set(
        job_description=job_description,
        resume_topics=skills,
        project_topics=project_topics or [],
    )

    def matched_signals(skill: str) -> list[str]:
        normalized_skill = skill.lower()
        return [
            signal
            for signal in signal_set.topSignals
            if normalized_skill in signal.lower() or signal.lower() in normalized_skill
        ]

    if mode == "per-skill-default":
        plans = [
            _skill_focus_plan(skill, mode, matched_signals(skill))
            for skill in skills[:desired_question_count]
        ]
        return _apply_historical_reinforcement(plans, historical_weakness_signals)

    unique_skill_plans = [
        _skill_focus_plan(skill, mode, matched_signals(skill))
        for skill in skills[: min(desired_question_count, len(skills))]
    ]
    overflow_count = max(0, desired_question_count - len(unique_skill_plans))
    gap_plans = [
        _jd_gap_plan(signal, skills, index)
        for index, signal in enumerate(signal_set.gapSignals[:overflow_count])
    ]
    remaining = max(0, overflow_count - len(gap_plans))
    overflow_plans = [
        _overflow_plan(skills, signal_set.alignedSignals, index) for index in range(remaining)
    ]
    plans = [*unique_skill_plans, *gap_plans, *overflow_plans]
    return _apply_historical_reinforcement(plans, historical_weakness_signals)


def _skill_focus_plan(
    skill: str,
    mode: ProfessionalQuestionMode,
    matched_signals: list[str],
) -> ProfessionalQuestionPlan:
    return ProfessionalQuestionPlan(
        kind="skill-focus",
        primarySkill=skill,
        relatedSkills=[],
        lens="implementation-depth",
        targetAbility=skill,
        questionType="knowledge-check",
        coverageIntent="implementation-depth",
        resumeSignals=[skill],
        jobDescriptionSignals=matched_signals,
        questionDriver=resolve_question_driver(
            has_resume_signals=True,
            has_job_description_signals=bool(matched_signals),
        ),
        expectedDifficulty="medium",
        selectionReason=(
            f"Selected {skill} as the canonical resume skill owner."
            if mode == "per-skill-default"
            else f"Selected {skill} as a unique primary skill before overflow allocation."
        ),
    )


def _jd_gap_plan(signal: str, skills: list[str], index: int) -> ProfessionalQuestionPlan:
    lens = OVERFLOW_LENSES[index % len(OVERFLOW_LENSES)]
    related = _cross_skill_group(skills, index)
    return ProfessionalQuestionPlan(
        kind="jd-gap-scenario",
        primarySkill=None,
        relatedSkills=related,
        lens=lens,
        targetAbility=signal,
        questionType="scenario",
        coverageIntent=lens,
        resumeSignals=related,
        jobDescriptionSignals=[signal],
        questionDriver="job-description",
        expectedDifficulty="hard",
        selectionReason=f'Selected JD-only capability gap "{signal}".',
    )


def _overflow_plan(
    skills: list[str],
    aligned_signals: list[str],
    index: int,
) -> ProfessionalQuestionPlan:
    lens = OVERFLOW_LENSES[index % len(OVERFLOW_LENSES)]
    related = _cross_skill_group(skills, index)
    kind: ProfessionalQuestionKind = (
        "cross-skill-scenario" if len(related) >= 2 else "broad-professional-scenario"
    )
    return ProfessionalQuestionPlan(
        kind=kind,
        primarySkill=None,
        relatedSkills=related,
        lens=lens,
        targetAbility=" + ".join(related) or "broader professional context",
        questionType="scenario",
        coverageIntent=lens,
        resumeSignals=related,
        jobDescriptionSignals=aligned_signals[:2],
        questionDriver=resolve_question_driver(
            has_resume_signals=bool(related),
            has_job_description_signals=bool(aligned_signals),
        ),
        expectedDifficulty="hard",
        selectionReason=f"Selected a {lens} scenario for broader coverage.",
    )


def _cross_skill_group(skills: list[str], index: int) -> list[str]:
    if not skills:
        return []
    start = index % len(skills)
    rotated = [*skills[start:], *skills[:start]]
    size = 3 if len(skills) >= 3 and index % 2 == 1 else 2
    return rotated[: min(size, len(skills))]


def _unique_skills(skills: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        normalized = " ".join(skill.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _apply_historical_reinforcement(
    plans: list[ProfessionalQuestionPlan],
    historical_weakness_signals: list[str] | None,
) -> list[ProfessionalQuestionPlan]:
    signals = _unique_skills(historical_weakness_signals or [])[:3]
    if not plans or not signals:
        return plans
    target_index = _best_reinforcement_plan_index(plans, signals)
    return [
        replace(
            plan,
            historicalWeaknessSignals=signals,
            reinforcementIntent="review-weakness",
            selectionReason=(
                f"{plan.selectionReason} Reinforces historical weak areas: "
                f"{', '.join(signals)}."
            ),
        )
        if index == target_index
        else plan
        for index, plan in enumerate(plans)
    ]


def _best_reinforcement_plan_index(
    plans: list[ProfessionalQuestionPlan],
    signals: list[str],
) -> int:
    scored = [
        (
            _reinforcement_overlap_score(plan, signals),
            -index,
            index,
        )
        for index, plan in enumerate(plans)
    ]
    return max(scored)[2]


def _reinforcement_overlap_score(
    plan: ProfessionalQuestionPlan,
    signals: list[str],
) -> int:
    plan_text = " ".join(
        [
            plan.primarySkill or "",
            *plan.relatedSkills,
            plan.targetAbility,
            *plan.resumeSignals,
            *plan.jobDescriptionSignals,
        ]
    ).lower()
    score = 0
    for signal in signals:
        normalized = signal.lower()
        has_overlap = normalized in plan_text or any(
            token in plan_text for token in normalized.split()
        )
        if normalized and has_overlap:
            score += 1
    return score
