from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.job_description_signals import extract_job_description_signal_set
from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.resume_parser import extract_normalized_resume_topics


@dataclass(frozen=True)
class RetrievalQueryIntent:
    type: str
    query: str


def describe_professional_plan_skill(plan: ProfessionalQuestionPlan) -> str:
    if plan.kind == "skill-focus":
        return plan.primarySkill or "professional-skill"
    if plan.kind == "cross-skill-scenario":
        return f"cross-skill:{' + '.join(plan.relatedSkills)}"
    if plan.kind == "jd-gap-scenario":
        return f"jd-gap:{plan.targetAbility}"
    return (
        f"broad-professional:{' + '.join(plan.relatedSkills)}"
        if plan.relatedSkills
        else "broad-professional-context"
    )


def build_professional_skill_query(
    *,
    selected_direction: str,
    plan: ProfessionalQuestionPlan,
    professional_skills: str,
    project_experience: str,
    normalized_skills: list[str] | None = None,
) -> str:
    plan_skills = [skill for skill in [plan.primarySkill, *plan.relatedSkills] if skill]
    excluded = {_normalize(skill) for skill in plan_skills}
    related = [
        skill
        for skill in (normalized_skills or extract_normalized_resume_topics(professional_skills))
        if _normalize(skill) not in excluded
    ][:4]
    project_highlights = _relevant_project_highlights(project_experience, plan_skills)[:2]
    parts = [
        f"Target role: {selected_direction}",
        "Round type: professional-skills",
        f"Question lens: {plan.lens}",
        f"Question driver: {plan.questionDriver}",
        f"Target ability: {plan.targetAbility}",
    ]
    if plan.primarySkill:
        parts.append(f"Primary skill: {plan.primarySkill}")
    if plan.relatedSkills:
        parts.append(f"Scenario skills: {', '.join(plan.relatedSkills)}")
    if related:
        parts.append(f"Related resume skills: {', '.join(related)}")
    if project_highlights:
        parts.append("Relevant project highlights:")
        parts.extend(f"- {line}" for line in project_highlights)
    if plan.jobDescriptionSignals:
        parts.append("Job description signals:")
        parts.extend(f"- {signal}" for signal in plan.jobDescriptionSignals)
    if plan.historicalWeaknessSignals:
        parts.append("Historical weak areas to reinforce:")
        parts.extend(f"- {signal}" for signal in plan.historicalWeaknessSignals)
    return "\n".join(parts)


def build_professional_requeries(
    *,
    selected_direction: str,
    plan: ProfessionalQuestionPlan,
    professional_skills: str,
    project_experience: str,
    normalized_skills: list[str] | None = None,
) -> list[RetrievalQueryIntent]:
    base_query = build_professional_skill_query(
        selected_direction=selected_direction,
        plan=plan,
        professional_skills=professional_skills,
        project_experience=project_experience,
        normalized_skills=normalized_skills,
    )
    skill_terms = _compact_terms([*(plan.resumeSignals or []), *(plan.relatedSkills or [])])
    if plan.primarySkill:
        skill_terms = _compact_terms([plan.primarySkill, *skill_terms])
    job_terms = _compact_terms(plan.jobDescriptionSignals)
    related = _compact_terms(
        normalized_skills or extract_normalized_resume_topics(professional_skills)
    )
    project_highlights = _relevant_project_highlights(
        project_experience, [*skill_terms, *job_terms]
    )[:2]

    exact_terms = _compact_terms([*skill_terms, *related])[:10]
    scenario_terms = _compact_terms([plan.targetAbility, *job_terms, *skill_terms])[:10]
    capability_terms = _compact_terms([
        plan.coverageIntent,
        plan.lens,
        plan.targetAbility,
        *skill_terms,
        *job_terms,
        *plan.historicalWeaknessSignals,
    ])[:10]

    job_signal_text = ", ".join(job_terms) if job_terms else selected_direction
    project_evidence = (
        ["Project evidence:", *[f"- {line}" for line in project_highlights]]
        if project_highlights
        else []
    )
    historical_reinforcement = (
        [
            f"Reinforcement intent: {plan.reinforcementIntent}",
            f"Historical weak areas: {', '.join(plan.historicalWeaknessSignals)}",
            "Prefer questions that verify whether the candidate has improved on these areas.",
        ]
        if plan.historicalWeaknessSignals
        else []
    )

    return [
        RetrievalQueryIntent(
            type="skill_exact",
            query="\n".join(
                [
                    base_query,
                    "Retrieval intent: skill_exact",
                    f"Exact skill keywords: {', '.join(exact_terms)}",
                    "Prefer concrete interview questions that test the named technologies.",
                ]
            ),
        ),
        RetrievalQueryIntent(
            type="job_scenario",
            query="\n".join(
                [
                    f"Target role: {selected_direction}",
                    "Round type: professional-skills",
                    "Retrieval intent: job_scenario",
                    f"Scenario ability: {plan.targetAbility}",
                    f"Job responsibility signals: {job_signal_text}",
                    f"Scenario skills: {', '.join(scenario_terms)}",
                    *project_evidence,
                    "Prefer system design, troubleshooting, and engineering trade-off "
                    "main questions.",
                ]
            ),
        ),
        RetrievalQueryIntent(
            type="capability_probe",
            query="\n".join(
                [
                    f"Target role: {selected_direction}",
                    "Round type: professional-skills",
                    "Retrieval intent: capability_probe",
                    f"Question driver: {plan.questionDriver}",
                    f"Capability focus: {', '.join(capability_terms)}",
                    *historical_reinforcement,
                    "Prefer questions that validate project depth, implementation choices, "
                    "evaluation, and continuous improvement.",
                ]
            ),
        ),
    ]


def build_project_experience_query(
    *,
    selected_direction: str,
    project_experience: str,
    raw_kickoff_message: str,
    job_description: str = "",
    normalized_project_topics: list[str] | None = None,
) -> str:
    fallback = project_experience.strip() or raw_kickoff_message
    signal_set = extract_job_description_signal_set(
        job_description=job_description,
        project_topics=normalized_project_topics or [],
    )
    evidence_signals = signal_set.alignedSignals or signal_set.topSignals
    parts = [
        f"Target role: {selected_direction}",
        "Round type: project-experience",
        "Project experience context:",
        fallback,
    ]
    if evidence_signals:
        parts.append("Cross-check these JD requirements against the project evidence:")
        parts.extend(f"- {signal}" for signal in evidence_signals[:3])
    if signal_set.gapSignals:
        parts.append("Capability gaps to validate when resume evidence is thin:")
        parts.extend(f"- {signal}" for signal in signal_set.gapSignals[:2])
    return "\n".join(parts)


def _relevant_project_highlights(project_experience: str, skills: list[str]) -> list[str]:
    keywords = {keyword for skill in skills for keyword in _keywords(skill)}
    result: list[str] = []
    for line in _context_lines(project_experience):
        normalized = _normalize(line)
        if any(keyword in normalized for keyword in keywords):
            result.append(line)
    return result


def _context_lines(value: str) -> list[str]:
    return [
        re.sub(r"^(?:[-*+•]\s+|\d+[.)]\s+)", "", line).strip()
        for line in value.splitlines()
        if line.strip()
    ]


def _keywords(value: str) -> list[str]:
    normalized = _normalize(value)
    tokens = [
        token
        for token in re.split(r"[^a-z0-9\u3400-\u9fff+#.-]+", normalized)
        if len(token) >= 4 or re.search(r"[\u3400-\u9fff]", token)
    ]
    return [normalized, *tokens]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _compact_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
