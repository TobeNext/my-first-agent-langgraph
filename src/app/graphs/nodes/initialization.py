from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from app.config import get_settings
from app.domain.interview_initialization_pipeline import (
    _fill_professional_questions,
    _fill_project_questions,
    _historical_reinforcement_signals,
    _resolve_selected_direction,
    _resolve_settings,
)
from app.domain.interview_memory_retriever import retrieve_user_interview_memory
from app.domain.kickoff_recovery import (
    extract_job_description_markdown_from_kickoff_message,
    extract_parsed_resume_from_kickoff_message,
    extract_structured_interview_start_request,
)
from app.domain.question_critic import judge_initialization_question_set
from app.domain.question_generator import generate_initialization_question_set
from app.domain.question_planner import plan_professional_question_queries
from app.domain.question_retriever import retrieve_initialization_questions
from app.domain.resume_jd_match import ResumeJdMatchAnalysis, build_resume_jd_match_analysis
from app.schemas.interview_state import HistoricalInterviewMemoryState, InterviewQuestionCandidate
from app.telemetry import interview_protocol_from_message


def prepare_initialization_input_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = str(state.get("raw_user_message") or "")
    structured = extract_structured_interview_start_request(raw_kickoff_message)
    initialization_input = {
        "threadId": str(state["thread_id"]),
        "resourceId": str(state["resource_id"]) if state.get("resource_id") else None,
        "rawKickoffMessage": raw_kickoff_message,
        "protocol": interview_protocol_from_message(raw_kickoff_message),
        "hasStructuredStart": structured is not None,
    }
    if structured:
        initialization_input["hasJobDescription"] = bool(structured.jobDescriptionMarkdown.strip())
        initialization_input["hasUserId"] = bool(structured.userId)
    return {"initialization_input": initialization_input}


def analyze_resume_jd_match_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    normalized_skills = parsed_resume.normalizedSkills or ["通用技术能力"]
    normalized_projects = parsed_resume.normalizedProjectTopics
    job_description = extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    analysis = build_resume_jd_match_analysis(
        professional_skills=parsed_resume.professionalSkillsSection,
        project_experience=parsed_resume.projectExperienceSection,
        job_description=job_description,
        normalized_skills=normalized_skills,
        normalized_project_topics=normalized_projects,
    )
    return {"resume_jd_match_analysis": analysis.model_dump(mode="json")}


def retrieve_historical_memory_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    structured = extract_structured_interview_start_request(raw_kickoff_message)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    job_description = extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    settings = _resolve_settings(
        raw_kickoff_message,
        parsed_resume.normalizedSkills or ["通用技术能力"],
    )
    if not settings.enableHistoricalMemory or _is_job_mismatch(state, job_description):
        memory = HistoricalInterviewMemoryState()
    else:
        memory = retrieve_user_interview_memory(
            user_id=(structured.userId if structured else None)
            or get_settings().interview_memory_user_id,
            target_role=_resolve_selected_direction(raw_kickoff_message),
            professional_skills=parsed_resume.professionalSkillsSection,
            job_description=job_description,
        )
    return {"historical_memory": memory.model_dump(mode="json")}


def plan_questions_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    normalized_skills = parsed_resume.normalizedSkills or ["通用技术能力"]
    normalized_projects = parsed_resume.normalizedProjectTopics
    job_description = extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    settings = _resolve_settings(raw_kickoff_message, normalized_skills)
    match_analysis = _match_analysis_from_state(state)
    memory = HistoricalInterviewMemoryState.model_validate(
        state.get("historical_memory") or HistoricalInterviewMemoryState().model_dump()
    )
    desired_professional_count = (
        0 if settings.skipProfessionalSkillsRound else settings.professionalQuestionCount
    )
    plan = (
        []
        if _is_job_mismatch(state, job_description)
        else plan_professional_question_queries(
            mode=settings.professionalQuestionMode,
            professional_skills=normalized_skills,
            desired_question_count=desired_professional_count,
            job_description=job_description,
            project_topics=normalized_projects,
            historical_weakness_signals=_historical_reinforcement_signals(memory),
            match_analysis=match_analysis,
        )
    )
    return {"professional_question_plan": [asdict(item) for item in plan]}


def retrieve_questions_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    normalized_skills = parsed_resume.normalizedSkills or ["通用技术能力"]
    normalized_projects = parsed_resume.normalizedProjectTopics
    job_description = extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    if _is_job_mismatch(state, job_description):
        return {
            "retrieved_professional_questions": [],
            "retrieved_project_questions": [],
            "recall_traces": [],
        }
    retrieval = retrieve_initialization_questions(
        selected_direction=_resolve_selected_direction(raw_kickoff_message),
        raw_kickoff_message=raw_kickoff_message,
        professional_skills=parsed_resume.professionalSkillsSection,
        normalized_professional_skills=normalized_skills,
        project_experience=parsed_resume.projectExperienceSection,
        normalized_project_topics=normalized_projects,
        job_description=job_description,
        professional_question_plan=_professional_plan_from_state(state),
        match_analysis=_match_analysis_from_state(state),
    )
    return {
        "retrieved_professional_questions": [
            item.model_dump(mode="json") for item in retrieval.professionalQuestions
        ],
        "retrieved_project_questions": [
            item.model_dump(mode="json") for item in retrieval.projectQuestions
        ],
        "recall_traces": [asdict(item) for item in retrieval.recallTraces],
    }


def generate_question_set_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    normalized_projects = parsed_resume.normalizedProjectTopics
    job_description = extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    settings = _resolve_settings(
        raw_kickoff_message,
        parsed_resume.normalizedSkills or ["通用技术能力"],
    )
    plan = _professional_plan_from_state(state)
    professional_candidates = _fill_professional_questions(
        retrieved=_questions_from_state(state, "retrieved_professional_questions"),
        plan=plan,
        target_role=_resolve_selected_direction(raw_kickoff_message),
        desired_count=0
        if settings.skipProfessionalSkillsRound
        else settings.professionalQuestionCount,
    )
    project_candidates = _fill_project_questions(
        retrieved=_questions_from_state(state, "retrieved_project_questions"),
        topics=normalized_projects,
        desired_count=0 if settings.skipProjectExperienceRound else settings.projectQuestionCount,
    )
    generated = generate_initialization_question_set(
        professional_question_plan=plan,
        professional_questions=professional_candidates,
        project_questions=project_candidates,
        job_description=job_description,
        normalized_project_topics=normalized_projects,
    )
    return {
        "generated_professional_questions": [
            item.model_dump(mode="json") for item in generated.professionalQuestions
        ],
        "generated_project_questions": [
            item.model_dump(mode="json") for item in generated.projectQuestions
        ],
        "generation_trace": [asdict(item) for item in generated.generationTrace],
    }


def judge_question_set_node(state: Mapping[str, Any]) -> dict[str, Any]:
    raw_kickoff_message = _raw_kickoff_message(state)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    judged = judge_initialization_question_set(
        professional_question_plan=_professional_plan_from_state(state),
        professional_questions=_questions_from_state(state, "generated_professional_questions"),
        project_questions=_questions_from_state(state, "generated_project_questions"),
        normalized_project_topics=parsed_resume.normalizedProjectTopics,
        target_role=_resolve_selected_direction(raw_kickoff_message),
    )
    return {
        "judged_professional_questions": [
            item.model_dump(mode="json") for item in judged.professionalQuestions
        ],
        "judged_project_questions": [
            item.model_dump(mode="json") for item in judged.projectQuestions
        ],
        "judge_trace": [asdict(item) for item in judged.judgeTrace],
    }


def _raw_kickoff_message(state: Mapping[str, Any]) -> str:
    initialization_input = state.get("initialization_input")
    if isinstance(initialization_input, Mapping):
        return str(initialization_input.get("rawKickoffMessage") or "")
    return str(state.get("raw_user_message") or "")


def _match_analysis_from_state(state: Mapping[str, Any]) -> ResumeJdMatchAnalysis:
    return ResumeJdMatchAnalysis.model_validate(state.get("resume_jd_match_analysis") or {})


def _is_job_mismatch(state: Mapping[str, Any], job_description: str) -> bool:
    analysis = _match_analysis_from_state(state)
    if not job_description.strip():
        return False
    return analysis.isJobMatched is False


def _professional_plan_from_state(state: Mapping[str, Any]) -> list[Any]:
    from app.domain.question_planner import ProfessionalQuestionPlan

    return [
        item
        if isinstance(item, ProfessionalQuestionPlan)
        else ProfessionalQuestionPlan(**item)
        for item in list(state.get("professional_question_plan") or [])
    ]


def _questions_from_state(
    state: Mapping[str, Any],
    field_name: str,
) -> list[InterviewQuestionCandidate]:
    return [
        item
        if isinstance(item, InterviewQuestionCandidate)
        else InterviewQuestionCandidate.model_validate(item)
        for item in list(state.get(field_name) or [])
    ]
