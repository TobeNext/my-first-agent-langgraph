from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings
from app.domain.interview_memory_retriever import retrieve_user_interview_memory
from app.domain.kickoff_recovery import (
    extract_job_description_markdown_from_kickoff_message,
    extract_parsed_resume_from_kickoff_message,
    extract_selected_direction_from_kickoff_message,
    extract_structured_interview_start_request,
)
from app.domain.question_critic import QuestionJudgeRecord, judge_initialization_question_set
from app.domain.question_generator import (
    GeneratedQuestionRecord,
    fallback_professional_question,
    fallback_project_question,
    generate_initialization_question_set,
)
from app.domain.question_planner import ProfessionalQuestionPlan, plan_professional_question_queries
from app.domain.question_retriever import RagRecallTrace, retrieve_initialization_questions
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.interview_state import (
    INTERVIEW_STATE_VERSION,
    PROFESSIONAL_MAX_FOLLOW_UPS,
    PROJECT_MAX_FOLLOW_UPS,
    HistoricalInterviewMemoryState,
    InterviewQuestionCandidate,
    InterviewRoundState,
    InterviewSessionState,
    InterviewSystemSettings,
    InterviewTopicNodeState,
    ResponseLanguage,
    RoundType,
)


@dataclass(frozen=True)
class InterviewInitializationResources:
    professionalSkills: str
    projectExperience: str
    normalizedProfessionalSkills: list[str]
    normalizedProjectTopics: list[str]
    jobDescription: str
    professionalQuestions: list[InterviewQuestionCandidate]
    projectQuestions: list[InterviewQuestionCandidate]
    generationTrace: list[GeneratedQuestionRecord]
    judgeTrace: list[QuestionJudgeRecord]
    recallTraces: list[RagRecallTrace]
    professionalQuestionPlan: list[ProfessionalQuestionPlan]
    historicalMemory: HistoricalInterviewMemoryState


@dataclass(frozen=True)
class InitializedInterview:
    state: InterviewSessionState
    assistantReply: str
    resources: InterviewInitializationResources


def initialize_interview_from_kickoff(
    *,
    thread_id: str,
    raw_kickoff_message: str,
    memory_repository: InterviewReportRepository | None = None,
) -> InitializedInterview:
    resources = resolve_interview_initialization_resources(
        raw_kickoff_message,
        memory_repository=memory_repository,
    )
    settings = _resolve_settings(raw_kickoff_message, resources.normalizedProfessionalSkills)
    selected_direction = _resolve_selected_direction(raw_kickoff_message)
    state = _build_session_state(
        thread_id=thread_id,
        selected_direction=selected_direction,
        response_language=_detect_response_language(
            "\n".join(
                [
                    raw_kickoff_message,
                    resources.professionalSkills,
                    resources.projectExperience,
                    resources.jobDescription,
                ]
            )
        ),
        settings=settings,
        resources=resources,
    )
    return InitializedInterview(
        state=state, assistantReply=_build_greeting(state), resources=resources
    )


def resolve_interview_initialization_resources(
    raw_kickoff_message: str,
    *,
    memory_repository: InterviewReportRepository | None = None,
) -> InterviewInitializationResources:
    structured = extract_structured_interview_start_request(raw_kickoff_message)
    parsed_resume = extract_parsed_resume_from_kickoff_message(raw_kickoff_message)
    normalized_skills = parsed_resume.normalizedSkills or ["通用技术能力"]
    normalized_projects = parsed_resume.normalizedProjectTopics
    job_description = (
        structured.jobDescriptionMarkdown
        if structured
        else extract_job_description_markdown_from_kickoff_message(raw_kickoff_message)
    )
    selected_direction = _resolve_selected_direction(raw_kickoff_message)
    settings = _resolve_settings(raw_kickoff_message, normalized_skills)
    desired_professional_count = (
        0 if settings.skipProfessionalSkillsRound else settings.professionalQuestionCount
    )
    historical_memory = (
        _retrieve_historical_memory(
            structured_user_id=structured.userId if structured else None,
            target_role=selected_direction,
            professional_skills=parsed_resume.professionalSkillsSection,
            job_description=job_description,
            repository=memory_repository,
        )
        if settings.enableHistoricalMemory
        else HistoricalInterviewMemoryState()
    )
    plan = plan_professional_question_queries(
        mode=settings.professionalQuestionMode,
        professional_skills=normalized_skills,
        desired_question_count=desired_professional_count,
        job_description=job_description,
        project_topics=normalized_projects,
        historical_weakness_signals=_historical_reinforcement_signals(historical_memory),
    )
    retrieval = retrieve_initialization_questions(
        selected_direction=selected_direction,
        raw_kickoff_message=raw_kickoff_message,
        professional_skills=parsed_resume.professionalSkillsSection,
        normalized_professional_skills=normalized_skills,
        project_experience=parsed_resume.projectExperienceSection,
        normalized_project_topics=normalized_projects,
        job_description=job_description,
        professional_question_plan=plan,
    )
    professional_candidates = _fill_professional_questions(
        retrieved=retrieval.professionalQuestions,
        plan=plan,
        target_role=selected_direction,
        desired_count=desired_professional_count,
    )
    project_candidates = _fill_project_questions(
        retrieved=retrieval.projectQuestions,
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
    judged = judge_initialization_question_set(
        professional_question_plan=plan,
        professional_questions=generated.professionalQuestions,
        project_questions=generated.projectQuestions,
        normalized_project_topics=normalized_projects,
        target_role=selected_direction,
    )
    return InterviewInitializationResources(
        professionalSkills=parsed_resume.professionalSkillsSection,
        projectExperience=parsed_resume.projectExperienceSection,
        normalizedProfessionalSkills=normalized_skills,
        normalizedProjectTopics=normalized_projects,
        jobDescription=job_description,
        professionalQuestions=judged.professionalQuestions,
        projectQuestions=judged.projectQuestions,
        generationTrace=generated.generationTrace,
        judgeTrace=judged.judgeTrace,
        recallTraces=retrieval.recallTraces,
        professionalQuestionPlan=plan,
        historicalMemory=historical_memory,
    )


def _build_session_state(
    *,
    thread_id: str,
    selected_direction: str,
    response_language: ResponseLanguage,
    settings: InterviewSystemSettings,
    resources: InterviewInitializationResources,
) -> InterviewSessionState:
    professional_round = _create_round(
        "professional-skills",
        _nodes_from_questions(
            questions=resources.professionalQuestions,
            max_follow_ups=PROFESSIONAL_MAX_FOLLOW_UPS,
        ),
    )
    project_round = _create_round(
        "project-experience",
        _nodes_from_questions(
            questions=resources.projectQuestions,
            max_follow_ups=PROJECT_MAX_FOLLOW_UPS,
        ),
    )
    rounds = [
        _skip_round(professional_round)
        if settings.skipProfessionalSkillsRound
        else professional_round,
        _skip_round(project_round) if settings.skipProjectExperienceRound else project_round,
    ]
    first_round = next((item for item in rounds if item.status != "skipped"), None)
    started_round = _start_round(first_round) if first_round else None
    rounds = [
        started_round if started_round and item.id == started_round.id else item for item in rounds
    ]
    phase = (
        "professional-skills-round"
        if started_round and started_round.type == "professional-skills"
        else "project-experience-round"
        if started_round and started_round.type == "project-experience"
        else "wrap-up"
    )
    return InterviewSessionState.model_validate(
        {
            "version": INTERVIEW_STATE_VERSION,
            "threadId": thread_id,
            "targetRole": selected_direction,
            "company": None,
            "responseLanguage": response_language,
            "phase": phase,
            "activeRoundId": started_round.id if started_round else None,
            "finalReportReady": False,
            "finalReport": None,
            "setup": {
                "selectedDirection": selected_direction,
                "directionSource": "derived",
                "settings": settings.model_dump(),
            },
            "resumeContext": {
                "professionalSkills": resources.professionalSkills,
                "projectExperience": resources.projectExperience,
                "jobDescription": resources.jobDescription,
                "resumeParsed": bool(resources.professionalSkills or resources.projectExperience),
            },
            "followUpMemory": {
                "askedQuestions": [],
                "resumeDigest": _compact_memory_digest(
                    "\n".join([resources.professionalSkills, resources.projectExperience])
                ),
                "jobDescriptionDigest": _compact_memory_digest(resources.jobDescription),
                "updatedAt": None,
            },
            "historicalMemory": resources.historicalMemory.model_dump(mode="json"),
            "lastCorrectionSummary": None,
            "rounds": [item.model_dump() for item in rounds],
        }
    )


def _retrieve_historical_memory(
    *,
    structured_user_id: str | None,
    target_role: str,
    professional_skills: str,
    job_description: str,
    repository: InterviewReportRepository | None,
) -> HistoricalInterviewMemoryState:
    user_id = structured_user_id or get_settings().interview_memory_user_id
    return retrieve_user_interview_memory(
        user_id=user_id,
        target_role=target_role,
        professional_skills=professional_skills,
        job_description=job_description,
        repository=repository,
    )


def _historical_reinforcement_signals(
    memory: HistoricalInterviewMemoryState,
) -> list[str]:
    if not memory.hasMemory:
        return []
    return _unique_signal_values(
        [
            *memory.weaknesses,
            *memory.missingPoints,
            *memory.reinforcementQuestionHints,
        ]
    )[:3]


def _nodes_from_questions(
    *,
    questions: list[InterviewQuestionCandidate],
    max_follow_ups: int,
) -> list[InterviewTopicNodeState]:
    nodes: list[InterviewTopicNodeState] = []
    for index, question in enumerate(questions, start=1):
        nodes.append(
            InterviewTopicNodeState.model_validate(
                {
                    "id": f"topic-node-{index}-{abs(hash(question.id)) % 10_000}",
                    "topic": _infer_topic(question),
                    "source": "knowledge-base",
                    "mainQuestion": question.text,
                    "referenceAnswer": question.answer,
                    "evaluationPoints": _extract_evaluation_points(question.answer),
                    "status": "pending",
                    "currentTargetType": "main-question",
                    "currentFollowUpId": None,
                    "followUpCount": 0,
                    "maxFollowUps": max_follow_ups,
                    "detourResponseCount": 0,
                    "earlyCompletionReason": None,
                    "followUps": [
                        _follow_up(index, follow_index)
                        for follow_index in range(1, max_follow_ups + 1)
                    ],
                    "answerAttempts": [],
                    "aggregatedScore": None,
                    "summary": None,
                }
            )
        )
    return nodes


def _follow_up(node_index: int, follow_index: int) -> dict:
    intents = ["depth", "accuracy", "experience", "breadth"]
    return {
        "id": f"follow-up-{node_index}-{follow_index}",
        "index": follow_index,
        "intent": intents[follow_index - 1] if follow_index <= len(intents) else "depth",
        "question": "",
        "status": "pending",
        "linkedAnswerId": None,
    }


def _compact_memory_digest(value: str, *, limit: int = 1200) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit].rstrip()


def _create_round(
    round_type: RoundType, nodes: list[InterviewTopicNodeState]
) -> InterviewRoundState:
    return InterviewRoundState.model_validate(
        {
            "id": f"{round_type}-round",
            "type": round_type,
            "status": "pending",
            "plannedNodeCount": len(nodes),
            "completedNodeCount": 0,
            "activeNodeId": nodes[0].id if nodes else None,
            "nodeOrder": [node.id for node in nodes],
            "nodes": [node.model_dump() for node in nodes],
        }
    )


def _start_round(round_item: InterviewRoundState | None) -> InterviewRoundState | None:
    if not round_item or not round_item.nodes:
        return round_item
    active_node = next(
        (node for node in round_item.nodes if node.id == round_item.activeNodeId),
        round_item.nodes[0],
    )
    started_node = active_node.model_copy(update={"status": "awaiting-main-answer"}, deep=True)
    return round_item.model_copy(
        update={
            "status": "in-progress",
            "activeNodeId": started_node.id,
            "nodes": [
                started_node if node.id == started_node.id else node for node in round_item.nodes
            ],
        },
        deep=True,
    )


def _skip_round(round_item: InterviewRoundState) -> InterviewRoundState:
    return round_item.model_copy(update={"status": "skipped", "activeNodeId": None}, deep=True)


def _build_greeting(state: InterviewSessionState) -> str:
    active_round = next((item for item in state.rounds if item.id == state.activeRoundId), None)
    active_node = (
        next((node for node in active_round.nodes if node.id == active_round.activeNodeId), None)
        if active_round
        else None
    )
    total = sum(item.plannedNodeCount for item in state.rounds)
    intro = (
        f"我们将围绕 {state.targetRole} 岗位进行一场结构化模拟面试。"
        f"本次共安排 {total} 道主问题，并在每道题后根据回答继续追问。"
        if state.responseLanguage == "zh"
        else (
            f"We will run a structured mock interview for the {state.targetRole} role. "
            f"This session includes {total} main questions, "
            "with follow-up questions based on your answers."
        )
    )
    return "\n\n".join(
        [item for item in [intro, active_node.mainQuestion if active_node else ""] if item]
    )


def _fill_professional_questions(
    *,
    retrieved: list[InterviewQuestionCandidate],
    plan: list[ProfessionalQuestionPlan],
    target_role: str,
    desired_count: int,
) -> list[InterviewQuestionCandidate]:
    questions = _unique_questions(retrieved)[:desired_count]
    for plan_item in plan:
        if len(questions) >= desired_count:
            break
        questions.append(fallback_professional_question(plan_item, target_role))
    while len(questions) < desired_count:
        questions.append(
            fallback_professional_question(plan[0] if plan else _default_plan(), target_role)
        )
    return questions


def _fill_project_questions(
    *,
    retrieved: list[InterviewQuestionCandidate],
    topics: list[str],
    desired_count: int,
) -> list[InterviewQuestionCandidate]:
    questions = _unique_questions(retrieved)[:desired_count]
    while len(questions) < desired_count:
        topic = topics[len(questions) % len(topics)] if topics else None
        questions.append(fallback_project_question(topic))
    return questions


def _resolve_settings(raw: str, normalized_skills: list[str]) -> InterviewSystemSettings:
    structured = extract_structured_interview_start_request(raw)
    if structured:
        return structured.settings
    skip_professional = _parse_bool(raw, "Skip professional-skills round", False)
    skip_project = _parse_bool(raw, "Skip project-experience round", True)
    mode = _parse_mode(raw)
    return InterviewSystemSettings.model_validate(
        {
            "reviewIncorrectOrMissingPoints": _parse_bool(
                raw,
                "Review incorrect or missing points after each completed question",
                True,
            ),
            "skipProfessionalSkillsRound": skip_professional,
            "skipProjectExperienceRound": skip_project,
            "enableFlowTestMode": _parse_bool(raw, "Flow test mode", False),
            "enableHistoricalMemory": _parse_bool(raw, "Historical memory", True),
            "professionalQuestionMode": mode,
            "professionalQuestionCount": 0
            if skip_professional
            else _parse_int(raw, "Professional question count", max(1, len(normalized_skills))),
            "projectQuestionCount": 0
            if skip_project
            else _parse_int(raw, "Project question count", 1),
        }
    )


def _resolve_selected_direction(raw: str) -> str:
    structured = extract_structured_interview_start_request(raw)
    if structured:
        combined = "\n".join([structured.resumeMarkdown, structured.jobDescriptionMarkdown])
        return (
            "通用技术岗位" if re.search(r"[\u3400-\u9fff]", combined) else "General Technical Role"
        )
    return extract_selected_direction_from_kickoff_message(raw)


def _detect_response_language(value: str) -> ResponseLanguage:
    return "zh" if re.search(r"[\u3400-\u9fff]", value) else "en"


def _parse_bool(raw: str, label: str, default: bool) -> bool:
    match = re.search(rf"{re.escape(label)}:\s*(enabled|disabled|yes|no)", raw, re.I)
    return (match.group(1).lower() in {"enabled", "yes"}) if match else default


def _parse_int(raw: str, label: str, default: int) -> int:
    match = re.search(rf"{re.escape(label)}:\s*(\d+)", raw, re.I)
    return int(match.group(1)) if match else default


def _parse_mode(raw: str) -> str:
    match = re.search(r"Professional question mode:\s*(per-skill-default|custom-count)", raw, re.I)
    return match.group(1).lower() if match else "per-skill-default"


def _unique_questions(
    questions: list[InterviewQuestionCandidate],
) -> list[InterviewQuestionCandidate]:
    result: list[InterviewQuestionCandidate] = []
    seen: set[str] = set()
    for question in questions:
        key = " ".join(question.text.lower().split())
        if key and key not in seen:
            seen.add(key)
            result.append(question)
    return result


def _unique_signal_values(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _infer_topic(question: InterviewQuestionCandidate) -> str:
    if question.skillArea:
        return " / ".join(question.skillArea[:2])
    cleaned = re.sub(r"[?？]\s*$", "", question.text)
    cleaned = re.sub(r"^请你?", "", cleaned).strip()
    first_clause = re.split(r"[。.!?？；;，,]", cleaned)[0].strip()
    return first_clause[:32] if first_clause else "面试问题"


def _extract_evaluation_points(answer: str | None) -> list[str] | None:
    if not answer:
        return None
    points = [
        re.sub(r"^(?:[-*+•]\s+|\d+[.)]\s+)", "", line).strip()
        for line in answer.splitlines()
        if line.strip()
    ]
    return points[:6] or None


def _default_plan() -> ProfessionalQuestionPlan:
    return ProfessionalQuestionPlan(
        kind="skill-focus",
        primarySkill="通用技术能力",
        relatedSkills=[],
        lens="implementation-depth",
        targetAbility="通用技术能力",
        questionType="knowledge-check",
        coverageIntent="implementation-depth",
        resumeSignals=["通用技术能力"],
        jobDescriptionSignals=[],
        questionDriver="resume",
        expectedDifficulty="medium",
        selectionReason="Default professional fallback plan.",
    )
