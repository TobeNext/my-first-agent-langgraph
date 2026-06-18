import json

from app.domain.interview_initialization_pipeline import initialize_interview_from_kickoff
from app.domain.kickoff_recovery import (
    detect_kickoff_payload_format,
    extract_job_description_markdown_from_kickoff_message,
    extract_parsed_resume_from_kickoff_message,
    extract_structured_interview_start_request,
)
from app.domain.question_planner import plan_professional_question_queries


def _structured_start_payload() -> str:
    return json.dumps(
        {
            "requestKind": "interview-start",
            "protocolVersion": "2026-05-structured-start-v1",
            "startInterview": True,
            "threadId": "thread-structured",
            "resumeMarkdown": "# Resume",
            "jobDescriptionMarkdown": (
                "# 岗位要求\n- 熟悉 RAG 检索和流式接口\n- 具备 Redis 异步任务经验"
            ),
            "settings": {
                "reviewIncorrectOrMissingPoints": True,
                "skipProfessionalSkillsRound": False,
                "skipProjectExperienceRound": False,
                "enableFlowTestMode": True,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 2,
                "projectQuestionCount": 1,
            },
            "resumeSections": {
                "professionalSkills": "- TypeScript\n- RAG 检索\n- Redis 队列",
                "projectExperience": "- AI 面试 Agent 状态机改造\n- BFF 流式代理联调",
            },
        },
        ensure_ascii=False,
    )


def test_kickoff_recovery_parses_structured_start_and_resume_sections() -> None:
    raw = _structured_start_payload()
    parsed = extract_structured_interview_start_request(raw)
    resume = extract_parsed_resume_from_kickoff_message(raw)

    assert parsed is not None
    assert detect_kickoff_payload_format(raw) == "structured-start-v1"
    assert extract_job_description_markdown_from_kickoff_message(raw).startswith("# 岗位要求")
    assert resume.normalizedSkills == ["TypeScript", "RAG 检索", "Redis 队列"]
    assert resume.normalizedProjectTopics == ["AI 面试 Agent 状态机改造", "BFF 流式代理联调"]


def test_planner_creates_jd_gap_plan_when_custom_count_exceeds_resume_skills() -> None:
    plans = plan_professional_question_queries(
        mode="custom-count",
        professional_skills=["RAG"],
        desired_question_count=2,
        job_description="# 岗位要求\n- Redis 异步任务\n- RAG 检索",
        project_topics=[],
    )

    assert len(plans) == 2
    assert plans[0].kind == "skill-focus"
    assert plans[1].kind == "jd-gap-scenario"
    assert plans[1].questionDriver == "job-description"


def test_initialize_interview_from_structured_start_builds_real_session() -> None:
    initialized = initialize_interview_from_kickoff(
        thread_id="thread-structured",
        raw_kickoff_message=_structured_start_payload(),
    )
    state = initialized.state

    assert state.threadId == "thread-structured"
    assert state.resumeContext.jobDescription.startswith("# 岗位要求")
    assert state.setup.settings.professionalQuestionCount == 2
    assert state.setup.settings.projectQuestionCount == 1
    assert state.rounds[0].plannedNodeCount == 2
    assert state.rounds[1].plannedNodeCount == 1
    assert state.rounds[0].status == "in-progress"
    assert state.rounds[0].nodes[0].mainQuestion
    assert initialized.resources.generationTrace
    assert initialized.resources.judgeTrace
    assert "结构化模拟面试" in initialized.assistantReply
