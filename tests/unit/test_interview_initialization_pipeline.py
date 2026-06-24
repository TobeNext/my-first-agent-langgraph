import json

from app.domain.interview_initialization_pipeline import initialize_interview_from_kickoff
from app.domain.kickoff_recovery import (
    detect_kickoff_payload_format,
    extract_job_description_markdown_from_kickoff_message,
    extract_parsed_resume_from_kickoff_message,
    extract_structured_interview_start_request,
)
from app.domain.question_planner import plan_professional_question_queries
from app.domain.resume_jd_match import ResumeJdMatchAnalysis
from app.graphs.interview_graph import should_start_background_report_generation
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.interview_report import InterviewUserMemoryWrite


def _structured_start_payload(
    *,
    user_id: str | None = None,
    enable_historical_memory: bool = True,
) -> str:
    payload = {
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
                "enableHistoricalMemory": enable_historical_memory,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 2,
                "projectQuestionCount": 1,
            },
            "resumeSections": {
                "professionalSkills": "- TypeScript\n- RAG 检索\n- Redis 队列",
                "projectExperience": "- AI 面试 Agent 状态机改造\n- BFF 流式代理联调",
            },
        }
    if user_id:
        payload["userId"] = user_id
    return json.dumps(payload, ensure_ascii=False)


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


def test_planner_marks_existing_plan_for_historical_weakness_reinforcement() -> None:
    plans = plan_professional_question_queries(
        mode="per-skill-default",
        professional_skills=["RAG", "Redis"],
        desired_question_count=2,
        job_description="# 岗位要求\n- RAG 检索",
        project_topics=[],
        historical_weakness_signals=["缺少 RAG 失败降级", "缺少指标阈值"],
    )

    reinforced = [plan for plan in plans if plan.reinforcementIntent == "review-weakness"]

    assert len(plans) == 2
    assert len(reinforced) == 1
    assert reinforced[0].primarySkill == "RAG"
    assert reinforced[0].historicalWeaknessSignals == ["缺少 RAG 失败降级", "缺少指标阈值"]


def test_planner_prioritizes_llm_resume_jd_match_sections() -> None:
    analysis = ResumeJdMatchAnalysis.model_validate(
        {
            "resumeJdMatch": [
                {
                    "resumeSignal": "RAG 检索",
                    "jobSignal": "Agent 检索增强",
                    "matchType": "skill",
                    "relevance": 0.9,
                    "priority": "high",
                    "evidence": {
                        "resumeSignals": ["RAG 检索"],
                        "jobSignals": ["Agent 检索增强"],
                        "projectSignals": [],
                    },
                    "interviewFocus": ["RAG 检索"],
                    "suggestedQuestionTypes": ["experience_probe"],
                }
            ],
            "resumeOnly": [
                {"resumeSignal": "Vue", "category": "skill", "evidence": ["Vue"]}
            ],
            "jdOnly": [
                {
                    "jobSignal": "模型评估",
                    "category": "requirement",
                    "priority": "medium",
                    "evidence": ["模型评估"],
                }
            ],
        }
    )

    plans = plan_professional_question_queries(
        mode="custom-count",
        professional_skills=["Vue", "RAG 检索"],
        desired_question_count=3,
        job_description="# 岗位要求\n- Agent 检索增强\n- 模型评估",
        project_topics=[],
        match_analysis=analysis,
    )

    assert [plan.kind for plan in plans] == [
        "skill-focus",
        "skill-focus",
        "jd-gap-scenario",
    ]
    assert plans[0].primarySkill == "RAG 检索"
    assert plans[0].jobDescriptionSignals == ["Agent 检索增强"]
    assert plans[1].primarySkill == "Vue"
    assert plans[1].jobDescriptionSignals == []
    assert plans[2].targetAbility == "模型评估"


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
    assert "TypeScript" in state.followUpMemory.resumeDigest
    assert "AI 面试 Agent 状态机改造" in state.followUpMemory.resumeDigest
    assert state.followUpMemory.jobDescriptionDigest.startswith("# 岗位要求")
    assert state.followUpMemory.askedQuestions == []
    assert initialized.resources.generationTrace
    assert initialized.resources.judgeTrace
    assert "结构化模拟面试" in initialized.assistantReply


def test_initialize_interview_ends_without_report_when_resume_jd_match_is_empty() -> None:
    payload = {
        "requestKind": "interview-start",
        "protocolVersion": "2026-05-structured-start-v1",
        "startInterview": True,
        "threadId": "thread-mismatch",
        "resumeMarkdown": "# Resume",
        "jobDescriptionMarkdown": "# 岗位要求\n- Java 后端开发\n- Spring Cloud 微服务治理",
        "settings": {
            "reviewIncorrectOrMissingPoints": True,
            "skipProfessionalSkillsRound": False,
            "skipProjectExperienceRound": False,
            "enableFlowTestMode": True,
            "enableHistoricalMemory": True,
            "professionalQuestionMode": "custom-count",
            "professionalQuestionCount": 2,
            "projectQuestionCount": 1,
        },
        "resumeSections": {
            "professionalSkills": "- Vue 组件开发\n- CSS 动效",
            "projectExperience": "- 营销页面搭建",
        },
    }

    initialized = initialize_interview_from_kickoff(
        thread_id="thread-mismatch",
        raw_kickoff_message=json.dumps(payload, ensure_ascii=False),
    )

    assert initialized.state.phase == "completed"
    assert initialized.state.activeRoundId is None
    assert initialized.state.finalReportReady is False
    assert "岗位不匹配" in initialized.assistantReply
    assert initialized.resources.recallTraces == []
    assert initialized.resources.generationTrace == []
    assert initialized.resources.judgeTrace == []
    assert should_start_background_report_generation(
        {"session": initialized.state.model_dump()}
    ) is False


def test_initialize_interview_loads_historical_memory_for_structured_user_id(tmp_path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(
        InterviewUserMemoryWrite(
            id="memory-1",
            user_id="user-a",
            source_interview_id="interview-previous",
            source_thread_id="thread-previous",
            target_role="通用技术岗位",
            overall_score=6.5,
            weakness_summary_json='["RAG 失败降级覆盖不足"]',
            missing_points_json='["缺少失败降级"]',
            improvement_advice_json='["补充监控阈值"]',
            reinforcement_question_hints_json='["追问失败时如何降级"]',
            report_markdown_excerpt="# 报告",
            embedding_text="RAG 检索 失败降级 监控阈值",
            embedding_json=None,
            source_report_completed_at="2026-06-19T00:00:00Z",
            summary_generated_at="2026-06-19T00:00:00Z",
            created_at="2026-06-19T00:00:00Z",
            updated_at="2026-06-19T00:00:00Z",
        )
    )

    initialized = initialize_interview_from_kickoff(
        thread_id="thread-structured",
        raw_kickoff_message=_structured_start_payload(user_id="user-a"),
        memory_repository=repository,
    )

    assert initialized.state.historicalMemory.hasMemory is True
    assert initialized.state.historicalMemory.sourceInterviewIds == ["interview-previous"]
    assert initialized.state.historicalMemory.missingPoints == ["缺少失败降级"]
    reinforced = [
        plan
        for plan in initialized.resources.professionalQuestionPlan
        if plan.reinforcementIntent == "review-weakness"
    ]

    assert len(reinforced) == 1
    assert reinforced[0].primarySkill == "RAG 检索"
    assert "缺少失败降级" in (
        reinforced[0].historicalWeaknessSignals
    )


def test_initialize_interview_skips_historical_memory_when_disabled(tmp_path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(
        InterviewUserMemoryWrite(
            id="memory-disabled",
            user_id="user-a",
            source_interview_id="interview-previous",
            source_thread_id="thread-previous",
            target_role="通用技术岗位",
            overall_score=6.5,
            weakness_summary_json='["RAG 失败降级覆盖不足"]',
            missing_points_json='["缺少失败降级"]',
            improvement_advice_json='["补充监控阈值"]',
            reinforcement_question_hints_json='["追问失败时如何降级"]',
            report_markdown_excerpt="# 报告",
            embedding_text="RAG 检索 失败降级 监控阈值",
            embedding_json=None,
            source_report_completed_at="2026-06-19T00:00:00Z",
            summary_generated_at="2026-06-19T00:00:00Z",
            created_at="2026-06-19T00:00:00Z",
            updated_at="2026-06-19T00:00:00Z",
        )
    )

    initialized = initialize_interview_from_kickoff(
        thread_id="thread-structured",
        raw_kickoff_message=_structured_start_payload(
            user_id="user-a",
            enable_historical_memory=False,
        ),
        memory_repository=repository,
    )

    assert initialized.state.historicalMemory.hasMemory is False
    assert all(
        plan.reinforcementIntent == "none"
        for plan in initialized.resources.professionalQuestionPlan
    )
