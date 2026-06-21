from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.question_query import build_professional_requeries


def test_build_professional_requeries_generates_three_interview_retrieval_intents() -> None:
    plan = ProfessionalQuestionPlan(
        kind="cross-skill-scenario",
        primarySkill=None,
        relatedSkills=["RAG", "Tool Calling", "Memory"],
        lens="cross-skill-integration",
        targetAbility="Agent architecture",
        questionType="scenario",
        coverageIntent="cross-skill-integration",
        resumeSignals=["RAG", "Tool Calling", "Memory"],
        jobDescriptionSignals=["设计 LLM Agent 核心架构", "工具调用和记忆管理"],
        questionDriver="resume-and-job-description",
        expectedDifficulty="hard",
        selectionReason="test",
    )

    intents = build_professional_requeries(
        selected_direction="AI Agent Engineer",
        plan=plan,
        professional_skills="- RAG\n- Tool Calling\n- Memory\n- TypeScript",
        project_experience="- 设计 RAG Agent，支持工具调用和会话记忆",
        normalized_skills=["RAG", "Tool Calling", "Memory", "TypeScript"],
    )

    assert [intent.type for intent in intents] == [
        "skill_exact",
        "job_scenario",
        "capability_probe",
    ]
    assert "Exact skill keywords:" in intents[0].query
    assert "Job responsibility signals:" in intents[1].query
    assert "Capability focus:" in intents[2].query
    assert all("Round type: professional-skills" in intent.query for intent in intents)


def test_build_professional_requeries_includes_historical_reinforcement_signals() -> None:
    plan = ProfessionalQuestionPlan(
        kind="skill-focus",
        primarySkill="RAG",
        relatedSkills=[],
        lens="implementation-depth",
        targetAbility="RAG",
        questionType="knowledge-check",
        coverageIntent="implementation-depth",
        resumeSignals=["RAG"],
        jobDescriptionSignals=["检索增强生成"],
        questionDriver="resume-and-job-description",
        expectedDifficulty="medium",
        selectionReason="test",
        historicalWeaknessSignals=["缺少失败降级", "缺少指标阈值"],
        reinforcementIntent="review-weakness",
    )

    intents = build_professional_requeries(
        selected_direction="AI Agent Engineer",
        plan=plan,
        professional_skills="- RAG",
        project_experience="- 设计 RAG Agent",
        normalized_skills=["RAG"],
    )

    assert "Historical weak areas to reinforce:" in intents[0].query
    assert "Reinforcement intent: review-weakness" in intents[2].query
    assert "缺少失败降级" in intents[2].query
