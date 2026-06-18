from app.domain.question_critic import judge_initialization_question_set
from app.domain.question_planner import ProfessionalQuestionPlan
from app.schemas.interview_state import InterviewQuestionCandidate


def test_professional_scenario_shape_mismatch_uses_ts_baseline_fallback() -> None:
    plan = _plan(kind="skill-focus", primary_skill="RAG", question_type="scenario")
    result = judge_initialization_question_set(
        professional_question_plan=[plan],
        professional_questions=[
            _candidate("q1", "请解释 RAG。", "professional-skills", skill_area=["RAG"])
        ],
        project_questions=[],
        normalized_project_topics=[],
        target_role="AI Engineer",
    )

    assert result.professionalQuestions[0].text == (
        "请结合你真实做过的项目，详细说明你在RAG上的实现思路、关键取舍与排障经验。"
    )
    assert result.judgeTrace[0].verdict == "fallback"
    assert result.judgeTrace[0].failureReasons == ["scenario-shape-mismatch"]


def test_duplicate_question_uses_generic_professional_fallback() -> None:
    duplicate = "请结合真实项目说明 RAG 检索链路如何落地。"
    result = judge_initialization_question_set(
        professional_question_plan=[],
        professional_questions=[
            _candidate("q1", duplicate, "professional-skills", skill_area=["RAG"]),
            _candidate("q2", duplicate, "professional-skills", skill_area=["RAG"]),
        ],
        project_questions=[],
        normalized_project_topics=[],
        target_role="AI Engineer",
    )

    assert result.professionalQuestions[0].text == duplicate
    assert result.professionalQuestions[1].text == (
        "请结合你真实做过的项目，详细说明你最熟悉的一项专业能力是如何落地、排障和优化的。"
    )
    assert result.judgeTrace[1].failureReasons == ["duplicate-question"]


def test_project_shape_mismatch_uses_topic_fallback() -> None:
    result = judge_initialization_question_set(
        professional_question_plan=[],
        professional_questions=[],
        project_questions=[_candidate("p1", "请介绍架构设计。", "project-experience")],
        normalized_project_topics=["异步评分队列"],
        target_role="AI Engineer",
    )

    assert result.projectQuestions[0].text == (
        "请结合项目“异步评分队列”，说明项目背景、你的职责、关键决策、遇到的挑战以及最终结果。"
    )
    assert result.judgeTrace[0].failureReasons == ["project-shape-mismatch"]


def _candidate(
    question_id: str,
    text: str,
    role: str,
    *,
    skill_area: list[str] | None = None,
) -> InterviewQuestionCandidate:
    return InterviewQuestionCandidate.model_validate(
        {"id": question_id, "text": text, "role": role, "skillArea": skill_area}
    )


def _plan(
    *,
    kind: str,
    primary_skill: str | None,
    question_type: str,
) -> ProfessionalQuestionPlan:
    return ProfessionalQuestionPlan(
        kind=kind,
        primarySkill=primary_skill,
        relatedSkills=[primary_skill] if primary_skill else [],
        lens="implementation-depth",
        targetAbility=primary_skill or "专业能力",
        questionType=question_type,
        coverageIntent="implementation-depth",
        resumeSignals=[],
        jobDescriptionSignals=[],
        questionDriver="resume",
        expectedDifficulty="medium",
        selectionReason="fixture",
    )
