import pytest

from app.domain.interview_state_machine import (
    FLOW_TEST_SKIP_MARKER,
    AnswerEvaluationResult,
    apply_user_reply,
    build_final_interview_state_from_evaluations,
    build_interview_progress_summary,
    build_rule_evaluation,
    classify_by_rules,
    validate_interview_state,
)
from app.schemas.answer_evaluation import LlmAnswerEvaluationResult
from app.schemas.interview_state import AnswerAttemptState, AnswerScore, InterviewSessionState


def _score(value: float = 8.8) -> AnswerScore:
    return AnswerScore.model_validate(
        {
            "relevance": value,
            "accuracy": value,
            "depth": value,
            "specificity": value,
            "clarity": value,
            "weightedTotal": value,
        }
    )


def _state_fixture(*, flow_test: bool = True) -> InterviewSessionState:
    return InterviewSessionState.model_validate(
        {
            "version": 1,
            "threadId": "thread-1",
            "targetRole": "通用技术岗位",
            "company": None,
            "responseLanguage": "zh",
            "phase": "professional-skills-round",
            "activeRoundId": "round-professional",
            "finalReportReady": False,
            "finalReport": None,
            "setup": {
                "selectedDirection": "通用技术岗位",
                "directionSource": "derived",
                "settings": {
                    "reviewIncorrectOrMissingPoints": True,
                    "skipProfessionalSkillsRound": False,
                    "skipProjectExperienceRound": False,
                    "enableFlowTestMode": flow_test,
                    "professionalQuestionMode": "custom-count",
                    "professionalQuestionCount": 1,
                    "projectQuestionCount": 1,
                },
            },
            "resumeContext": {
                "professionalSkills": "TypeScript\nRAG",
                "projectExperience": "AI 面试 Agent 状态机改造",
                "jobDescription": "",
                "resumeParsed": True,
            },
            "lastCorrectionSummary": None,
            "rounds": [
                {
                    "id": "round-professional",
                    "type": "professional-skills",
                    "status": "in-progress",
                    "plannedNodeCount": 1,
                    "completedNodeCount": 0,
                    "activeNodeId": "node-rag",
                    "nodeOrder": ["node-rag"],
                    "nodes": [
                        {
                            "id": "node-rag",
                            "topic": "RAG",
                            "source": "knowledge-base",
                            "mainQuestion": "请解释你的 RAG 链路。",
                            "status": "awaiting-main-answer",
                            "currentTargetType": "main-question",
                            "currentFollowUpId": None,
                            "followUpCount": 0,
                            "maxFollowUps": 3,
                            "detourResponseCount": 0,
                            "earlyCompletionReason": None,
                            "followUps": [
                                {
                                    "id": "follow-up-1",
                                    "index": 1,
                                    "intent": "depth",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                },
                                {
                                    "id": "follow-up-2",
                                    "index": 2,
                                    "intent": "accuracy",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                },
                            ],
                            "answerAttempts": [],
                            "aggregatedScore": None,
                            "summary": None,
                        }
                    ],
                },
                {
                    "id": "round-project",
                    "type": "project-experience",
                    "status": "pending",
                    "plannedNodeCount": 1,
                    "completedNodeCount": 0,
                    "activeNodeId": "node-project",
                    "nodeOrder": ["node-project"],
                    "nodes": [
                        {
                            "id": "node-project",
                            "topic": "状态机改造",
                            "source": "resume",
                            "mainQuestion": "请介绍你的状态机项目。",
                            "status": "pending",
                            "currentTargetType": "main-question",
                            "currentFollowUpId": None,
                            "followUpCount": 0,
                            "maxFollowUps": 2,
                            "detourResponseCount": 0,
                            "earlyCompletionReason": None,
                            "followUps": [
                                {
                                    "id": "project-follow-up-1",
                                    "index": 1,
                                    "intent": "depth",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                }
                            ],
                            "answerAttempts": [],
                            "aggregatedScore": None,
                            "summary": None,
                        }
                    ],
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("结束面试", "stop-request"),
        ("skip this question", "skip-request"),
        ("能详细说明题意吗", "clarification-request"),
        ("how are you scoring this?", "meta-question"),
        ("我会从召回、排序和生成三个阶段说明", None),
    ],
)
def test_classify_by_rules_matches_ts_literal_contract(
    message: str,
    expected: str | None,
) -> None:
    assert classify_by_rules(message) == expected


def test_build_interview_progress_summary_matches_active_main_question() -> None:
    summary = build_interview_progress_summary(_state_fixture())

    assert summary.model_dump() == {
        "totalQuestionCount": 2,
        "completedQuestionCount": 0,
        "remainingQuestionCount": 2,
        "currentQuestionIndex": 1,
        "currentRoundType": "professional-skills",
        "currentRoundLabel": "【第一轮：专业技能面试】",
        "currentStage": "main-question",
        "currentFollowUpIndex": None,
        "currentQuestionText": "请解释你的 RAG 链路。",
        "currentNodeTopic": "RAG",
    }


def test_apply_user_reply_creates_follow_up_for_direct_answer() -> None:
    state = _state_fixture()
    result = apply_user_reply(
        state,
        "我会先做 query rewrite，再召回 topK，最后根据上下文生成答案。",
        AnswerEvaluationResult(
            classification="direct-answer",
            score=_score(8.9),
            strengths=["结构清晰"],
            missingPoints=[],
            incorrectPoints=[],
            recommendedIntent="depth",
            followUpFocus=["query rewrite"],
            shouldCompleteNode=True,
        ),
    )

    next_state = validate_interview_state(result.state.model_dump())
    active_node = next_state.rounds[0].nodes[0]

    assert active_node.currentTargetType == "follow-up"
    assert active_node.followUpCount == 1
    assert active_node.followUps[0].status == "asked"
    assert result.state.followUpMemory.askedQuestions == [active_node.followUps[0].question]
    assert result.state.followUpMemory.updatedAt is not None
    assert "query rewrite" in result.assistantReply


def test_apply_user_reply_skip_advances_to_project_round() -> None:
    result = apply_user_reply(
        _state_fixture(),
        "跳过",
        AnswerEvaluationResult(
            classification="skip-request",
            score=None,
            strengths=[],
            missingPoints=[],
            incorrectPoints=[],
            recommendedIntent="depth",
            followUpFocus=[],
            shouldCompleteNode=True,
        ),
    )

    assert result.state.phase == "project-experience-round"
    assert result.state.activeRoundId == "round-project"
    assert result.state.rounds[0].status == "completed"
    assert result.state.rounds[1].status == "in-progress"
    assert "项目" in result.assistantReply


def test_apply_user_reply_off_topic_keeps_current_question() -> None:
    result = apply_user_reply(
        _state_fixture(),
        "我想问一下流程是什么",
        AnswerEvaluationResult(
            classification="meta-question",
            score=None,
            strengths=[],
            missingPoints=[],
            incorrectPoints=[],
            recommendedIntent="depth",
            followUpFocus=[],
            shouldCompleteNode=False,
        ),
    )

    active_node = result.state.rounds[0].nodes[0]

    assert active_node.status == "awaiting-main-answer"
    assert active_node.detourResponseCount == 1
    assert "请继续回答" in result.assistantReply


def test_flow_test_skip_marker_builds_mock_reply_and_preserves_state_shape() -> None:
    state = _state_fixture(flow_test=True)
    stored_message, evaluation = build_rule_evaluation(FLOW_TEST_SKIP_MARKER, state=state)
    result = apply_user_reply(state, stored_message, evaluation)

    assert stored_message != FLOW_TEST_SKIP_MARKER
    assert "RAG" in stored_message
    assert result.state.rounds[0].nodes[0].answerAttempts[0].userMessage == stored_message
    validate_interview_state(result.state.model_dump())


def test_build_final_interview_state_from_evaluations_overrides_rule_scores() -> None:
    state = _state_fixture(flow_test=False)
    attempt = AnswerAttemptState.model_validate(
        {
            "id": "attempt-1",
            "targetType": "main-question",
            "targetId": "node-rag",
            "userMessage": "我会先做召回再重排。",
            "classification": "direct-answer",
            "score": _score(5).model_dump(),
            "strengths": ["本地规则优势"],
            "missingPoints": [],
            "incorrectPoints": [],
            "isDetour": False,
            "createdAt": "2026-06-15T00:00:00Z",
        }
    )
    completed_node = state.rounds[0].nodes[0].model_copy(
        update={
            "status": "completed",
            "answerAttempts": [attempt],
            "currentFollowUpId": None,
            "currentTargetType": "main-question",
        },
        deep=True,
    )
    completed_round = state.rounds[0].model_copy(
        update={"status": "completed", "completedNodeCount": 1, "nodes": [completed_node]},
        deep=True,
    )
    completed_state = state.model_copy(
        update={
            "phase": "completed",
            "activeRoundId": None,
            "finalReportReady": True,
            "rounds": [completed_round, state.rounds[1].model_copy(update={"status": "skipped"})],
        },
        deep=True,
    )
    evaluation = LlmAnswerEvaluationResult.model_validate(
        {
            "schemaVersion": 1,
            "taskId": "task-1",
            "interviewId": "thread-1",
            "threadId": "thread-1",
            "nodeId": "node-rag",
            "roundId": "round-professional",
            "roundType": "professional-skills",
            "attemptId": "attempt-1",
            "classification": "partial-answer",
            "score": _score(8.4).model_dump(),
            "strengths": ["LLM 覆盖了召回链路"],
            "missingPoints": ["还缺少失败降级"],
            "incorrectPoints": [],
            "shouldAskFollowUp": False,
            "followUpFocus": ["失败降级"],
            "evaluatorModel": "mock-model",
            "promptVersion": "answer-evaluation-v1",
            "createdAt": "2026-06-15T00:00:01Z",
        }
    )

    final_state = build_final_interview_state_from_evaluations(completed_state, [evaluation])
    final_attempt = final_state.rounds[0].nodes[0].answerAttempts[0]

    assert final_state.finalReportReady is True
    assert final_attempt.classification == "partial-answer"
    assert final_attempt.score and final_attempt.score.weightedTotal == 8.4
    assert final_state.rounds[0].nodes[0].summary
    assert final_state.rounds[0].nodes[0].summary.missingPoints == ["还缺少失败降级"]
    assert "综合得分" in (final_state.finalReport or "")
