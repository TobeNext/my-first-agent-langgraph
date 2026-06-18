import pytest
from pydantic import ValidationError

from app.schemas.answer_evaluation import (
    AnswerEvaluationTask,
    AnswerEvaluationTaskStatus,
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)

NOW = "2026-06-07T00:00:00.000Z"


def test_answer_evaluation_task_matches_ts_contract_defaults() -> None:
    task = AnswerEvaluationTask.model_validate(
        {
            "schemaVersion": 1,
            "taskId": "task-1",
            "interviewId": "interview-1",
            "threadId": "thread-1",
            "nodeId": "node-1",
            "roundId": "round-1",
            "roundType": "professional-skills",
            "attemptId": "attempt-1",
            "targetType": "main-question",
            "targetId": "node-1",
            "targetRole": "Backend Engineer",
            "responseLanguage": "zh",
            "question": "请说明 Spring 事务传播机制。",
            "mainQuestion": "请说明 Spring 事务传播机制。",
            "candidateAnswer": "我会说明 REQUIRED 和 REQUIRES_NEW 的区别。",
            "createdAt": NOW,
        }
    )

    assert task.schemaVersion == 1
    assert task.evaluationPoints == []
    assert task.nodeConversation == []
    assert task.model_dump(exclude_none=True)["roundType"] == "professional-skills"


def test_answer_evaluation_schemas_reject_invalid_literals() -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationTaskStatus.model_validate(
            {
                "schemaVersion": 1,
                "taskId": "task-1",
                "interviewId": "interview-1",
                "attemptId": "attempt-1",
                "status": "queued",
                "attempts": 0,
                "createdAt": NOW,
            }
        )

    with pytest.raises(ValidationError):
        LlmAnswerEvaluationResult.model_validate(
            {
                **_result_payload(),
                "classification": "unknown",
            }
        )


def test_result_and_manifest_accept_ts_payload_shape() -> None:
    result = LlmAnswerEvaluationResult.model_validate(_result_payload())
    manifest = InterviewEvaluationManifest.model_validate(
        {
            "schemaVersion": 1,
            "interviewId": "interview-1",
            "threadId": "thread-1",
            "expectedTaskIds": ["task-1"],
            "completedTaskIds": ["task-1"],
            "failedTaskIds": [],
            "sealed": True,
            "sealedAt": NOW,
            "updatedAt": NOW,
        }
    )

    assert result.score.weightedTotal == 7.65
    assert manifest.sealed is True


def _result_payload() -> dict:
    return {
        "schemaVersion": 1,
        "taskId": "task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "nodeId": "node-1",
        "roundId": "round-1",
        "roundType": "professional-skills",
        "attemptId": "attempt-1",
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 8,
            "depth": 7,
            "specificity": 7,
            "clarity": 8,
            "weightedTotal": 7.65,
        },
        "strengths": ["覆盖了事务传播机制"],
        "missingPoints": ["异常回滚边界还不够完整"],
        "incorrectPoints": [],
        "shouldAskFollowUp": True,
        "followUpFocus": ["异常回滚边界"],
        "evaluatorModel": "test-model",
        "promptVersion": "answer-evaluation-v1",
        "createdAt": NOW,
    }
