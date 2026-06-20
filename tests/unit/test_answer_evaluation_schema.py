import pytest
from pydantic import ValidationError

from app.schemas.answer_evaluation import (
    LlmAnswerEvaluationResult,
)

NOW = "2026-06-07T00:00:00.000Z"


def test_answer_evaluation_schemas_reject_invalid_literals() -> None:
    with pytest.raises(ValidationError):
        LlmAnswerEvaluationResult.model_validate(
            {
                **_result_payload(),
                "classification": "unknown",
            }
        )


def test_result_accepts_ts_payload_shape() -> None:
    result = LlmAnswerEvaluationResult.model_validate(_result_payload())

    assert result.score.weightedTotal == 7.65


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
