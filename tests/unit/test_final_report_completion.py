from app.graphs.nodes.process_user_reply import complete_final_report_with_async_evaluations
from app.schemas.answer_evaluation import (
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)
from app.schemas.interview_state import AnswerAttemptState, InterviewSessionState
from tests.unit.test_interview_state_machine import _score, _state_fixture


class FakeStore:
    def __init__(
        self,
        *,
        manifest: InterviewEvaluationManifest | None,
        results: list[LlmAnswerEvaluationResult] | None = None,
    ) -> None:
        self.manifest = manifest
        self.results = results or []
        self.sealed = False

    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None:
        return self.manifest

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]:
        return self.results

    async def seal_interview(self, interview_id: str) -> None:
        self.sealed = True
        if self.manifest:
            self.manifest = self.manifest.model_copy(
                update={"sealed": True, "sealedAt": "2026-06-15T00:00:02Z"},
                deep=True,
            )


def completed_state_with_attempt() -> InterviewSessionState:
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
    return state.model_copy(
        update={
            "phase": "completed",
            "activeRoundId": None,
            "finalReportReady": True,
            "finalReport": "local report",
            "rounds": [completed_round, state.rounds[1].model_copy(update={"status": "skipped"})],
        },
        deep=True,
    )


def build_manifest(**overrides: object) -> InterviewEvaluationManifest:
    payload = {
        "schemaVersion": 1,
        "interviewId": "thread-1",
        "threadId": "thread-1",
        "expectedTaskIds": ["task-1"],
        "completedTaskIds": ["task-1"],
        "failedTaskIds": [],
        "sealed": False,
        "updatedAt": "2026-06-15T00:00:01Z",
    }
    payload.update(overrides)
    return InterviewEvaluationManifest.model_validate(payload)


def build_result() -> LlmAnswerEvaluationResult:
    return LlmAnswerEvaluationResult.model_validate(
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


def test_complete_final_report_ready_only_with_complete_evaluations() -> None:
    store = FakeStore(manifest=build_manifest(), results=[build_result()])

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        max_wait_seconds=0,
    )

    assert result["ready"] is True
    assert result["state"].finalReportReady is True
    assert result["state"].rounds[0].nodes[0].answerAttempts[0].score.weightedTotal == 8.4
    assert store.sealed is True


def test_complete_final_report_blocks_failed_evaluations_without_partial_report() -> None:
    store = FakeStore(
        manifest=build_manifest(
            completedTaskIds=[],
            failedTaskIds=["task-1"],
            sealed=True,
        ),
        results=[],
    )

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        max_wait_seconds=0,
    )

    assert result["ready"] is False
    assert result["state"].finalReportReady is False
    assert result["state"].finalReport is None
    assert "失败" in result["assistant_reply"]


def test_complete_final_report_blocks_pending_evaluations_without_partial_report() -> None:
    store = FakeStore(
        manifest=build_manifest(completedTaskIds=[], sealed=True),
        results=[],
    )

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        max_wait_seconds=0,
    )

    assert result["ready"] is False
    assert result["state"].phase == "wrap-up"
    assert result["state"].finalReportReady is False
    assert result["state"].finalReport is None
    assert "等待异步评分完成" in result["assistant_reply"]
