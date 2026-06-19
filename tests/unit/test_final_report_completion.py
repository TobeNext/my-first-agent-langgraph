from app.graphs.nodes.process_user_reply import complete_final_report_with_async_evaluations
from app.schemas.answer_evaluation import (
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)
from app.schemas.interview_report import ReportGenerationTask
from app.schemas.interview_state import AnswerAttemptState, InterviewSessionState
from tests.unit.test_interview_state_machine import _score, _state_fixture

REPORT_GENERATING_REPLY = "面试已结束，报告生成中。生成进度和最终报告可在右上角通知中查看。"


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


class FakeReportStore:
    def __init__(self) -> None:
        self.tasks: list[ReportGenerationTask] = []

    async def enqueue_task(self, task: ReportGenerationTask) -> None:
        self.tasks.append(task)


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


def test_complete_final_report_seals_manifest_and_enqueues_report_generation() -> None:
    store = FakeStore(manifest=build_manifest(), results=[build_result()])
    report_store = FakeReportStore()

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        report_store=report_store,  # type: ignore[arg-type]
        resource_id="resource-1",
        max_wait_seconds=0,
    )

    assert result["ready"] is False
    assert result["state"].phase == "wrap-up"
    assert result["state"].finalReportReady is False
    assert result["state"].finalReport is None
    assert result["assistant_reply"] == REPORT_GENERATING_REPLY
    assert store.sealed is True
    assert len(report_store.tasks) == 1
    assert report_store.tasks[0].interviewId == "thread-1"
    assert report_store.tasks[0].resourceId == "resource-1"
    assert report_store.tasks[0].evaluationManifestKey == "interview:thread-1:evaluation:manifest"


def test_complete_final_report_still_enqueues_when_evaluations_failed() -> None:
    store = FakeStore(
        manifest=build_manifest(
            completedTaskIds=[],
            failedTaskIds=["task-1"],
            sealed=True,
        ),
        results=[],
    )
    report_store = FakeReportStore()

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        report_store=report_store,  # type: ignore[arg-type]
        max_wait_seconds=0,
    )

    assert result["ready"] is False
    assert result["state"].finalReportReady is False
    assert result["state"].finalReport is None
    assert result["assistant_reply"] == REPORT_GENERATING_REPLY
    assert store.sealed is True
    assert len(report_store.tasks) == 1


def test_complete_final_report_keeps_short_reply_for_pending_evaluations() -> None:
    store = FakeStore(
        manifest=build_manifest(completedTaskIds=[], sealed=True),
        results=[],
    )
    report_store = FakeReportStore()

    result = complete_final_report_with_async_evaluations(
        completed_state_with_attempt(),
        store=store,  # type: ignore[arg-type]
        report_store=report_store,  # type: ignore[arg-type]
        max_wait_seconds=0,
    )

    assert result["ready"] is False
    assert result["state"].phase == "wrap-up"
    assert result["state"].finalReportReady is False
    assert result["state"].finalReport is None
    assert result["assistant_reply"] == REPORT_GENERATING_REPLY
    assert store.sealed is True
    assert len(report_store.tasks) == 1
