from app.domain.evaluation_report_reader import wait_and_read_interview_evaluations
from app.schemas.answer_evaluation import InterviewEvaluationManifest, LlmAnswerEvaluationResult
from tests.unit.test_redis_evaluation_store import build_result

NOW = "2026-06-09T00:00:00.000Z"


def build_manifest(overrides: dict | None = None) -> InterviewEvaluationManifest:
    payload = {
        "schemaVersion": 1,
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "expectedTaskIds": ["task-1", "task-2"],
        "completedTaskIds": [],
        "failedTaskIds": [],
        "sealed": False,
        "updatedAt": NOW,
    }
    payload.update(overrides or {})
    return InterviewEvaluationManifest.model_validate(payload)


class FakeAnswerEvaluationStore:
    def __init__(
        self,
        *,
        manifest: InterviewEvaluationManifest | None = None,
        results: list[LlmAnswerEvaluationResult] | None = None,
    ) -> None:
        self.manifest = manifest
        self.results = results or []

    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None:
        return self.manifest

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]:
        return self.results


async def test_wait_and_read_returns_evaluations_in_manifest_order() -> None:
    task_1_result = build_result({"taskId": "task-1", "attemptId": "attempt-1"})
    task_2_result = build_result(
        {"taskId": "task-2", "nodeId": "node-2", "attemptId": "attempt-2"}
    )
    store = FakeAnswerEvaluationStore(
        manifest=build_manifest(
            {
                "sealed": True,
                "completedTaskIds": ["task-2", "task-1"],
            }
        ),
        results=[task_2_result, task_1_result],
    )

    output = await wait_and_read_interview_evaluations(
        interview_id="interview-1",
        thread_id="thread-1",
        store=store,
        poll_interval_seconds=0.01,
        max_wait_seconds=0.1,
        now=lambda: 0,
    )

    assert output.ready is True
    assert [item.taskId for item in output.evaluations] == ["task-1", "task-2"]
    assert output.blockingReason is None


async def test_wait_and_read_does_not_return_partial_results() -> None:
    store = FakeAnswerEvaluationStore(
        manifest=build_manifest({"sealed": True, "completedTaskIds": ["task-1"]}),
        results=[build_result({"taskId": "task-1"})],
    )
    now = 0.0

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    output = await wait_and_read_interview_evaluations(
        interview_id="interview-1",
        thread_id="thread-1",
        store=store,
        poll_interval_seconds=0.025,
        max_wait_seconds=0.05,
        now=lambda: now,
        sleep=sleep,
    )

    assert output.ready is False
    assert output.evaluations == []
    assert output.blockingReason == "pending"
    assert output.waitElapsedMs == 50


async def test_wait_and_read_blocks_failed_and_missing_manifest() -> None:
    failed = await wait_and_read_interview_evaluations(
        interview_id="interview-1",
        thread_id="thread-1",
        store=FakeAnswerEvaluationStore(
            manifest=build_manifest(
                {
                    "sealed": True,
                    "completedTaskIds": ["task-1"],
                    "failedTaskIds": ["task-2"],
                }
            ),
            results=[build_result({"taskId": "task-1"})],
        ),
        max_wait_seconds=0,
        now=lambda: 0,
    )

    missing = await wait_and_read_interview_evaluations(
        interview_id="interview-1",
        thread_id="thread-1",
        store=FakeAnswerEvaluationStore(),
        max_wait_seconds=0,
        now=lambda: 0,
    )

    assert failed.blockingReason == "failed"
    assert failed.evaluations == []
    assert missing.blockingReason == "manifest-missing"


async def test_wait_and_read_rejects_thread_mismatch() -> None:
    try:
        await wait_and_read_interview_evaluations(
            interview_id="interview-1",
            thread_id="thread-1",
            store=FakeAnswerEvaluationStore(
                manifest=build_manifest({"threadId": "other-thread"})
            ),
            max_wait_seconds=0,
            now=lambda: 0,
        )
    except ValueError as exc:
        assert "Evaluation manifest thread mismatch" in str(exc)
    else:
        raise AssertionError("Expected thread mismatch to raise.")
