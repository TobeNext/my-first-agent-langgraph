from app.domain.report_generation_enqueue import (
    build_report_generation_task,
    enqueue_report_generation_task_best_effort,
)
from app.schemas.interview_report import ReportGenerationTask
from tests.unit.test_interview_state_machine import _state_fixture

NOW = "2026-06-19T00:00:00.000Z"


class FakeStore:
    def __init__(self) -> None:
        self.tasks: list[ReportGenerationTask] = []

    async def enqueue_task(self, task: ReportGenerationTask) -> None:
        self.tasks.append(task)


class FailingStore:
    async def enqueue_task(self, task: ReportGenerationTask) -> None:
        raise RuntimeError("Redis unavailable")


def test_build_report_generation_task_matches_redis_contract() -> None:
    state = _state_fixture(flow_test=False)

    task = build_report_generation_task(
        state=state,
        resource_id="resource-1",
        now=lambda: NOW,
        create_task_id=lambda session: f"task-{session.threadId}",
    )

    assert task.schemaVersion == 1
    assert task.taskId == f"task-{state.threadId}"
    assert task.interviewId == state.threadId
    assert task.threadId == state.threadId
    assert task.resourceId == "resource-1"
    assert task.targetRole == state.targetRole
    assert task.responseLanguage == state.responseLanguage
    assert task.evaluationManifestKey == f"interview:{state.threadId}:evaluation:manifest"
    assert task.createdAt == NOW


def test_enqueue_report_generation_task_best_effort_uses_injected_store() -> None:
    state = _state_fixture(flow_test=False)
    store = FakeStore()

    task = enqueue_report_generation_task_best_effort(
        state=state,
        store=store,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert store.tasks == [task]


def test_enqueue_report_generation_task_best_effort_keeps_flow_safe_on_store_failure() -> None:
    state = _state_fixture(flow_test=False)

    task = enqueue_report_generation_task_best_effort(
        state=state,
        store=FailingStore(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert task.interviewId == state.threadId
