from app.domain.report_status import (
    mark_interview_report_read,
    resolve_interview_report_status,
)
from app.schemas.answer_evaluation import InterviewEvaluationManifest
from app.schemas.interview_report import InterviewReportManifest, InterviewReportRecord

NOW = "2026-06-19T00:00:00.000Z"


def build_evaluation_manifest(**overrides: object) -> InterviewEvaluationManifest:
    payload = {
        "schemaVersion": 1,
        "interviewId": "thread-1",
        "threadId": "thread-1",
        "expectedTaskIds": ["task-1"],
        "completedTaskIds": [],
        "failedTaskIds": [],
        "sealed": True,
        "sealedAt": NOW,
        "updatedAt": NOW,
    }
    payload.update(overrides)
    return InterviewEvaluationManifest.model_validate(payload)


def build_report_manifest(**overrides: object) -> InterviewReportManifest:
    payload = {
        "schemaVersion": 1,
        "interviewId": "thread-1",
        "threadId": "thread-1",
        "taskId": "report-task-1",
        "status": "pending",
        "evaluationExpectedCount": 1,
        "evaluationCompletedCount": 0,
        "evaluationFailedCount": 0,
        "markdownAvailable": False,
        "attempts": 1,
        "createdAt": NOW,
        "updatedAt": NOW,
    }
    payload.update(overrides)
    return InterviewReportManifest.model_validate(payload)


def build_report(**overrides: object) -> InterviewReportRecord:
    payload = {
        "id": "report-1",
        "interview_id": "thread-1",
        "thread_id": "thread-1",
        "target_role": "Backend Engineer",
        "response_language": "zh",
        "status": "succeeded",
        "overall_score": 8,
        "markdown": "## Report",
        "structured_json": "{}",
        "prompt_version": "report-generation-v1",
        "model_name": "mock",
        "source_evaluation_manifest_json": "{}",
        "created_at": NOW,
        "updated_at": NOW,
        "completed_at": NOW,
    }
    payload.update(overrides)
    return InterviewReportRecord(**payload)


class FakeEvaluationStore:
    def __init__(self, manifest: InterviewEvaluationManifest | None) -> None:
        self.manifest = manifest

    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None:
        return self.manifest


class FakeReportStore:
    def __init__(self, manifest: InterviewReportManifest | None) -> None:
        self.manifest = manifest
        self.read_receipts: dict[str, str] = {}

    async def read_manifest(self, interview_id: str) -> InterviewReportManifest | None:
        return self.manifest

    async def mark_read(self, interview_id: str, read_at: str) -> None:
        self.read_receipts[interview_id] = read_at


class FakeRepository:
    def __init__(self, report: InterviewReportRecord | None = None) -> None:
        self.report = report
        self.read_receipts: dict[tuple[str, str], str] = {}

    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None:
        return self.report

    def get_markdown_by_interview_id(self, interview_id: str) -> str | None:
        return self.report.markdown if self.report else None

    def mark_read(
        self,
        interview_id: str,
        thread_id: str,
        read_at: str,
        receipt_id: str | None = None,
    ):
        self.read_receipts[(interview_id, thread_id)] = read_at

        class Receipt:
            pass

        receipt = Receipt()
        receipt.read_at = read_at
        return receipt

    def get_read_receipt(self, interview_id: str, thread_id: str):
        read_at = self.read_receipts.get((interview_id, thread_id))
        if not read_at:
            return None

        class Receipt:
            pass

        receipt = Receipt()
        receipt.read_at = read_at
        return receipt


async def test_resolve_status_manifest_missing() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(None),
        report_store=FakeReportStore(None),
        repository=FakeRepository(),
    )

    assert status.reportState == "not-started"
    assert status.blockingReason == "manifest-missing"
    assert status.markdownAvailable is False


async def test_resolve_status_generating_from_report_manifest() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(build_evaluation_manifest()),
        report_store=FakeReportStore(
            build_report_manifest(evaluationExpectedCount=2, evaluationCompletedCount=1)
        ),
        repository=FakeRepository(),
    )

    assert status.reportState == "generating"
    assert status.expectedCount == 2
    assert status.completedCount == 1
    assert status.blockingReason == "pending"


async def test_resolve_status_failed_from_report_manifest() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(build_evaluation_manifest()),
        report_store=FakeReportStore(
            build_report_manifest(status="failed", evaluationFailedCount=1)
        ),
        repository=FakeRepository(),
    )

    assert status.reportState == "failed"
    assert status.failedCount == 1
    assert status.blockingReason == "failed"


async def test_resolve_status_ready_unread_and_read() -> None:
    repository = FakeRepository(build_report())

    unread_status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(
            build_evaluation_manifest(completedTaskIds=["task-1"])
        ),
        report_store=FakeReportStore(
            build_report_manifest(
                status="succeeded",
                reportId="report-1",
                markdownAvailable=True,
                evaluationCompletedCount=1,
            )
        ),
        repository=repository,
    )
    repository.read_receipts[("thread-1", "thread-1")] = NOW
    read_status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(
            build_evaluation_manifest(completedTaskIds=["task-1"])
        ),
        report_store=FakeReportStore(
            build_report_manifest(
                status="succeeded",
                reportId="report-1",
                markdownAvailable=True,
                evaluationCompletedCount=1,
            )
        ),
        repository=repository,
    )

    assert unread_status.reportState == "ready"
    assert unread_status.unreadCount == 1
    assert read_status.unreadCount == 0


async def test_resolve_status_recovers_ready_from_db_when_redis_manifest_lost() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        evaluation_store=FakeEvaluationStore(
            build_evaluation_manifest(completedTaskIds=["task-1"])
        ),
        report_store=FakeReportStore(None),
        repository=FakeRepository(build_report()),
    )

    assert status.reportState == "ready"
    assert status.markdownAvailable is True
    assert status.reportId == "report-1"


async def test_mark_interview_report_read_writes_db_and_redis_receipt() -> None:
    repository = FakeRepository(build_report())
    report_store = FakeReportStore(None)

    receipt = await mark_interview_report_read(
        thread_id="thread-1",
        repository=repository,
        report_store=report_store,
        now=lambda: NOW,
    )

    assert receipt.read_at == NOW
    assert repository.read_receipts[("thread-1", "thread-1")] == NOW
    assert report_store.read_receipts["thread-1"] == NOW
