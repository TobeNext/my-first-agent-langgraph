from app.domain.report_status import (
    mark_interview_report_read,
    resolve_interview_report_status,
)
from app.schemas.interview_report import InterviewReportReadReceipt, InterviewReportRecord

NOW = "2026-06-19T00:00:00.000Z"


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


class FakeRepository:
    def __init__(self, report: InterviewReportRecord | None = None) -> None:
        self.report = report
        self.read_receipts: dict[tuple[str, str], str] = {}

    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None:
        return self.report if self.report and self.report.interview_id == interview_id else None

    def get_markdown_by_interview_id(self, interview_id: str) -> str | None:
        report = self.get_report_by_interview_id(interview_id)
        return report.markdown if report else None

    def mark_read(
        self,
        interview_id: str,
        thread_id: str,
        read_at: str,
        receipt_id: str | None = None,
    ) -> InterviewReportReadReceipt:
        self.read_receipts[(interview_id, thread_id)] = read_at
        return InterviewReportReadReceipt(
            id=receipt_id or "receipt-1",
            interview_id=interview_id,
            thread_id=thread_id,
            read_at=read_at,
        )

    def get_read_receipt(
        self,
        interview_id: str,
        thread_id: str,
    ) -> InterviewReportReadReceipt | None:
        read_at = self.read_receipts.get((interview_id, thread_id))
        if not read_at:
            return None
        return InterviewReportReadReceipt(
            id="receipt-1",
            interview_id=interview_id,
            thread_id=thread_id,
            read_at=read_at,
        )


async def test_resolve_status_pending_when_db_has_no_report() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        repository=FakeRepository(),
    )

    assert status.reportState == "generating"
    assert status.blockingReason == "pending"
    assert status.markdownAvailable is False
    assert status.reportId is None


async def test_resolve_status_failed_from_db_report() -> None:
    status = await resolve_interview_report_status(
        thread_id="thread-1",
        repository=FakeRepository(build_report(status="failed", markdown="")),
    )

    assert status.reportState == "failed"
    assert status.failedCount == 1
    assert status.markdownAvailable is False
    assert status.blockingReason == "failed"


async def test_resolve_status_ready_unread_and_read_from_db() -> None:
    repository = FakeRepository(build_report())

    unread_status = await resolve_interview_report_status(
        thread_id="thread-1",
        repository=repository,
    )
    repository.read_receipts[("thread-1", "thread-1")] = NOW
    read_status = await resolve_interview_report_status(
        thread_id="thread-1",
        repository=repository,
    )

    assert unread_status.reportState == "ready"
    assert unread_status.markdownAvailable is True
    assert unread_status.reportId == "report-1"
    assert unread_status.unreadCount == 1
    assert read_status.unreadCount == 0


async def test_mark_interview_report_read_writes_db_receipt_only() -> None:
    repository = FakeRepository(build_report())

    receipt = await mark_interview_report_read(
        thread_id="thread-1",
        repository=repository,
        now=lambda: NOW,
    )

    assert receipt.read_at == NOW
    assert repository.read_receipts[("thread-1", "thread-1")] == NOW
