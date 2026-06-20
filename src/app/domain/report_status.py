from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from app.schemas.interview_report import (
    InterviewReportReadReceipt,
    InterviewReportRecord,
    InterviewReportStatus,
)


class ReportStatusRepositoryLike(Protocol):
    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None: ...

    def get_markdown_by_interview_id(self, interview_id: str) -> str | None: ...

    def mark_read(
        self,
        interview_id: str,
        thread_id: str,
        read_at: str,
        receipt_id: str | None = None,
    ) -> InterviewReportReadReceipt: ...

    def get_read_receipt(
        self,
        interview_id: str,
        thread_id: str,
    ) -> InterviewReportReadReceipt | None: ...


async def resolve_interview_report_status(
    *,
    thread_id: str,
    repository: ReportStatusRepositoryLike,
) -> InterviewReportStatus:
    report = repository.get_report_by_interview_id(thread_id)
    read_receipt = repository.get_read_receipt(thread_id, thread_id)

    if report and report.status == "succeeded" and report.markdown:
        report_state = "ready"
        markdown_available = True
        report_id = report.id
        updated_at = report.updated_at
        blocking_reason = None
    elif report and report.status == "failed":
        report_state = "failed"
        markdown_available = False
        report_id = report.id
        updated_at = report.updated_at
        blocking_reason = "failed"
    else:
        report_state = "generating"
        markdown_available = False
        report_id = None
        updated_at = report.updated_at if report else None
        blocking_reason = "pending"

    return InterviewReportStatus.model_validate(
        {
            "threadId": thread_id,
            "reportState": report_state,
            "sealed": True,
            "expectedCount": 0,
            "completedCount": 0,
            "failedCount": 1 if report_state == "failed" else 0,
            "unreadCount": 1 if markdown_available and not read_receipt else 0,
            "markdownAvailable": markdown_available,
            "reportId": report_id,
            "updatedAt": updated_at,
            "blockingReason": blocking_reason,
        }
    )


def get_report_markdown(
    *,
    thread_id: str,
    repository: ReportStatusRepositoryLike,
) -> str | None:
    return repository.get_markdown_by_interview_id(thread_id)


async def mark_interview_report_read(
    *,
    thread_id: str,
    repository: ReportStatusRepositoryLike,
    now: Callable[[], str] | None = None,
) -> InterviewReportReadReceipt:
    read_at = now() if now else _utc_now()
    return repository.mark_read(thread_id, thread_id, read_at)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
