from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from app.schemas.answer_evaluation import InterviewEvaluationManifest
from app.schemas.interview_report import (
    InterviewReportManifest,
    InterviewReportReadReceipt,
    InterviewReportRecord,
    InterviewReportStatus,
)


class ReportStatusEvaluationStoreLike(Protocol):
    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None: ...


class ReportStatusGenerationStoreLike(Protocol):
    async def read_manifest(self, interview_id: str) -> InterviewReportManifest | None: ...

    async def mark_read(self, interview_id: str, read_at: str) -> None: ...


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
    evaluation_store: ReportStatusEvaluationStoreLike,
    report_store: ReportStatusGenerationStoreLike,
    repository: ReportStatusRepositoryLike,
) -> InterviewReportStatus:
    evaluation_manifest = await evaluation_store.read_manifest(thread_id)
    report_manifest = await report_store.read_manifest(thread_id)
    report = repository.get_report_by_interview_id(thread_id)
    read_receipt = repository.get_read_receipt(thread_id, thread_id)
    markdown_available = bool(report and report.status == "succeeded" and report.markdown)

    if report_manifest:
        expected_count = report_manifest.evaluationExpectedCount
        completed_count = report_manifest.evaluationCompletedCount
        failed_count = report_manifest.evaluationFailedCount
        updated_at = report_manifest.updatedAt
        report_id = report_manifest.reportId or (report.id if markdown_available else None)
        sealed = bool(evaluation_manifest.sealed) if evaluation_manifest else False
        if report_manifest.status == "succeeded" and markdown_available:
            report_state = "ready"
            blocking_reason = None
        elif report_manifest.status == "failed":
            report_state = "failed"
            blocking_reason = "failed"
        elif not sealed:
            report_state = "generating"
            blocking_reason = "not-sealed"
        else:
            report_state = "generating"
            blocking_reason = "pending"
    elif markdown_available and report:
        expected_count = _evaluation_expected_count(evaluation_manifest)
        completed_count = _evaluation_completed_count(evaluation_manifest)
        failed_count = _evaluation_failed_count(evaluation_manifest)
        updated_at = report.updated_at
        report_id = report.id
        sealed = bool(evaluation_manifest.sealed) if evaluation_manifest else True
        report_state = "ready"
        blocking_reason = None
    else:
        expected_count = _evaluation_expected_count(evaluation_manifest)
        completed_count = _evaluation_completed_count(evaluation_manifest)
        failed_count = _evaluation_failed_count(evaluation_manifest)
        updated_at = evaluation_manifest.updatedAt if evaluation_manifest else None
        report_id = None
        sealed = bool(evaluation_manifest.sealed) if evaluation_manifest else False
        report_state = "not-started"
        blocking_reason = "manifest-missing"

    return InterviewReportStatus.model_validate(
        {
            "threadId": thread_id,
            "reportState": report_state,
            "sealed": sealed,
            "expectedCount": expected_count,
            "completedCount": completed_count,
            "failedCount": failed_count,
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
    report_store: ReportStatusGenerationStoreLike,
    now: Callable[[], str] | None = None,
) -> InterviewReportReadReceipt:
    read_at = now() if now else _utc_now()
    receipt = repository.mark_read(thread_id, thread_id, read_at)
    await report_store.mark_read(thread_id, read_at)
    return receipt


def _evaluation_expected_count(manifest: InterviewEvaluationManifest | None) -> int:
    return len(manifest.expectedTaskIds) if manifest else 0


def _evaluation_completed_count(manifest: InterviewEvaluationManifest | None) -> int:
    return len(manifest.completedTaskIds) if manifest else 0


def _evaluation_failed_count(manifest: InterviewEvaluationManifest | None) -> int:
    return len(manifest.failedTaskIds) if manifest else 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
