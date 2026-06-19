import sqlite3
from pathlib import Path

from app.integrations.report_repository import (
    InterviewReportRepository,
    sqlite_report_database_path,
)
from app.schemas.interview_report import InterviewReportItemWrite, InterviewReportWrite

NOW = "2026-06-19T00:00:00Z"


def test_sqlite_report_database_path_normalizes_sqlite_url() -> None:
    assert sqlite_report_database_path("sqlite:///./interview_reports.db") == "interview_reports.db"


def test_repository_initializes_report_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "reports.db"

    InterviewReportRepository(str(db_path))

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {
        "interview_reports",
        "interview_report_items",
        "interview_report_reads",
    }.issubset(tables)


def test_write_report_is_idempotent_for_succeeded_report(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    report = _report()

    first = repository.write_report(report)
    second = repository.write_report(
        InterviewReportWrite(
            **{
                **report.__dict__,
                "id": "report-second",
                "markdown": "# Different",
                "items": [
                    InterviewReportItemWrite(
                        **{**report.items[0].__dict__, "id": "item-second"}
                    )
                ],
            }
        )
    )

    assert second == first
    assert repository.get_markdown_by_interview_id("interview-1") == "# 模拟面试报告"
    assert len(repository.list_items(first.id)) == 1


def test_repository_reads_markdown_structured_json_items_and_receipt(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    stored = repository.write_report(_report())
    items = repository.list_items(stored.id)
    receipt = repository.mark_read("interview-1", "thread-1", "2026-06-19T00:01:00Z")
    updated_receipt = repository.mark_read("interview-1", "thread-1", "2026-06-19T00:02:00Z")

    assert stored.structured_json == '{"summary":{"overallScore":8}}'
    assert repository.get_markdown_by_interview_id("interview-1") == "# 模拟面试报告"
    assert items[0].missing_points_json == '["事务回滚边界"]'
    assert receipt.interview_id == "interview-1"
    assert updated_receipt.read_at == "2026-06-19T00:02:00Z"


def _report() -> InterviewReportWrite:
    return InterviewReportWrite(
        id="report-1",
        interview_id="interview-1",
        thread_id="thread-1",
        target_role="Backend Engineer",
        response_language="zh",
        status="succeeded",
        overall_score=8.0,
        markdown="# 模拟面试报告",
        structured_json='{"summary":{"overallScore":8}}',
        prompt_version="report-generation-v1",
        model_name="mock/interview-runtime",
        source_evaluation_manifest_json='{"schemaVersion":1}',
        created_at=NOW,
        updated_at=NOW,
        completed_at=NOW,
        items=[
            InterviewReportItemWrite(
                id="item-1",
                task_id="task-1",
                attempt_id="attempt-1",
                node_id="node-1",
                round_id="round-1",
                round_type="professional-skills",
                target_type="main-question",
                question="请说明 Spring 事务传播机制。",
                candidate_answer="我会说明 REQUIRED 和 REQUIRES_NEW。",
                score=8.0,
                comment="回答覆盖了核心传播机制。",
                missing_points_json='["事务回滚边界"]',
                improvement_advice_json='["补充异常场景"]',
            )
        ],
    )
