import sqlite3
from pathlib import Path

from app.integrations.report_repository import (
    InterviewReportRepository,
    sqlite_report_database_path,
)
from app.schemas.interview_report import (
    InterviewReportItemWrite,
    InterviewReportWrite,
    InterviewUserMemoryProfile,
    InterviewUserMemoryWrite,
)

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
        "interview_user_memories",
        "interview_user_memory_profiles",
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


def test_repository_writes_and_lists_user_memories_by_user(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    first = repository.write_user_memory(_memory("memory-1", "user-a", "interview-1"))
    second = repository.write_user_memory(
        _memory(
            "memory-2",
            "user-a",
            "interview-2",
            summary_generated_at="2026-06-19T00:02:00Z",
        )
    )
    repository.write_user_memory(_memory("memory-3", "user-b", "interview-3"))

    memories = repository.list_user_memories("user-a")

    assert first.user_id == "user-a"
    assert [item.id for item in memories] == [second.id, first.id]
    assert repository.list_user_memories("user-b")[0].id == "memory-3"


def test_repository_upserts_user_memory_for_same_user_and_interview(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    repository.write_user_memory(_memory("memory-1", "user-a", "interview-1"))
    updated = repository.write_user_memory(
        _memory(
            "memory-updated",
            "user-a",
            "interview-1",
            weakness_summary_json='["updated"]',
            updated_at="2026-06-19T00:03:00Z",
        )
    )

    memories = repository.list_user_memories("user-a")
    assert len(memories) == 1
    assert updated.id == "memory-updated"
    assert updated.weakness_summary_json == '["updated"]'


def test_repository_deletes_only_oldest_memory_for_user(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_report(_report())
    repository.write_user_memory(_memory("old", "user-a", "interview-old"))
    repository.write_user_memory(
        _memory("new", "user-a", "interview-new", summary_generated_at="2026-06-19T00:04:00Z")
    )
    repository.write_user_memory(_memory("other", "user-b", "interview-other"))

    deleted = repository.delete_oldest_user_memory("user-a")

    assert deleted and deleted.id == "old"
    assert [item.id for item in repository.list_user_memories("user-a")] == ["new"]
    assert [item.id for item in repository.list_user_memories("user-b")] == ["other"]
    assert repository.get_report_by_interview_id("interview-1") is not None


def test_repository_upserts_user_memory_profile(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    profile = InterviewUserMemoryProfile(
        user_id="user-a",
        stable_weaknesses_json='["事务边界"]',
        improved_areas_json="[]",
        recurring_mistakes_json='["异常处理"]',
        weakness_counters_json='{"transaction-boundary":1}',
        last_memory_ids_json='["memory-1"]',
        summary_count=1,
        updated_at=NOW,
    )

    stored = repository.upsert_user_memory_profile(profile)
    updated = repository.upsert_user_memory_profile(
        InterviewUserMemoryProfile(
            **{
                **profile.__dict__,
                "stable_weaknesses_json": '["事务边界","降级策略"]',
                "summary_count": 2,
                "updated_at": "2026-06-19T00:05:00Z",
            }
        )
    )

    assert stored.user_id == "user-a"
    assert updated.summary_count == 2
    assert updated.stable_weaknesses_json == '["事务边界","降级策略"]'


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


def _memory(
    memory_id: str,
    user_id: str,
    interview_id: str,
    *,
    weakness_summary_json: str = '["事务边界"]',
    summary_generated_at: str = NOW,
    updated_at: str = NOW,
) -> InterviewUserMemoryWrite:
    return InterviewUserMemoryWrite(
        id=memory_id,
        user_id=user_id,
        source_interview_id=interview_id,
        source_thread_id=f"thread-{interview_id}",
        target_role="Backend Engineer",
        overall_score=6.5,
        weakness_summary_json=weakness_summary_json,
        missing_points_json='["缺少事务回滚边界"]',
        improvement_advice_json='["补充异常传播场景"]',
        reinforcement_question_hints_json='["追问事务失败时如何回滚"]',
        report_markdown_excerpt="# 模拟面试报告\n需要补充事务边界。",
        embedding_text="事务边界 缺少事务回滚边界 补充异常传播场景",
        embedding_json=None,
        source_report_completed_at=NOW,
        summary_generated_at=summary_generated_at,
        created_at=NOW,
        updated_at=updated_at,
    )
