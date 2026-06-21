import json
from pathlib import Path

from app.domain.interview_memory_tool import (
    REPORT_MARKDOWN_EXCERPT_LIMIT,
    update_interview_memory_tool,
)
from app.integrations.report_repository import InterviewReportRepository

NOW = "2026-06-19T00:00:00Z"


def test_update_interview_memory_tool_writes_one_memory_per_user_interview(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    first = update_interview_memory_tool(_payload(), repository=repository, max_memory_count=20)
    second = update_interview_memory_tool(
        {
            **_payload(),
            "weaknessSummary": ["更新后的事务弱项"],
            "summaryGeneratedAt": "2026-06-19T00:01:00Z",
        },
        repository=repository,
        max_memory_count=20,
    )

    memories = repository.list_user_memories("user-a")
    profile = repository.get_user_memory_profile("user-a")

    assert first.id == second.id
    assert len(memories) == 1
    assert memories[0].weakness_summary_json == '["更新后的事务弱项"]'
    assert profile is not None
    assert profile.summary_count == 1


def test_update_interview_memory_tool_enforces_user_capacity_without_cross_user_delete(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    update_interview_memory_tool(
        _payload(source_interview_id="interview-old"),
        repository=repository,
        max_memory_count=1,
    )
    update_interview_memory_tool(
        _payload(user_id="user-b", source_interview_id="interview-other"),
        repository=repository,
        max_memory_count=1,
    )

    update_interview_memory_tool(
        _payload(
            source_interview_id="interview-new",
            summary_generated_at="2026-06-19T00:02:00Z",
        ),
        repository=repository,
        max_memory_count=1,
    )

    assert [item.source_interview_id for item in repository.list_user_memories("user-a")] == [
        "interview-new"
    ]
    assert [item.source_interview_id for item in repository.list_user_memories("user-b")] == [
        "interview-other"
    ]


def test_update_interview_memory_tool_updates_profile_counters_and_recurring_mistakes(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    update_interview_memory_tool(
        _payload(source_interview_id="interview-1"),
        repository=repository,
        max_memory_count=20,
    )
    update_interview_memory_tool(
        _payload(
            source_interview_id="interview-2",
            summary_generated_at="2026-06-19T00:02:00Z",
        ),
        repository=repository,
        max_memory_count=20,
    )

    profile = repository.get_user_memory_profile("user-a")
    assert profile is not None
    counters = json.loads(profile.weakness_counters_json)
    assert counters["transaction-boundary"] == 2
    assert "transaction-boundary" in json.loads(profile.recurring_mistakes_json)
    assert profile.summary_count == 2


def test_update_interview_memory_tool_truncates_report_markdown_excerpt(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    stored = update_interview_memory_tool(
        _payload(report_markdown_excerpt="报告" * 2000),
        repository=repository,
        max_memory_count=20,
    )

    assert len(stored.report_markdown_excerpt) <= REPORT_MARKDOWN_EXCERPT_LIMIT


def _payload(
    *,
    user_id: str = "user-a",
    source_interview_id: str = "interview-1",
    summary_generated_at: str = NOW,
    report_markdown_excerpt: str = "# 模拟面试报告\n需要补充事务边界。",
) -> dict:
    return {
        "userId": user_id,
        "sourceInterviewId": source_interview_id,
        "sourceThreadId": f"thread-{source_interview_id}",
        "targetRole": "Backend Engineer",
        "overallScore": 6.5,
        "weaknessSummary": ["事务边界"],
        "missingPoints": ["缺少事务回滚边界"],
        "improvementAdvice": ["补充异常传播场景"],
        "reinforcementQuestionHints": ["追问事务失败时如何回滚"],
        "normalizedWeaknessKeys": ["transaction-boundary"],
        "improvedAreas": ["基础概念解释"],
        "reportMarkdownExcerpt": report_markdown_excerpt,
        "embeddingText": "事务边界 缺少事务回滚边界 补充异常传播场景",
        "embeddingJson": None,
        "sourceReportCompletedAt": NOW,
        "summaryGeneratedAt": summary_generated_at,
    }
