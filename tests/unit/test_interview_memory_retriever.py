from pathlib import Path

from app.domain.interview_memory_retriever import (
    merge_canonical_user_memories,
    retrieve_user_interview_memory,
    trim_historical_memory_budget,
)
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.interview_report import InterviewUserMemoryProfile, InterviewUserMemoryWrite
from app.schemas.interview_state import HistoricalInterviewMemoryState

NOW = "2026-06-19T00:00:00Z"


def test_retrieve_user_interview_memory_filters_by_user_and_keyword_relevance(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(
        _memory(
            "memory-rag",
            "user-a",
            "interview-rag",
            embedding_text="RAG 检索 失败降级 指标阈值",
            missing_points_json='["缺少失败降级","缺少指标阈值"]',
        )
    )
    repository.write_user_memory(
        _memory(
            "memory-java",
            "user-a",
            "interview-java",
            embedding_text="Java 事务传播 回滚边界",
            missing_points_json='["缺少事务传播"]',
            summary_generated_at="2026-06-19T00:02:00Z",
        )
    )
    repository.write_user_memory(
        _memory(
            "memory-other",
            "user-b",
            "interview-other",
            embedding_text="RAG 检索 失败降级",
        )
    )

    memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG 检索",
        job_description="需要处理失败降级和指标阈值",
        repository=repository,
        top_k=1,
    )

    assert memory.hasMemory is True
    assert memory.sourceInterviewIds == ["interview-rag"]
    assert memory.missingPoints == ["缺少失败降级", "缺少指标阈值"]


def test_retrieve_user_interview_memory_uses_keyword_fallback_when_embedding_unavailable(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(
        _memory(
            "memory-rag",
            "user-a",
            "interview-rag",
            embedding_text="RAG 检索 失败降级 指标阈值",
            embedding_json="{not-json",
            missing_points_json='["缺少失败降级"]',
            summary_generated_at="2026-06-19T00:00:00Z",
        )
    )
    repository.write_user_memory(
        _memory(
            "memory-java",
            "user-a",
            "interview-java",
            embedding_text="Java 事务传播 回滚边界",
            embedding_json=None,
            missing_points_json='["缺少事务传播"]',
            summary_generated_at="2026-06-19T00:03:00Z",
        )
    )

    memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="需要处理失败降级和指标阈值",
        repository=repository,
        top_k=1,
    )

    assert memory.hasMemory is True
    assert memory.sourceInterviewIds == ["interview-rag"]
    assert memory.missingPoints == ["缺少失败降级"]


def test_retrieve_user_interview_memory_loads_profile(tmp_path: Path) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(_memory("memory-rag", "user-a", "interview-rag"))
    repository.upsert_user_memory_profile(
        InterviewUserMemoryProfile(
            user_id="user-a",
            stable_weaknesses_json='["failure-degradation"]',
            improved_areas_json='["链路解释"]',
            recurring_mistakes_json='["failure-degradation"]',
            weakness_counters_json='{"failure-degradation":2}',
            last_memory_ids_json='["memory-rag"]',
            summary_count=2,
            updated_at=NOW,
        )
    )

    memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
    )

    assert memory.profile.stableWeaknesses == ["failure-degradation"]
    assert memory.profile.improvedAreas == ["链路解释"]
    assert memory.profile.recurringMistakes == ["failure-degradation"]
    assert memory.profile.updatedAt == NOW


def test_retrieve_user_interview_memory_uses_latest_canonical_weakness(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    repository.write_user_memory(
        _memory(
            "memory-old",
            "user-a",
            "interview-old",
            missing_points_json='["缺少失败降级"]',
            summary_generated_at="2026-06-19T00:00:00Z",
        )
    )
    repository.write_user_memory(
        _memory(
            "memory-new",
            "user-a",
            "interview-new",
            missing_points_json='["缺少失败降级"]',
            summary_generated_at="2026-06-19T00:02:00Z",
        )
    )

    memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
        top_k=3,
    )

    assert memory.sourceInterviewIds == ["interview-new"]
    assert memory.missingPoints == ["缺少失败降级"]


def test_merge_canonical_user_memories_prefers_latest_summary_timestamp(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    old = repository.write_user_memory(
        _memory(
            "memory-old",
            "user-a",
            "interview-old",
            missing_points_json='["缺少失败降级"]',
            summary_generated_at="2026-06-19T00:00:00Z",
        )
    )
    new = repository.write_user_memory(
        _memory(
            "memory-new",
            "user-a",
            "interview-new",
            missing_points_json='["缺少失败降级"]',
            summary_generated_at="2026-06-19T00:02:00Z",
        )
    )

    canonical = merge_canonical_user_memories([old, new])

    assert [item.id for item in canonical] == ["memory-new"]


def test_retrieve_user_interview_memory_triggers_best_effort_lazy_backfill(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))
    calls: list[str] = []

    memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
        lazy_backfill=lambda user_id: calls.append(user_id),
    )
    memory_after_error = retrieve_user_interview_memory(
        user_id="user-b",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
        lazy_backfill=lambda _user_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert memory.hasMemory is False
    assert memory_after_error.hasMemory is False
    assert calls == ["user-a"]


def test_trim_historical_memory_budget_drops_lower_priority_fields() -> None:
    memory = HistoricalInterviewMemoryState(
        hasMemory=True,
        sourceInterviewIds=["interview-1"],
        weaknesses=["核心弱项" * 20],
        missingPoints=["漏点" * 40],
        improvementAdvice=["建议" * 80, "建议二" * 80],
        reinforcementQuestionHints=["追问" * 80, "追问二" * 80],
    )

    trimmed = trim_historical_memory_budget(memory, budget_chars=260)

    assert trimmed.hasMemory is True
    assert trimmed.sourceInterviewIds == ["interview-1"]
    assert trimmed.reinforcementQuestionHints == []
    assert trimmed.improvementAdvice == []
    assert len(trimmed.model_dump_json(exclude_none=True)) <= 260


def test_retrieve_user_interview_memory_returns_empty_without_user_or_memory(
    tmp_path: Path,
) -> None:
    repository = InterviewReportRepository(str(tmp_path / "reports.db"))

    no_user = retrieve_user_interview_memory(
        user_id=None,
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
    )
    no_memory = retrieve_user_interview_memory(
        user_id="user-a",
        target_role="AI Engineer",
        professional_skills="RAG",
        job_description="失败降级",
        repository=repository,
    )

    assert no_user.hasMemory is False
    assert no_memory.hasMemory is False
    assert no_memory.sourceInterviewIds == []


def _memory(
    memory_id: str,
    user_id: str,
    interview_id: str,
    *,
    embedding_text: str = "RAG 检索 失败降级",
    embedding_json: str | None = None,
    missing_points_json: str = '["缺少失败降级"]',
    summary_generated_at: str = NOW,
) -> InterviewUserMemoryWrite:
    return InterviewUserMemoryWrite(
        id=memory_id,
        user_id=user_id,
        source_interview_id=interview_id,
        source_thread_id=f"thread-{interview_id}",
        target_role="AI Engineer",
        overall_score=6.5,
        weakness_summary_json='["失败降级覆盖不足"]',
        missing_points_json=missing_points_json,
        improvement_advice_json='["补充降级策略"]',
        reinforcement_question_hints_json='["追问失败时如何降级"]',
        report_markdown_excerpt="# 报告",
        embedding_text=embedding_text,
        embedding_json=embedding_json,
        source_report_completed_at=NOW,
        summary_generated_at=summary_generated_at,
        created_at=NOW,
        updated_at=summary_generated_at,
    )
