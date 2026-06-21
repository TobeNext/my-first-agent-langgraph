from __future__ import annotations

import json
import re
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.interview_report import (
    InterviewUserMemoryProfile,
    InterviewUserMemoryRecord,
    InterviewUserMemoryWrite,
)

REPORT_MARKDOWN_EXCERPT_LIMIT = 2000


class UpdateInterviewMemoryInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    userId: str = Field(min_length=1)
    sourceInterviewId: str = Field(min_length=1)
    sourceThreadId: str = Field(min_length=1)
    targetRole: str = Field(min_length=1)
    overallScore: float | None = Field(default=None, ge=0, le=10)
    weaknessSummary: list[str] = Field(default_factory=list)
    missingPoints: list[str] = Field(default_factory=list)
    improvementAdvice: list[str] = Field(default_factory=list)
    reinforcementQuestionHints: list[str] = Field(default_factory=list)
    normalizedWeaknessKeys: list[str] = Field(default_factory=list)
    improvedAreas: list[str] = Field(default_factory=list)
    reportMarkdownExcerpt: str = ""
    embeddingText: str = Field(min_length=1)
    embeddingJson: str | None = None
    sourceReportCompletedAt: str = Field(min_length=1)
    summaryGeneratedAt: str = Field(min_length=1)


def update_interview_memory_tool(
    input_data: UpdateInterviewMemoryInput | dict,
    *,
    repository: InterviewReportRepository | None = None,
    max_memory_count: int | None = None,
) -> InterviewUserMemoryRecord:
    payload = (
        input_data
        if isinstance(input_data, UpdateInterviewMemoryInput)
        else UpdateInterviewMemoryInput.model_validate(input_data)
    )
    repo = repository or InterviewReportRepository()
    existing = repo.get_user_memory(payload.userId, payload.sourceInterviewId)
    now = payload.summaryGeneratedAt
    memory_id = existing.id if existing else f"user-memory-{uuid4()}"
    memory = InterviewUserMemoryWrite(
        id=memory_id,
        user_id=payload.userId,
        source_interview_id=payload.sourceInterviewId,
        source_thread_id=payload.sourceThreadId,
        target_role=payload.targetRole,
        overall_score=payload.overallScore,
        weakness_summary_json=_json(payload.weaknessSummary),
        missing_points_json=_json(payload.missingPoints),
        improvement_advice_json=_json(payload.improvementAdvice),
        reinforcement_question_hints_json=_json(payload.reinforcementQuestionHints),
        report_markdown_excerpt=_truncate(payload.reportMarkdownExcerpt),
        embedding_text=payload.embeddingText,
        embedding_json=payload.embeddingJson,
        source_report_completed_at=payload.sourceReportCompletedAt,
        summary_generated_at=payload.summaryGeneratedAt,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    profile = _next_profile(repo.get_user_memory_profile(payload.userId), payload, memory_id)
    return repo.write_user_memory_with_profile(
        memory=memory,
        profile=profile,
        max_memory_count=max_memory_count or get_settings().max_user_interview_memory_count,
    )


def _next_profile(
    current: InterviewUserMemoryProfile | None,
    payload: UpdateInterviewMemoryInput,
    memory_id: str,
) -> InterviewUserMemoryProfile:
    counters = _json_object(current.weakness_counters_json if current else "{}")
    existing_memory_ids = _json_list(current.last_memory_ids_json if current else "[]")
    is_new_memory = memory_id not in existing_memory_ids
    if is_new_memory:
        for key in _weakness_keys(payload):
            counters[key] = int(counters.get(key, 0)) + 1
    last_memory_ids = [item for item in existing_memory_ids if item != memory_id]
    last_memory_ids.append(memory_id)
    stable_weaknesses = [
        key for key, _count in sorted(counters.items(), key=lambda item: (-int(item[1]), item[0]))
    ][:10]
    improved_areas = _merge_unique(
        _json_list(current.improved_areas_json if current else "[]"),
        payload.improvedAreas,
    )
    summary_count = (current.summary_count if current else 0) + (1 if is_new_memory else 0)
    recurring_mistakes = [key for key in stable_weaknesses if counters.get(key, 0) > 1]
    return InterviewUserMemoryProfile(
        user_id=payload.userId,
        stable_weaknesses_json=_json(stable_weaknesses),
        improved_areas_json=_json(improved_areas),
        recurring_mistakes_json=_json(recurring_mistakes),
        weakness_counters_json=_json(counters),
        last_memory_ids_json=_json(last_memory_ids[-20:]),
        summary_count=summary_count,
        updated_at=payload.summaryGeneratedAt,
    )


def _weakness_keys(payload: UpdateInterviewMemoryInput) -> list[str]:
    raw_keys = [
        *payload.normalizedWeaknessKeys,
        *payload.weaknessSummary,
        *payload.missingPoints,
    ]
    return _merge_unique([], [_normalize_weakness_key(item) for item in raw_keys])


def _normalize_weakness_key(value: str) -> str:
    normalized = re.sub(r"[\s_]+", "-", value.strip().lower())
    normalized = re.sub(r"[^a-z0-9\-\u4e00-\u9fff]+", "", normalized)
    return normalized.strip("-") or "general-improvement"


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    values: list[str] = []
    for item in [*existing, *incoming]:
        normalized = item.strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, str)] if isinstance(parsed, list) else []


def _json_object(value: str) -> dict[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): int(count) for key, count in parsed.items() if isinstance(count, int | float)}


def _truncate(value: str, *, limit: int = REPORT_MARKDOWN_EXCERPT_LIMIT) -> str:
    normalized = value.strip()
    return normalized[:limit].rstrip()
