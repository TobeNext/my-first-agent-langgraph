from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.config import get_settings
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.interview_report import InterviewUserMemoryProfile, InterviewUserMemoryRecord
from app.schemas.interview_state import (
    HistoricalInterviewMemoryProfileState,
    HistoricalInterviewMemoryState,
)


def retrieve_user_interview_memory(
    *,
    user_id: str | None,
    target_role: str,
    professional_skills: str,
    job_description: str,
    current_main_question: str = "",
    repository: InterviewReportRepository | None = None,
    top_k: int | None = None,
    lazy_backfill: Callable[[str], None] | None = None,
) -> HistoricalInterviewMemoryState:
    if not user_id:
        return HistoricalInterviewMemoryState()
    repo = repository or InterviewReportRepository()
    memories = repo.list_user_memories(user_id)
    profile = _profile_state(repo.get_user_memory_profile(user_id))
    if not memories:
        _run_lazy_backfill(user_id, lazy_backfill)
        return HistoricalInterviewMemoryState(profile=profile)

    canonical = merge_canonical_user_memories(memories)
    query_tokens = _tokens(
        "\n".join([target_role, professional_skills, job_description, current_main_question])
    )
    limit = top_k or get_settings().user_memory_retrieval_top_k
    ranked = sorted(
        canonical,
        key=lambda memory: (
            _memory_score(memory, query_tokens),
            memory.summary_generated_at,
            memory.source_report_completed_at,
            memory.created_at,
        ),
        reverse=True,
    )[:limit]
    memory = HistoricalInterviewMemoryState(
        hasMemory=bool(ranked),
        sourceInterviewIds=[memory.source_interview_id for memory in ranked],
        weaknesses=_dedupe(
            item for memory in ranked for item in _json_list(memory.weakness_summary_json)
        ),
        missingPoints=_dedupe(
            item for memory in ranked for item in _json_list(memory.missing_points_json)
        ),
        improvementAdvice=_dedupe(
            item for memory in ranked for item in _json_list(memory.improvement_advice_json)
        ),
        reinforcementQuestionHints=_dedupe(
            item
            for memory in ranked
            for item in _json_list(memory.reinforcement_question_hints_json)
        ),
        profile=profile,
    )
    return trim_historical_memory_budget(
        memory,
        budget_chars=get_settings().user_memory_prompt_budget_chars,
    )


def trim_historical_memory_budget(
    memory: HistoricalInterviewMemoryState,
    *,
    budget_chars: int,
) -> HistoricalInterviewMemoryState:
    if _memory_size(memory) <= budget_chars:
        return memory
    next_memory = memory.model_copy(deep=True)
    for field_name in [
        "reinforcementQuestionHints",
        "improvementAdvice",
        "missingPoints",
        "weaknesses",
    ]:
        while getattr(next_memory, field_name) and _memory_size(next_memory) > budget_chars:
            values = list(getattr(next_memory, field_name))
            values.pop()
            next_memory = next_memory.model_copy(update={field_name: values}, deep=True)
    return next_memory


def merge_canonical_user_memories(
    memories: list[InterviewUserMemoryRecord],
) -> list[InterviewUserMemoryRecord]:
    by_key: dict[str, InterviewUserMemoryRecord] = {}
    for memory in memories:
        for key in _memory_keys(memory):
            existing = by_key.get(key)
            if existing is None or _is_newer_memory(memory, existing):
                by_key[key] = memory
    result: list[InterviewUserMemoryRecord] = []
    seen_ids: set[str] = set()
    for memory in sorted(
        by_key.values(),
        key=lambda item: (
            item.summary_generated_at,
            item.source_report_completed_at,
            item.created_at,
        ),
        reverse=True,
    ):
        if memory.id not in seen_ids:
            result.append(memory)
            seen_ids.add(memory.id)
    return result


def _memory_keys(memory: InterviewUserMemoryRecord) -> list[str]:
    explicit = _json_list(memory.embedding_json or "[]")
    candidates = [
        *explicit,
        *_json_list(memory.weakness_summary_json),
        *_json_list(memory.missing_points_json),
    ]
    keys = [_normalize_key(item) for item in candidates if item.strip()]
    return _dedupe(keys) or [memory.id]


def _is_newer_memory(
    candidate: InterviewUserMemoryRecord,
    existing: InterviewUserMemoryRecord,
) -> bool:
    return (
        candidate.summary_generated_at,
        candidate.source_report_completed_at,
        candidate.created_at,
    ) > (
        existing.summary_generated_at,
        existing.source_report_completed_at,
        existing.created_at,
    )


def _memory_score(memory: InterviewUserMemoryRecord, query_tokens: set[str]) -> int:
    memory_tokens = _tokens(
        "\n".join(
            [
                memory.target_role,
                memory.embedding_text,
                memory.weakness_summary_json,
                memory.missing_points_json,
                memory.improvement_advice_json,
                memory.reinforcement_question_hints_json,
            ]
        )
    )
    if not query_tokens or not memory_tokens:
        return 0
    return len(query_tokens & memory_tokens)


def _profile_state(
    profile: InterviewUserMemoryProfile | None,
) -> HistoricalInterviewMemoryProfileState:
    if not profile:
        return HistoricalInterviewMemoryProfileState()
    return HistoricalInterviewMemoryProfileState(
        stableWeaknesses=_json_list(profile.stable_weaknesses_json),
        improvedAreas=_json_list(profile.improved_areas_json),
        recurringMistakes=_json_list(profile.recurring_mistakes_json),
        updatedAt=profile.updated_at,
    )


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    return set(words + cjk)


def _run_lazy_backfill(user_id: str, lazy_backfill: Callable[[str], None] | None) -> None:
    if not lazy_backfill:
        return
    try:
        lazy_backfill(user_id)
    except Exception:
        return


def _json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, str)] if isinstance(parsed, list) else []


def _dedupe(values: object) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result


def _normalize_key(value: str) -> str:
    normalized = re.sub(r"[\s_]+", "-", value.strip().lower())
    normalized = re.sub(r"[^a-z0-9\-\u4e00-\u9fff]+", "", normalized)
    return normalized.strip("-")


def _memory_size(memory: HistoricalInterviewMemoryState) -> int:
    return len(memory.model_dump_json(exclude_none=True))
