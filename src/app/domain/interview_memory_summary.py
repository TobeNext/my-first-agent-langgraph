from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.integrations.models import ChatModelLike, create_chat_model
from app.schemas.interview_report import ReportGenerationOutput

MEMORY_SUMMARY_PROMPT_VERSION = "interview-memory-summary-v1"
WEAK_REVIEW_SCORE_THRESHOLD = 7.0

MEMORY_SUMMARY_SYSTEM_PROMPT = """You are maintaining long-term interview memory for future
mock interviews.
Use the report data to create a compact, structured memory summary.
Scores use a 1-10 scale. Only include weak areas when score < 7.0 or missingPoints is non-empty.
Do not copy full candidate answers.
Do not expose private narrative details unless they are necessary as technical evidence.
Return JSON only with: weaknessSummary, missingPoints, improvementAdvice,
reinforcementQuestionHints, normalizedWeaknessKeys, improvedAreas, embeddingText."""


class InterviewMemorySummaryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weaknessSummary: list[str] = Field(default_factory=list)
    missingPoints: list[str] = Field(default_factory=list)
    improvementAdvice: list[str] = Field(default_factory=list)
    reinforcementQuestionHints: list[str] = Field(default_factory=list)
    normalizedWeaknessKeys: list[str] = Field(default_factory=list)
    improvedAreas: list[str] = Field(default_factory=list)
    embeddingText: str = Field(min_length=1)


MemorySummaryEvaluator = Callable[
    [str],
    InterviewMemorySummaryOutput
    | dict[str, Any]
    | Awaitable[InterviewMemorySummaryOutput | dict[str, Any]],
]


def build_interview_memory_summary_prompt(
    *,
    report: ReportGenerationOutput,
    target_role: str,
) -> str:
    weak_reviews = _weak_reviews_from_report(report)
    payload = {
        "promptVersion": MEMORY_SUMMARY_PROMPT_VERSION,
        "targetRole": target_role,
        "reportSummary": report.summary.model_dump(mode="json"),
        "weakQuestionReviews": weak_reviews,
    }
    return "\n".join(
        [
            MEMORY_SUMMARY_SYSTEM_PROMPT,
            "",
            "Report memory summary input:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


async def generate_interview_memory_summary_with_model(
    *,
    report: ReportGenerationOutput,
    target_role: str,
    evaluator: MemorySummaryEvaluator | None = None,
    model: ChatModelLike | None = None,
) -> InterviewMemorySummaryOutput:
    prompt = build_interview_memory_summary_prompt(report=report, target_role=target_role)
    if evaluator:
        raw = await _maybe_await(evaluator(prompt))
        return InterviewMemorySummaryOutput.model_validate(raw)

    chat_model = model or create_chat_model()
    if hasattr(chat_model, "with_structured_output"):
        structured_model = chat_model.with_structured_output(InterviewMemorySummaryOutput)
        return InterviewMemorySummaryOutput.model_validate(structured_model.invoke(prompt))
    raw = chat_model.invoke(prompt)
    return InterviewMemorySummaryOutput.model_validate_json(str(raw))


def deterministic_interview_memory_summary(
    report: ReportGenerationOutput,
) -> InterviewMemorySummaryOutput:
    weak_reviews = _weak_reviews_from_report(report)
    missing_points = _dedupe(
        point for review in weak_reviews for point in review["missingPoints"]
    )
    improvement_advice = _dedupe(
        item for review in weak_reviews for item in review["improvementAdvice"]
    )
    weakness_summary = _dedupe(
        f"{review['question']}: {', '.join(review['missingPoints'])}"
        for review in weak_reviews
        if review["missingPoints"]
    )
    hints = _dedupe(
        f"Ask how the candidate would address: {point}" for point in missing_points
    )
    improved_areas = report.summary.strengths[:3]
    embedding_text = " ".join([*weakness_summary, *missing_points, *improvement_advice])
    return InterviewMemorySummaryOutput(
        weaknessSummary=weakness_summary or ["No reinforcement weakness identified."],
        missingPoints=missing_points,
        improvementAdvice=improvement_advice,
        reinforcementQuestionHints=hints,
        normalizedWeaknessKeys=[_normalize_key(item) for item in missing_points],
        improvedAreas=improved_areas,
        embeddingText=embedding_text or "No reinforcement weakness identified.",
    )


def _weak_reviews_from_report(report: ReportGenerationOutput) -> list[dict[str, Any]]:
    return [
        {
            "questionId": review.questionId,
            "targetType": review.targetType,
            "question": review.question,
            "score": review.score,
            "comment": review.comment,
            "missingPoints": review.missingPoints,
            "improvementAdvice": review.improvementAdvice,
        }
        for review in report.questionReviews
        if review.score < WEAK_REVIEW_SCORE_THRESHOLD or review.missingPoints
    ]


def _dedupe(values: object) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result


def _normalize_key(value: str) -> str:
    return "-".join(value.lower().split()) or "general-improvement"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
