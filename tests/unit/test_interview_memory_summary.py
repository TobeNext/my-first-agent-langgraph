import json
from typing import Any

from app.config import Settings
from app.domain.interview_memory_summary import (
    MEMORY_SUMMARY_SYSTEM_PROMPT,
    build_interview_memory_summary_prompt,
    deterministic_interview_memory_summary,
    generate_interview_memory_summary_with_model,
)
from app.integrations.models import TracedChatModel
from app.schemas.interview_report import ReportGenerationOutput


def test_memory_summary_prompt_uses_only_weak_reviews_and_no_candidate_answer() -> None:
    report = ReportGenerationOutput.model_validate(_report_payload())

    prompt = build_interview_memory_summary_prompt(
        report=report,
        target_role="Backend Engineer",
    )

    assert MEMORY_SUMMARY_SYSTEM_PROMPT in prompt
    assert "Scores use a 1-10 scale" in prompt
    assert "缺少失败降级" in prompt
    assert "健康的高分回答" not in prompt
    assert "candidateAnswer" not in prompt
    assert "我会先做" not in prompt


def test_deterministic_memory_summary_filters_high_score_reviews_without_missing_points() -> None:
    report = ReportGenerationOutput.model_validate(_report_payload())

    summary = deterministic_interview_memory_summary(report)

    assert summary.missingPoints == ["缺少失败降级", "缺少指标阈值"]
    assert summary.improvementAdvice == ["补充降级策略", "说明告警阈值"]
    assert summary.reinforcementQuestionHints == [
        "Ask how the candidate would address: 缺少失败降级",
        "Ask how the candidate would address: 缺少指标阈值",
    ]
    assert "健康的高分回答" not in summary.embeddingText


async def test_generate_interview_memory_summary_with_fake_evaluator() -> None:
    report = ReportGenerationOutput.model_validate(_report_payload())
    seen_prompts: list[str] = []

    async def evaluator(prompt: str) -> dict[str, Any]:
        seen_prompts.append(prompt)
        assert "weakQuestionReviews" in prompt
        return {
            "weaknessSummary": ["失败降级覆盖不足"],
            "missingPoints": ["缺少失败降级"],
            "improvementAdvice": ["补充降级策略"],
            "reinforcementQuestionHints": ["追问失败时如何降级"],
            "normalizedWeaknessKeys": ["failure-degradation"],
            "improvedAreas": ["链路解释"],
            "embeddingText": "失败降级覆盖不足 缺少失败降级",
        }

    summary = await generate_interview_memory_summary_with_model(
        report=report,
        target_role="Backend Engineer",
        evaluator=evaluator,
    )

    assert seen_prompts
    assert summary.weaknessSummary == ["失败降级覆盖不足"]
    assert summary.embeddingText == "失败降级覆盖不足 缺少失败降级"


async def test_generate_interview_memory_summary_uses_raw_json_for_deepseek_reasoner() -> None:
    report = ReportGenerationOutput.model_validate(_report_payload())
    inner_model = DeepSeekMemorySummaryModel()

    summary = await generate_interview_memory_summary_with_model(
        report=report,
        target_role="Backend Engineer",
        model=TracedChatModel(
            model=inner_model,
            settings=_settings(
                MODEL_PROVIDER="deepseek",
                MODEL_NAME="deepseek-reasoner",
                MODEL_API_KEY="test-key",
            ),
        ),
    )

    assert summary.weaknessSummary == ["失败降级覆盖不足"]
    assert summary.embeddingText == "失败降级覆盖不足 缺少失败降级"
    assert inner_model.invoke_kwargs == {"response_format": {"type": "json_object"}}


class DeepSeekMemorySummaryModel:
    def __init__(self) -> None:
        self.invoke_kwargs: dict[str, object] | None = None

    def invoke(self, prompt: str, **kwargs: object) -> Any:
        self.invoke_kwargs = kwargs
        return _Message(
            json.dumps(
                {
                    "weaknessSummary": ["失败降级覆盖不足"],
                    "missingPoints": ["缺少失败降级"],
                    "improvementAdvice": ["补充降级策略"],
                    "reinforcementQuestionHints": ["追问失败时如何降级"],
                    "normalizedWeaknessKeys": ["failure-degradation"],
                    "improvedAreas": ["链路解释"],
                    "embeddingText": "失败降级覆盖不足 缺少失败降级",
                },
                ensure_ascii=False,
            )
        )

    def with_structured_output(self, schema: type[Any]) -> Any:
        raise AssertionError("DeepSeek should not use native structured output by default")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


def _settings(**overrides: object) -> Settings:
    values = {
        "APP_ENV": "test",
        "MODEL_PROVIDER": "mock",
        "MODEL_NAME": "mock/interview-runtime",
        "MODEL_BASE_URL": None,
        "MODEL_STRUCTURED_OUTPUT_MODE": "auto",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _report_payload() -> dict:
    return {
        "summary": {
            "overallScore": 7.5,
            "overallComment": "整体可用，但需要补充稳定性细节。",
            "strengths": ["链路解释"],
            "improvementPriorities": ["补充失败降级"],
        },
        "questionReviews": [
            {
                "questionId": "node-weak",
                "attemptId": "attempt-weak",
                "targetType": "main-question",
                "question": "如何处理 RAG 服务失败？",
                "score": 8,
                "comment": "回答缺少稳定性细节。",
                "missingPoints": ["缺少失败降级", "缺少指标阈值"],
                "improvementAdvice": ["补充降级策略", "说明告警阈值"],
            },
            {
                "questionId": "node-good",
                "attemptId": "attempt-good",
                "targetType": "main-question",
                "question": "健康的高分回答",
                "score": 9,
                "comment": "回答完整。",
                "missingPoints": [],
                "improvementAdvice": ["保持结构化表达"],
            },
        ],
        "markdown": "# 报告",
    }
