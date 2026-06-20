from typing import Any

from app.domain.answer_evaluation_runtime import (
    build_answer_evaluation_contexts_from_state,
)
from app.domain.report_generation_runtime import (
    build_report_prompt_from_session,
    build_report_write_from_output,
    generate_report_from_evaluations,
)
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.answer_evaluation import LlmAnswerEvaluationResult
from app.schemas.interview_report import ReportGenerationOutput
from app.schemas.interview_state import AnswerAttemptState, AnswerScore, InterviewSessionState
from tests.unit.test_interview_state_machine import _state_fixture

NOW = "2026-06-19T00:00:00.000Z"


def _score(value: float = 8) -> AnswerScore:
    return AnswerScore.model_validate(
        {
            "relevance": value,
            "accuracy": value,
            "depth": value,
            "specificity": value,
            "clarity": value,
            "weightedTotal": value,
        }
    )


def _state_with_attempt() -> InterviewSessionState:
    state = _state_fixture(flow_test=False)
    attempt = AnswerAttemptState.model_validate(
        {
            "id": "attempt-1",
            "targetType": "main-question",
            "targetId": "node-rag",
            "userMessage": "我会先做 query rewrite，再召回 topK，最后重排并生成答案。",
            "classification": "direct-answer",
            "score": _score().model_dump(),
            "strengths": ["结构清晰"],
            "missingPoints": [],
            "incorrectPoints": [],
            "isDetour": False,
            "createdAt": NOW,
        }
    )
    node = state.rounds[0].nodes[0].model_copy(
        update={
            "referenceAnswer": "覆盖 query rewrite、召回、重排和生成。",
            "evaluationPoints": ["说明 query rewrite", "说明重排"],
            "answerAttempts": [attempt],
            "status": "completed",
        },
        deep=True,
    )
    round_item = state.rounds[0].model_copy(
        update={"nodes": [node], "status": "completed", "completedNodeCount": 1},
        deep=True,
    )
    return state.model_copy(
        update={
            "phase": "wrap-up",
            "activeRoundId": None,
            "rounds": [round_item, state.rounds[1].model_copy(update={"status": "skipped"})],
        },
        deep=True,
    )


def _evaluation_result() -> LlmAnswerEvaluationResult:
    return LlmAnswerEvaluationResult.model_validate(
        {
            "schemaVersion": 1,
            "taskId": "answer-evaluation-attempt-1",
            "interviewId": "thread-1",
            "threadId": "thread-1",
            "nodeId": "node-rag",
            "roundId": "round-professional",
            "roundType": "professional-skills",
            "attemptId": "attempt-1",
            "classification": "direct-answer",
            "score": {
                "relevance": 8,
                "accuracy": 8,
                "depth": 7,
                "specificity": 7,
                "clarity": 8,
                "weightedTotal": 7.65,
            },
            "strengths": ["覆盖召回和重排。"],
            "missingPoints": ["还缺少失败降级。"],
            "incorrectPoints": [],
            "shouldAskFollowUp": False,
            "followUpFocus": [],
            "evaluatorModel": "fake-evaluator",
            "promptVersion": "answer-evaluation-v1",
            "createdAt": NOW,
        }
    )


def _report_output_payload(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "summary": {
            "overallScore": 8,
            "overallComment": "候选人理解核心流程，但需要补充边界场景。",
            "strengths": ["能说明核心链路。"],
            "improvementPriorities": ["补充失败降级和观测指标。"],
        },
        "questionReviews": [
            {
                "questionId": "node-rag",
                "attemptId": "attempt-1",
                "targetType": "main-question",
                "question": "请解释你的 RAG 链路。",
                "score": 8,
                "comment": "回答覆盖 query rewrite、召回和重排。",
                "missingPoints": ["还缺少失败降级。"],
                "improvementAdvice": ["补充失败降级和监控指标。"],
            }
        ],
        "markdown": "## 面试评估报告\n\n候选人理解核心流程。",
    }
    payload.update(overrides or {})
    return payload


def test_build_report_prompt_from_session_uses_evaluations_and_answer_context() -> None:
    state = _state_with_attempt()
    contexts = build_answer_evaluation_contexts_from_state(state, resource_id="resource-1")

    prompt = build_report_prompt_from_session(
        state=state,
        evaluation_contexts=contexts,
        evaluation_results=[_evaluation_result()],
        resource_id="resource-1",
        now=lambda: NOW,
        model_name="fake-model",
    )

    assert "Question and answer context:" in prompt
    assert "我会先做 query rewrite" in prompt
    assert "还缺少失败降级" in prompt
    assert "langgraph-inline" in prompt


async def test_generate_report_from_evaluations_uses_fake_evaluator() -> None:
    state = _state_with_attempt()
    contexts = build_answer_evaluation_contexts_from_state(state)
    seen_prompts: list[str] = []

    async def evaluator(prompt: str, context) -> dict[str, Any]:
        seen_prompts.append(prompt)
        assert context.interviewId == "thread-1"
        return _report_output_payload()

    output = await generate_report_from_evaluations(
        state=state,
        evaluation_contexts=contexts,
        evaluation_results=[_evaluation_result()],
        evaluator=evaluator,
        now=lambda: NOW,
    )

    assert isinstance(output, ReportGenerationOutput)
    assert output.summary.overallScore == 8
    assert output.questionReviews[0].attemptId == "attempt-1"
    assert seen_prompts and "Evaluation results:" in seen_prompts[0]


def test_build_report_write_from_output_can_be_persisted(tmp_path) -> None:
    state = _state_with_attempt()
    contexts = build_answer_evaluation_contexts_from_state(state)
    output = ReportGenerationOutput.model_validate(_report_output_payload())

    report = build_report_write_from_output(
        state=state,
        evaluation_contexts=contexts,
        evaluation_results=[_evaluation_result()],
        output=output,
        now=lambda: NOW,
        model_name="fake-model",
    )
    repository = InterviewReportRepository(
        database_url=f"sqlite:///{tmp_path / 'reports.db'}"
    )

    stored = repository.write_report(report)
    items = repository.list_items(stored.id)

    assert stored.id == "report-thread-1"
    assert stored.status == "succeeded"
    assert stored.overall_score == 8
    assert stored.markdown.startswith("## 面试评估报告")
    assert items[0].task_id == "answer-evaluation-attempt-1"
    assert items[0].candidate_answer.startswith("我会先做 query rewrite")
