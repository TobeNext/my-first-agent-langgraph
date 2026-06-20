from typing import Any

from app.graphs.nodes.report_generation import (
    evaluate_answers_node,
    generate_report_node,
    persist_report_node,
)
from app.integrations.report_repository import InterviewReportRepository
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


def _wrap_up_state_with_attempt() -> InterviewSessionState:
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
            "finalReportReady": False,
            "finalReport": None,
            "rounds": [round_item, state.rounds[1].model_copy(update={"status": "skipped"})],
        },
        deep=True,
    )


def _raw_answer_evaluation() -> dict[str, Any]:
    return {
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 8,
            "depth": 7,
            "specificity": 7,
            "clarity": 8,
        },
        "strengths": ["覆盖召回和重排。"],
        "missingPoints": ["还缺少失败降级。"],
        "incorrectPoints": [],
        "shouldAskFollowUp": False,
        "followUpFocus": [],
    }


def _report_output_payload() -> dict[str, Any]:
    return {
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


def _invalid_report_output_payload() -> dict[str, Any]:
    return {"report": "## 面试评估报告\n\n这是旧形状 markdown。"}


def test_report_generation_nodes_complete_inline_report_flow(tmp_path) -> None:
    state: dict[str, Any] = {
        "thread_id": "thread-1",
        "resource_id": "resource-1",
        "session": _wrap_up_state_with_attempt().model_dump(mode="json"),
    }
    seen_answer_prompts: list[str] = []
    seen_report_prompts: list[str] = []

    async def answer_evaluator(prompt: str, context) -> dict[str, Any]:
        seen_answer_prompts.append(prompt)
        assert context.attemptId == "attempt-1"
        return _raw_answer_evaluation()

    async def report_evaluator(prompt: str, context) -> dict[str, Any]:
        seen_report_prompts.append(prompt)
        assert context.interviewId == "thread-1"
        return _report_output_payload()

    state.update(evaluate_answers_node(state, evaluator=answer_evaluator))
    state.update(generate_report_node(state, evaluator=report_evaluator))

    repository = InterviewReportRepository(
        database_url=f"sqlite:///{tmp_path / 'reports.db'}"
    )
    state.update(persist_report_node(state, repository=repository))

    session = InterviewSessionState.model_validate(state["session"])
    stored = repository.get_report_by_interview_id("thread-1")
    output = ReportGenerationOutput.model_validate(state["report_output"])

    assert state["report_status"] == "succeeded"
    assert state["report_id"] == "report-thread-1"
    assert state["report_markdown_available"] is True
    assert session.phase == "completed"
    assert session.finalReportReady is True
    assert stored and stored.status == "succeeded"
    assert stored.markdown.startswith("## 面试评估报告")
    assert output.summary.overallScore == 8
    assert seen_answer_prompts and "Candidate answer:" in seen_answer_prompts[0]
    assert seen_report_prompts and "Question and answer context:" in seen_report_prompts[0]


def test_report_generation_agent_prompt_requires_new_schema() -> None:
    from app.domain.report_generation import _build_report_generation_system_prompt

    prompt = _build_report_generation_system_prompt()

    assert '"summary": {' in prompt
    assert '"questionReviews": [' in prompt
    assert '"markdown": "# 面试评估报告\\n\\n..."' in prompt
    assert "Do not return a top-level report field." in prompt


def test_generate_report_node_rejects_legacy_report_markdown_shape() -> None:
    state: dict[str, Any] = {
        "thread_id": "thread-1",
        "resource_id": "resource-1",
        "session": _wrap_up_state_with_attempt().model_dump(mode="json"),
    }

    async def answer_evaluator(_prompt: str, _context) -> dict[str, Any]:
        return _raw_answer_evaluation()

    async def report_evaluator(_prompt: str, _context) -> dict[str, Any]:
        return _invalid_report_output_payload()

    state.update(evaluate_answers_node(state, evaluator=answer_evaluator))
    state.update(generate_report_node(state, evaluator=report_evaluator))

    assert state["report_status"] == "failed"
    assert "ReportGenerationOutput" in state["report_error"]


def test_generate_report_node_preserves_failed_state() -> None:
    result = generate_report_node({"report_status": "failed", "report_error": "boom"})

    assert result == {}


def test_persist_report_node_writes_failed_report_for_failed_state(tmp_path) -> None:
    repository = InterviewReportRepository(
        database_url=f"sqlite:///{tmp_path / 'reports.db'}"
    )
    result = persist_report_node(
        {
            "resource_id": "resource-1",
            "session": _wrap_up_state_with_attempt().model_dump(mode="json"),
            "report_status": "failed",
            "report_error": "answer evaluation failed",
        },
        repository=repository,
    )

    stored = repository.get_report_by_interview_id("thread-1")

    assert result["report_status"] == "failed"
    assert result["report_id"] == "report-thread-1"
    assert result["report_markdown_available"] is False
    assert stored and stored.status == "failed"
    assert "answer evaluation failed" in stored.structured_json


def test_persist_report_node_reports_missing_output_as_failed() -> None:
    result = persist_report_node(
        {
            "session": _wrap_up_state_with_attempt().model_dump(mode="json"),
            "evaluation_contexts": [],
            "evaluation_results": [],
        }
    )

    assert result["report_status"] == "failed"
    assert result["report_error"]
