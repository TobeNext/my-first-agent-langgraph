from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from app.domain.answer_evaluation_runtime import (
    AnswerEvaluationContext,
    AnswerEvaluationModelEvaluator,
    build_answer_evaluation_contexts_from_state,
    evaluate_answer_contexts,
)
from app.domain.report_generation_runtime import (
    ReportGenerationModelEvaluator,
    build_failed_report_write,
    build_report_write_from_output,
    generate_report_from_evaluations,
)
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.answer_evaluation import LlmAnswerEvaluationResult
from app.schemas.interview_report import ReportGenerationOutput
from app.schemas.interview_state import InterviewSessionState

REPORT_READY_REPLY_ZH = "面试评估报告已生成，可在右上角通知中下载。"
REPORT_READY_REPLY_EN = (
    "The interview report is ready. You can download it from the notification bell."
)


def evaluate_answers_node(
    state: Mapping[str, Any],
    *,
    evaluator: AnswerEvaluationModelEvaluator | None = None,
) -> dict[str, Any]:
    try:
        session = _session_from_state(state)
        contexts = build_answer_evaluation_contexts_from_state(
            session,
            resource_id=_resource_id_from_state(state),
        )
        results = asyncio.run(evaluate_answer_contexts(contexts, evaluator=evaluator))
        return {
            "evaluation_contexts": [context.model_dump(mode="json") for context in contexts],
            "evaluation_results": [result.model_dump(mode="json") for result in results],
            "report_status": "evaluated",
            "report_error": None,
        }
    except Exception as exc:
        return {"report_status": "failed", "report_error": str(exc)}


def generate_report_node(
    state: Mapping[str, Any],
    *,
    evaluator: ReportGenerationModelEvaluator | None = None,
) -> dict[str, Any]:
    if state.get("report_status") == "failed":
        return {}
    try:
        session = _session_from_state(state)
        contexts = _evaluation_contexts_from_state(state)
        results = _evaluation_results_from_state(state)
        output = asyncio.run(
            generate_report_from_evaluations(
                state=session,
                evaluation_contexts=contexts,
                evaluation_results=results,
                resource_id=_resource_id_from_state(state),
                evaluator=evaluator,
            )
        )
        return {
            "report_output": output.model_dump(mode="json"),
            "report_status": "generated",
            "report_error": None,
        }
    except Exception as exc:
        return {"report_status": "failed", "report_error": str(exc)}


def persist_report_node(
    state: Mapping[str, Any],
    *,
    repository: InterviewReportRepository | None = None,
) -> dict[str, Any]:
    if state.get("report_status") == "failed":
        try:
            session = _session_from_state(state)
            stored = (repository or InterviewReportRepository()).write_report(
                build_failed_report_write(
                    state=session,
                    error=str(state.get("report_error") or "Report generation failed."),
                    resource_id=_resource_id_from_state(state),
                )
            )
            return {
                "report_id": stored.id,
                "report_markdown_available": False,
                "report_status": "failed",
            }
        except Exception as exc:
            return {"report_status": "failed", "report_error": str(exc)}
    try:
        session = _session_from_state(state)
        contexts = _evaluation_contexts_from_state(state)
        results = _evaluation_results_from_state(state)
        output = ReportGenerationOutput.model_validate(state.get("report_output"))
        report = build_report_write_from_output(
            state=session,
            evaluation_contexts=contexts,
            evaluation_results=results,
            output=output,
            resource_id=_resource_id_from_state(state),
        )
        stored = (repository or InterviewReportRepository()).write_report(report)
        completed_session = session.model_copy(
            update={
                "phase": "completed",
                "activeRoundId": None,
                "finalReportReady": True,
                "finalReport": _report_ready_reply(session),
            },
            deep=True,
        )
        return {
            "session": completed_session.model_dump(mode="json"),
            "assistant_reply": completed_session.finalReport,
            "final_report_ready": True,
            "report_id": stored.id,
            "report_markdown_available": bool(stored.markdown),
            "report_status": "succeeded",
            "report_error": None,
        }
    except Exception as exc:
        return {"report_status": "failed", "report_error": str(exc)}


def _session_from_state(state: Mapping[str, Any]) -> InterviewSessionState:
    session_payload = state.get("session")
    if not session_payload:
        raise ValueError("report generation nodes require an interview session.")
    return InterviewSessionState.model_validate(session_payload)


def _resource_id_from_state(state: Mapping[str, Any]) -> str | None:
    value = state.get("resource_id")
    return str(value) if value else None


def _evaluation_contexts_from_state(state: Mapping[str, Any]) -> list[AnswerEvaluationContext]:
    return [
        AnswerEvaluationContext.model_validate(item)
        for item in list(state.get("evaluation_contexts") or [])
    ]


def _evaluation_results_from_state(state: Mapping[str, Any]) -> list[LlmAnswerEvaluationResult]:
    return [
        LlmAnswerEvaluationResult.model_validate(item)
        for item in list(state.get("evaluation_results") or [])
    ]


def _report_ready_reply(session: InterviewSessionState) -> str:
    return REPORT_READY_REPLY_ZH if session.responseLanguage == "zh" else REPORT_READY_REPLY_EN
