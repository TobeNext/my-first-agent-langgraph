from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.follow_up_generation import ensure_generated_follow_up_question
from app.domain.interview_outcome import update_interview_outcome_snapshot
from app.domain.interview_state_machine import (
    apply_user_reply,
    build_rule_evaluation,
)
from app.domain.rag_recall_sample import update_rag_recall_sample_answer_performance
from app.schemas.interview_state import InterviewSessionState

REPORT_GENERATING_REPLY_ZH = "面试已结束，报告生成中。生成进度和最终报告可在右上角通知中查看。"
REPORT_GENERATING_REPLY_EN = (
    "The interview has ended and the report is being generated. "
    "You can check progress and download the final report from the notification bell."
)


def process_user_reply_node(state: Mapping[str, Any]) -> dict[str, Any]:
    session_payload = state.get("session")
    if not session_payload:
        raise ValueError("process_user_reply requires a checkpointed interview session.")

    session = InterviewSessionState.model_validate(session_payload)
    raw_user_message = str(state.get("raw_user_message") or "")
    stored_message, evaluation = build_rule_evaluation(
        raw_user_message,
        state=session,
    )
    evaluation = ensure_generated_follow_up_question(
        state=session,
        user_message=stored_message,
        evaluation=evaluation,
    )
    result = apply_user_reply(session, stored_message, evaluation)
    final_state = result.state
    assistant_reply = result.assistantReply
    if result.state.finalReportReady:
        final_state = _build_pending_final_report_state(result.state)
        assistant_reply = _build_report_generating_reply(final_state)

    update_answer_artifacts(state, final_state)

    return {
        "session": final_state.model_dump(),
        "assistant_reply": assistant_reply,
        "final_report_ready": final_state.finalReportReady,
    }


def _build_pending_final_report_state(state: InterviewSessionState) -> InterviewSessionState:
    return state.model_copy(
        update={
            "phase": "wrap-up",
            "activeRoundId": None,
            "finalReportReady": False,
            "finalReport": None,
        },
        deep=True,
    )


def _build_report_generating_reply(state: InterviewSessionState) -> str:
    if state.responseLanguage == "zh":
        return REPORT_GENERATING_REPLY_ZH
    return REPORT_GENERATING_REPLY_EN


def update_answer_artifacts(
    state: Mapping[str, Any],
    session: InterviewSessionState,
) -> None:
    outcome_file_path = state.get("outcome_file_path")
    if isinstance(outcome_file_path, str) and outcome_file_path:
        try:
            update_interview_outcome_snapshot(
                file_path=outcome_file_path,
                state=session,
                recall_traces=list(state.get("recall_traces") or []),
                generation_trace=list(state.get("generation_trace") or []),
            )
        except Exception:
            pass

    rag_sample_file_path = state.get("rag_recall_sample_file_path")
    if isinstance(rag_sample_file_path, str) and rag_sample_file_path:
        try:
            update_rag_recall_sample_answer_performance(rag_sample_file_path, session)
        except Exception:
            pass
