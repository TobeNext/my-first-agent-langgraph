from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from app.domain.answer_evaluation_enqueue import enqueue_answer_evaluation_task_best_effort
from app.domain.follow_up_generation import ensure_generated_follow_up_question
from app.domain.interview_outcome import update_interview_outcome_snapshot
from app.domain.interview_state_machine import (
    FLOW_TEST_SKIP_MARKER,
    apply_user_reply,
    build_rule_evaluation,
)
from app.domain.rag_recall_sample import update_rag_recall_sample_answer_performance
from app.domain.report_generation_enqueue import build_report_generation_task
from app.integrations.redis_client import (
    create_redis_answer_evaluation_store,
    create_redis_report_generation_store,
)
from app.integrations.redis_evaluation_store import RedisAnswerEvaluationStore
from app.integrations.redis_report_generation_store import RedisReportGenerationStore
from app.schemas.interview_state import InterviewSessionState

REPORT_EVALUATION_POLL_INTERVAL_SECONDS = 1.0
REPORT_EVALUATION_MAX_WAIT_SECONDS = 120.0
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
    is_flow_test_skip = _is_flow_test_skip(raw_user_message, session)
    if not is_flow_test_skip:
        enqueue_answer_evaluation_task_best_effort(
            before_state=session,
            after_state=result.state,
            user_message=stored_message,
            resource_id=str(state.get("resource_id") or "") or None,
        )
    final_state = result.state
    assistant_reply = result.assistantReply
    if result.state.finalReportReady and not is_flow_test_skip:
        finalization = complete_final_report_with_async_evaluations(
            result.state,
            resource_id=str(state.get("resource_id") or "") or None,
        )
        final_state = finalization["state"]
        assistant_reply = finalization["assistant_reply"]

    update_answer_artifacts(state, final_state)

    return {
        "session": final_state.model_dump(),
        "assistant_reply": assistant_reply,
        "final_report_ready": final_state.finalReportReady,
    }


def _is_flow_test_skip(raw_user_message: str, session: InterviewSessionState) -> bool:
    return (
        session.setup.settings.enableFlowTestMode
        and raw_user_message.strip() == FLOW_TEST_SKIP_MARKER
    )


def complete_final_report_with_async_evaluations(
    state: InterviewSessionState,
    *,
    store: RedisAnswerEvaluationStore | None = None,
    report_store: RedisReportGenerationStore | None = None,
    resource_id: str | None = None,
    poll_interval_seconds: float = REPORT_EVALUATION_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = REPORT_EVALUATION_MAX_WAIT_SECONDS,
) -> dict[str, Any]:
    try:
        return asyncio.run(
            _seal_and_enqueue_report_generation(
                state=state,
                store=store,
                report_store=report_store,
                resource_id=resource_id,
            )
        )
    except Exception:
        pending_state = _build_pending_final_report_state(state)
        return {
            "state": pending_state,
            "assistant_reply": _build_report_generating_reply(pending_state),
            "ready": False,
        }


async def _seal_and_enqueue_report_generation(
    *,
    state: InterviewSessionState,
    store: RedisAnswerEvaluationStore | None,
    report_store: RedisReportGenerationStore | None,
    resource_id: str | None,
) -> dict[str, Any]:
    expected_count = _count_expected_evaluation_attempts(state)
    if store is None:
        store = create_redis_answer_evaluation_store()
        should_disconnect = True
    else:
        should_disconnect = False

    try:
        manifest = await store.read_manifest(state.threadId)
        if manifest and len(manifest.expectedTaskIds) >= expected_count:
            await store.seal_interview(state.threadId)
        await _enqueue_report_generation_task(
            state=state,
            report_store=report_store,
            resource_id=resource_id,
        )
        pending_state = _build_pending_final_report_state(state)
        return {
            "state": pending_state,
            "assistant_reply": _build_report_generating_reply(pending_state),
            "ready": False,
        }
    finally:
        if should_disconnect:
            client = getattr(store, "client", None)
            disconnect = getattr(client, "disconnect", None)
            if disconnect:
                await disconnect()


async def _enqueue_report_generation_task(
    *,
    state: InterviewSessionState,
    report_store: RedisReportGenerationStore | None,
    resource_id: str | None,
) -> None:
    resolved_store = report_store or create_redis_report_generation_store()
    task = build_report_generation_task(state=state, resource_id=resource_id)
    await resolved_store.enqueue_task(task)
    client = getattr(resolved_store, "client", None)
    disconnect = getattr(client, "disconnect", None)
    if report_store is None and disconnect:
        await disconnect()


def _count_expected_evaluation_attempts(state: InterviewSessionState) -> int:
    return sum(
        1
        for round_item in state.rounds
        for node in round_item.nodes
        for attempt in node.answerAttempts
        if attempt.score is not None and not attempt.isDetour
    )


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
