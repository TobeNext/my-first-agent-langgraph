from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from app.domain.answer_evaluation_enqueue import enqueue_answer_evaluation_task_best_effort
from app.domain.evaluation_report_reader import (
    WaitAndReadInterviewEvaluationsOutput,
    wait_and_read_interview_evaluations,
)
from app.domain.follow_up_generation import ensure_generated_follow_up_question
from app.domain.interview_outcome import update_interview_outcome_snapshot
from app.domain.interview_state_machine import (
    FLOW_TEST_SKIP_MARKER,
    apply_user_reply,
    build_final_interview_state_from_evaluations,
    build_rule_evaluation,
)
from app.domain.rag_recall_sample import update_rag_recall_sample_answer_performance
from app.integrations.redis_client import create_redis_answer_evaluation_store
from app.integrations.redis_evaluation_store import RedisAnswerEvaluationStore
from app.schemas.interview_state import InterviewSessionState

REPORT_EVALUATION_POLL_INTERVAL_SECONDS = 1.0
REPORT_EVALUATION_MAX_WAIT_SECONDS = 120.0


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
        finalization = complete_final_report_with_async_evaluations(result.state)
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
    poll_interval_seconds: float = REPORT_EVALUATION_POLL_INTERVAL_SECONDS,
    max_wait_seconds: float = REPORT_EVALUATION_MAX_WAIT_SECONDS,
) -> dict[str, Any]:
    try:
        return asyncio.run(
            _complete_final_report_with_async_evaluations(
                state=state,
                store=store,
                poll_interval_seconds=poll_interval_seconds,
                max_wait_seconds=max_wait_seconds,
            )
        )
    except Exception:
        pending_state = _build_pending_final_report_state(state)
        return {
            "state": pending_state,
            "assistant_reply": _build_evaluation_wait_blocked_reply(
                pending_state,
                WaitAndReadInterviewEvaluationsOutput.model_validate(
                    {
                        "ready": False,
                        "sealed": False,
                        "expectedCount": _count_expected_evaluation_attempts(state),
                        "completedCount": 0,
                        "failedCount": 0,
                        "evaluations": [],
                        "waitElapsedMs": 0,
                        "blockingReason": "manifest-missing",
                    }
                ),
            ),
            "ready": False,
        }


async def _complete_final_report_with_async_evaluations(
    *,
    state: InterviewSessionState,
    store: RedisAnswerEvaluationStore | None,
    poll_interval_seconds: float,
    max_wait_seconds: float,
) -> dict[str, Any]:
    expected_count = _count_expected_evaluation_attempts(state)
    if store is None:
        store = create_redis_answer_evaluation_store()
        should_disconnect = True
    else:
        should_disconnect = False

    try:
        manifest = await store.read_manifest(state.threadId)
        if not manifest:
            if expected_count == 0:
                return {"state": state, "assistant_reply": state.finalReport or "", "ready": True}
            pending_state = _build_pending_final_report_state(state)
            return {
                "state": pending_state,
                "assistant_reply": _build_evaluation_wait_blocked_reply(
                    pending_state,
                    WaitAndReadInterviewEvaluationsOutput.model_validate(
                        {
                            "ready": False,
                            "sealed": False,
                            "expectedCount": expected_count,
                            "completedCount": 0,
                            "failedCount": 0,
                            "evaluations": [],
                            "waitElapsedMs": 0,
                            "blockingReason": "manifest-missing",
                        }
                    ),
                ),
                "ready": False,
            }

        if len(manifest.expectedTaskIds) < expected_count:
            pending_state = _build_pending_final_report_state(state)
            return {
                "state": pending_state,
                "assistant_reply": _build_evaluation_wait_blocked_reply(
                    pending_state,
                    WaitAndReadInterviewEvaluationsOutput.model_validate(
                        {
                            "ready": False,
                            "sealed": manifest.sealed,
                            "expectedCount": expected_count,
                            "completedCount": len(manifest.completedTaskIds),
                            "failedCount": len(manifest.failedTaskIds),
                            "evaluations": [],
                            "waitElapsedMs": 0,
                            "blockingReason": "pending",
                        }
                    ),
                ),
                "ready": False,
            }

        await store.seal_interview(state.threadId)
        wait_result = await wait_and_read_interview_evaluations(
            interview_id=state.threadId,
            thread_id=state.threadId,
            store=store,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
        )
        if not wait_result.ready:
            pending_state = _build_pending_final_report_state(state)
            return {
                "state": pending_state,
                "assistant_reply": _build_evaluation_wait_blocked_reply(
                    pending_state,
                    wait_result,
                ),
                "ready": False,
            }

        final_state = build_final_interview_state_from_evaluations(
            state,
            wait_result.evaluations,
        )
        return {
            "state": final_state,
            "assistant_reply": final_state.finalReport or "",
            "ready": True,
        }
    finally:
        if should_disconnect:
            client = getattr(store, "client", None)
            disconnect = getattr(client, "disconnect", None)
            if disconnect:
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


def _build_evaluation_wait_blocked_reply(
    state: InterviewSessionState,
    wait_result: WaitAndReadInterviewEvaluationsOutput,
) -> str:
    if state.responseLanguage == "zh":
        if wait_result.blockingReason == "failed":
            return (
                f"面试题目已经完成，但异步评分中有 {wait_result.failedCount} "
                "个任务失败，暂时不能生成最终报告。请稍后重试或让系统重新处理失败任务。"
            )
        return (
            "面试题目已经完成，我正在等待异步评分完成后生成最终报告。"
            f"当前进度：{wait_result.completedCount}/{wait_result.expectedCount}。"
            "请稍后再发送一条消息获取报告。"
        )
    if wait_result.blockingReason == "failed":
        return (
            "The interview questions are complete, but "
            f"{wait_result.failedCount} async evaluation task(s) failed, so I cannot "
            "generate the final report yet. Please retry after the failed task is reprocessed."
        )
    return (
        "The interview questions are complete. I am waiting for async evaluations before "
        f"generating the final report. Current progress: "
        f"{wait_result.completedCount}/{wait_result.expectedCount}. "
        "Please send another message shortly to fetch the report."
    )


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
