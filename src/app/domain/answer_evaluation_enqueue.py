from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.domain.interview_state_machine import (
    get_active_node,
    get_active_round,
    get_current_question,
)
from app.integrations.redis_client import create_redis_answer_evaluation_store
from app.integrations.redis_evaluation_store import RedisAnswerEvaluationStore
from app.schemas.answer_evaluation import AnswerEvaluationTask
from app.schemas.interview_state import (
    AnswerAttemptState,
    InterviewSessionState,
    InterviewTopicNodeState,
)

logger = logging.getLogger(__name__)


def build_answer_evaluation_task(
    *,
    before_state: InterviewSessionState,
    after_state: InterviewSessionState,
    user_message: str,
    resource_id: str | None = None,
    now: Callable[[], str] | None = None,
    create_task_id: Callable[[AnswerAttemptState], str] | None = None,
) -> AnswerEvaluationTask | None:
    active_round = get_active_round(before_state)
    active_node = get_active_node(active_round)
    if not active_round or not active_node:
        return None

    answer_attempt = _find_new_answer_attempt(
        before_state=before_state,
        after_state=after_state,
        user_message=user_message,
    )
    if not answer_attempt:
        return None

    created_at = now() if now else _utc_now()
    current_question = get_current_question(before_state) or active_node.mainQuestion
    follow_up_question = (
        current_question if active_node.currentTargetType == "follow-up" else None
    )

    return AnswerEvaluationTask.model_validate(
        {
            "schemaVersion": 1,
            "taskId": (
                create_task_id(answer_attempt)
                if create_task_id
                else f"answer-evaluation-{answer_attempt.id}"
            ),
            "interviewId": before_state.threadId,
            "threadId": before_state.threadId,
            "resourceId": resource_id,
            "nodeId": active_node.id,
            "roundId": active_round.id,
            "roundType": active_round.type,
            "attemptId": answer_attempt.id,
            "targetType": answer_attempt.targetType,
            "targetId": answer_attempt.targetId,
            "targetRole": before_state.targetRole,
            "responseLanguage": before_state.responseLanguage,
            "question": current_question,
            "mainQuestion": active_node.mainQuestion,
            "followUpQuestion": follow_up_question,
            "referenceAnswer": active_node.referenceAnswer,
            "evaluationPoints": active_node.evaluationPoints or [],
            "candidateAnswer": user_message,
            "nodeConversation": _build_node_conversation(
                node=active_node,
                current_question=current_question,
                user_message=user_message,
                created_at=created_at,
            ),
            "createdAt": created_at,
        }
    )


def enqueue_answer_evaluation_task_best_effort(
    *,
    before_state: InterviewSessionState,
    after_state: InterviewSessionState,
    user_message: str,
    resource_id: str | None = None,
    store: RedisAnswerEvaluationStore | None = None,
    now: Callable[[], str] | None = None,
    create_task_id: Callable[[AnswerAttemptState], str] | None = None,
) -> AnswerEvaluationTask | None:
    task = build_answer_evaluation_task(
        before_state=before_state,
        after_state=after_state,
        user_message=user_message,
        resource_id=resource_id,
        now=now,
        create_task_id=create_task_id,
    )
    if not task:
        return None

    try:
        asyncio.run(_enqueue_task(task, store))
        logger.info(
            "Answer evaluation task enqueued",
            extra={
                "event": "answer_evaluation.task.enqueued",
                "interviewId": task.interviewId,
                "taskId": task.taskId,
                "attemptId": task.attemptId,
            },
        )
    except Exception as exc:
        logger.warning(
            "Failed to enqueue answer evaluation task",
            extra={
                "event": "answer_evaluation.task.enqueue_failed",
                "interviewId": task.interviewId,
                "taskId": task.taskId,
                "attemptId": task.attemptId,
                "err": str(exc),
            },
        )

    return task


async def _enqueue_task(
    task: AnswerEvaluationTask,
    store: RedisAnswerEvaluationStore | None,
) -> None:
    resolved_store = store or create_redis_answer_evaluation_store()
    await resolved_store.enqueue_task(task)
    client = getattr(resolved_store, "client", None)
    disconnect = getattr(client, "disconnect", None)
    if store is None and disconnect:
        await disconnect()


def _find_new_answer_attempt(
    *,
    before_state: InterviewSessionState,
    after_state: InterviewSessionState,
    user_message: str,
) -> AnswerAttemptState | None:
    previous_attempt_ids = {
        attempt.id
        for round_item in before_state.rounds
        for node in round_item.nodes
        for attempt in node.answerAttempts
    }
    normalized_user_message = user_message.strip()
    for round_item in after_state.rounds:
        for node in round_item.nodes:
            for attempt in node.answerAttempts:
                if (
                    attempt.id not in previous_attempt_ids
                    and attempt.userMessage.strip() == normalized_user_message
                    and attempt.score is not None
                    and not attempt.isDetour
                ):
                    return attempt
    return None


def _build_node_conversation(
    *,
    node: InterviewTopicNodeState,
    current_question: str,
    user_message: str,
    created_at: str,
) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = [
        {
            "role": "interviewer",
            "targetType": "main-question",
            "text": node.mainQuestion,
            "createdAt": created_at,
        }
    ]

    for attempt in node.answerAttempts:
        conversation.append(
            {
                "role": "candidate",
                "targetType": attempt.targetType,
                "text": attempt.userMessage,
                "createdAt": attempt.createdAt,
            }
        )

    if node.currentTargetType == "follow-up" and current_question.strip():
        conversation.append(
            {
                "role": "interviewer",
                "targetType": "follow-up",
                "text": current_question,
                "createdAt": created_at,
            }
        )

    conversation.append(
        {
            "role": "candidate",
            "targetType": node.currentTargetType,
            "text": user_message,
            "createdAt": created_at,
        }
    )
    return conversation


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
