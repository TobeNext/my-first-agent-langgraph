from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from app.schemas.answer_evaluation import LlmAnswerEvaluationResult
from app.schemas.interview_snapshot import InterviewProgressSummary
from app.schemas.interview_state import (
    MAX_DETOUR_RESPONSES,
    AnswerAttemptState,
    AnswerClassification,
    AnswerScore,
    FollowUpIntent,
    InterviewRoundState,
    InterviewSessionState,
    InterviewTopicNodeState,
    ResponseLanguage,
    RoundType,
    TopicSummary,
)

FLOW_TEST_SKIP_MARKER = "[FLOW_TEST_SKIP]"
ACTIVE_ATTEMPT_LIMIT = 2
DETOUR_CLASSIFICATIONS: set[AnswerClassification] = {"off-topic", "meta-question"}


@dataclass(frozen=True)
class AnswerEvaluationResult:
    classification: AnswerClassification
    score: AnswerScore | None
    strengths: list[str]
    missingPoints: list[str]
    incorrectPoints: list[str]
    recommendedIntent: FollowUpIntent
    followUpFocus: list[str]
    followUpQuestion: str | None = None
    detourReply: str | None = None
    clarificationReply: str | None = None
    shouldCompleteNode: bool = False
    earlyCompletionReason: str | None = None


@dataclass(frozen=True)
class ProcessAnswerResult:
    state: InterviewSessionState
    assistantReply: str


def validate_interview_state(state: object) -> InterviewSessionState:
    return InterviewSessionState.model_validate(state)


def sanitize_answer_classification(value: str) -> AnswerClassification:
    allowed = {
        "direct-answer",
        "partial-answer",
        "deep-answer",
        "off-topic",
        "clarification-request",
        "skip-request",
        "stop-request",
        "meta-question",
    }
    return cast(AnswerClassification, value if value in allowed else "partial-answer")


def sanitize_follow_up_intent(value: str) -> FollowUpIntent:
    allowed = {"breadth", "depth", "accuracy", "experience"}
    return cast(FollowUpIntent, value if value in allowed else "depth")


def classify_by_rules(user_message: str) -> AnswerClassification | None:
    normalized = user_message.strip()

    if _is_strong_signal(
        normalized,
        [
            r"结束面试",
            r"结束吧",
            r"wrap up",
            r"finish (the )?interview",
            r"stop (the )?interview",
            r"give me (the )?(evaluation|report)",
        ],
    ):
        return "stop-request"

    if _is_strong_signal(
        normalized,
        [r"跳过", r"先过这一题", r"skip (this|the) question", r"next question", r"pass"],
    ):
        return "skip-request"

    if _is_strong_signal(
        normalized,
        [
            r"什么意思",
            r"解释一下",
            r"能详细说明题意吗",
            r"what do you mean",
            r"can you clarify",
            r"can you explain",
        ],
    ):
        return "clarification-request"

    if _is_strong_signal(
        normalized,
        [
            r"怎么评分",
            r"为什么问这个",
            r"流程是什么",
            r"how are you scoring",
            r"why are you asking",
        ],
    ):
        return "meta-question"

    return None


def build_interview_progress_summary(state: InterviewSessionState) -> InterviewProgressSummary:
    ordered_nodes = get_ordered_nodes(state)
    total_question_count = len(ordered_nodes)
    completed_question_count = len(
        [node for node in ordered_nodes if node.status in {"completed", "skipped"}]
    )
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    current_follow_up = None
    if active_node:
        current_follow_up = next(
            (item for item in active_node.followUps if item.id == active_node.currentFollowUpId),
            None,
        )

    return InterviewProgressSummary.model_validate(
        {
            "totalQuestionCount": total_question_count,
            "completedQuestionCount": completed_question_count,
            "remainingQuestionCount": max(0, total_question_count - completed_question_count),
            "currentQuestionIndex": (
                min(total_question_count, completed_question_count + 1) if active_node else None
            ),
            "currentRoundType": active_round.type if active_round else None,
            "currentRoundLabel": (
                get_round_label(active_round.type, state.responseLanguage) if active_round else None
            ),
            "currentStage": "completed" if not active_node else active_node.currentTargetType,
            "currentFollowUpIndex": current_follow_up.index if current_follow_up else None,
            "currentQuestionText": get_current_question(state),
            "currentNodeTopic": active_node.topic if active_node else None,
        }
    )


def apply_user_reply(
    state: InterviewSessionState,
    user_message: str,
    evaluation: AnswerEvaluationResult,
) -> ProcessAnswerResult:
    resumed_state = restore_interview_progress_if_needed(state)

    if resumed_state.phase == "completed" and resumed_state.finalReport:
        return ProcessAnswerResult(state=resumed_state, assistantReply=resumed_state.finalReport)

    if evaluation.classification == "stop-request":
        if count_remaining_question_nodes(resumed_state) > 0:
            return ProcessAnswerResult(
                state=resumed_state,
                assistantReply=build_pending_questions_guard_reply(resumed_state),
            )

        final_state = finalize_interview(
            resumed_state.model_copy(update={"phase": "wrap-up", "activeRoundId": None}, deep=True)
        )
        return ProcessAnswerResult(state=final_state, assistantReply=final_state.finalReport or "")

    active_round = get_active_round(resumed_state)
    active_node = get_active_node(active_round)

    if not active_round or not active_node:
        if count_remaining_question_nodes(resumed_state) > 0:
            return ProcessAnswerResult(
                state=resumed_state,
                assistantReply=build_pending_questions_guard_reply(resumed_state),
            )

        final_state = finalize_interview(
            resumed_state.model_copy(update={"phase": "wrap-up", "activeRoundId": None}, deep=True)
        )
        return ProcessAnswerResult(state=final_state, assistantReply=final_state.finalReport or "")

    target_id = (
        active_node.id
        if active_node.currentTargetType == "main-question"
        else active_node.currentFollowUpId or active_node.id
    )
    answer_attempt = AnswerAttemptState.model_validate(
        {
            "id": _create_id("answer-attempt"),
            "targetType": active_node.currentTargetType,
            "targetId": target_id,
            "userMessage": user_message,
            "classification": evaluation.classification,
            "score": evaluation.score.model_dump() if evaluation.score else None,
            "strengths": evaluation.strengths,
            "missingPoints": evaluation.missingPoints,
            "incorrectPoints": evaluation.incorrectPoints,
            "isDetour": evaluation.classification in DETOUR_CLASSIFICATIONS,
            "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    updated_node = active_node.model_copy(
        update={"answerAttempts": [*active_node.answerAttempts, answer_attempt]},
        deep=True,
    )

    if evaluation.classification == "clarification-request":
        updated_node = updated_node.model_copy(update={"detourResponseCount": 0}, deep=True)
        next_round = update_node(active_round, updated_node)
        next_state = update_round(resumed_state, next_round)
        return ProcessAnswerResult(
            state=next_state,
            assistantReply=evaluation.clarificationReply
            or build_clarification_reply(next_state, updated_node),
        )

    if evaluation.classification == "skip-request":
        updated_node = summarize_node(
            updated_node.model_copy(
                update={
                    "status": "skipped",
                    "currentFollowUpId": None,
                    "currentTargetType": "main-question",
                    "detourResponseCount": 0,
                },
                deep=True,
            ),
            resumed_state.responseLanguage,
        )
        skipped_round = update_node(active_round, compress_completed_node(updated_node))
        advanced_state = transition_after_node(
            resumed_state,
            skipped_round.model_copy(
                update={
                    "completedNodeCount": len(
                        [node for node in skipped_round.nodes if node.status == "completed"]
                    )
                },
                deep=True,
            ),
            updated_node.model_copy(update={"status": "completed"}, deep=True),
        )
        return _result_after_transition(advanced_state, None)

    if evaluation.classification in DETOUR_CLASSIFICATIONS:
        updated_node = updated_node.model_copy(
            update={
                "status": "detour-handling",
                "detourResponseCount": active_node.detourResponseCount + 1,
            },
            deep=True,
        )
        resumed_node = updated_node.model_copy(
            update={
                "status": (
                    "awaiting-main-answer"
                    if active_node.currentTargetType == "main-question"
                    else "awaiting-follow-up-answer"
                )
            },
            deep=True,
        )
        next_state = update_round(resumed_state, update_node(active_round, resumed_node))
        return ProcessAnswerResult(
            state=next_state,
            assistantReply=evaluation.detourReply or build_detour_reply(next_state, resumed_node),
        )

    if active_node.currentTargetType == "follow-up":
        updated_node = mark_follow_up_answered(updated_node, answer_attempt.id)

    if should_keep_following_up(active_round, updated_node, evaluation):
        next_node = apply_follow_up(
            updated_node,
            evaluation,
            resumed_state.responseLanguage,
            resumed_state.setup.settings.enableFlowTestMode,
        )
        next_state = update_round(resumed_state, update_node(active_round, next_node))
        next_state = append_asked_follow_up_memory(next_state, next_node)
        return ProcessAnswerResult(
            state=next_state,
            assistantReply=build_next_question_reply(next_state, None),
        )

    updated_node = summarize_node(
        updated_node.model_copy(
            update={
                "status": "completed",
                "earlyCompletionReason": evaluation.earlyCompletionReason,
                "currentFollowUpId": None,
                "currentTargetType": "main-question",
                "detourResponseCount": 0,
            },
            deep=True,
        ),
        resumed_state.responseLanguage,
    )
    correction_summary = (
        build_correction_summary(updated_node, resumed_state.responseLanguage)
        if resumed_state.setup.settings.reviewIncorrectOrMissingPoints
        else None
    )
    transitioned_state = transition_after_node(
        resumed_state.model_copy(update={"lastCorrectionSummary": correction_summary}, deep=True),
        active_round,
        updated_node,
    )
    return _result_after_transition(transitioned_state, correction_summary)


def build_rule_evaluation(
    user_message: str,
    *,
    state: InterviewSessionState | None = None,
) -> tuple[str, AnswerEvaluationResult]:
    stored_message = user_message
    if (
        user_message.strip() == FLOW_TEST_SKIP_MARKER
        and state
        and state.setup.settings.enableFlowTestMode
    ):
        stored_message = build_flow_test_mock_user_reply(state)

    classification = classify_by_rules(stored_message) or "direct-answer"
    score = (
        None
        if classification in {"clarification-request", "stop-request", "meta-question"}
        else _score(7)
    )
    should_complete = classification in {"skip-request", "direct-answer", "deep-answer"}

    return stored_message, AnswerEvaluationResult(
        classification=classification,
        score=score,
        strengths=["回答与当前题目相关"] if score else [],
        missingPoints=[] if should_complete else ["需要继续补充关键实现细节"],
        incorrectPoints=[],
        recommendedIntent="depth",
        followUpFocus=[_active_topic(state) or "当前问题"],
        shouldCompleteNode=should_complete,
    )


def get_active_round(state: InterviewSessionState) -> InterviewRoundState | None:
    return next(
        (round_item for round_item in state.rounds if round_item.id == state.activeRoundId),
        None,
    )


def get_active_node(round_item: InterviewRoundState | None) -> InterviewTopicNodeState | None:
    if not round_item:
        return None
    return next((node for node in round_item.nodes if node.id == round_item.activeNodeId), None)


def get_current_question(state: InterviewSessionState) -> str | None:
    node = get_active_node(get_active_round(state))
    if not node:
        return None
    if node.currentTargetType == "main-question":
        return node.mainQuestion
    follow_up = next((item for item in node.followUps if item.id == node.currentFollowUpId), None)
    return follow_up.question if follow_up else node.mainQuestion


def get_ordered_nodes(state: InterviewSessionState) -> list[InterviewTopicNodeState]:
    nodes: list[InterviewTopicNodeState] = []
    for round_item in state.rounds:
        for node_id in round_item.nodeOrder:
            node = next((item for item in round_item.nodes if item.id == node_id), None)
            if node:
                nodes.append(node)
    return nodes


def update_round(
    state: InterviewSessionState,
    updated_round: InterviewRoundState,
) -> InterviewSessionState:
    return state.model_copy(
        update={
            "rounds": [
                updated_round if round_item.id == updated_round.id else round_item
                for round_item in state.rounds
            ]
        },
        deep=True,
    )


def update_node(
    round_item: InterviewRoundState,
    updated_node: InterviewTopicNodeState,
) -> InterviewRoundState:
    return round_item.model_copy(
        update={
            "nodes": [
                updated_node if node.id == updated_node.id else node for node in round_item.nodes
            ]
        },
        deep=True,
    )


def move_to_next_node(round_item: InterviewRoundState) -> InterviewRoundState:
    try:
        current_index = round_item.nodeOrder.index(round_item.activeNodeId or "")
    except ValueError:
        current_index = -1

    next_node_id = None
    for node_id in round_item.nodeOrder[current_index + 1 :]:
        node = next((item for item in round_item.nodes if item.id == node_id), None)
        if node and node.status == "pending":
            next_node_id = node_id
            break

    if not next_node_id:
        return round_item.model_copy(
            update={"activeNodeId": None, "status": "completed"},
            deep=True,
        )

    return start_round(round_item.model_copy(update={"activeNodeId": next_node_id}, deep=True))


def start_node(node: InterviewTopicNodeState) -> InterviewTopicNodeState:
    return node.model_copy(
        update={
            "status": "awaiting-main-answer",
            "currentTargetType": "main-question",
            "currentFollowUpId": None,
            "detourResponseCount": 0,
        },
        deep=True,
    )


def start_round(round_item: InterviewRoundState) -> InterviewRoundState:
    active_node = get_active_node(round_item) or (round_item.nodes[0] if round_item.nodes else None)
    started_node = start_node(active_node) if active_node else None
    return round_item.model_copy(
        update={
            "status": "in-progress",
            "activeNodeId": started_node.id if started_node else None,
            "nodes": [
                started_node if started_node and node.id == started_node.id else node
                for node in round_item.nodes
            ],
        },
        deep=True,
    )


def get_round_label(round_type: RoundType, language: ResponseLanguage) -> str:
    if round_type == "professional-skills":
        return (
            "【第一轮：专业技能面试】"
            if language == "zh"
            else "[Round 1: Professional Skills Interview]"
        )
    return (
        "【第二轮：项目经历面试】"
        if language == "zh"
        else "[Round 2: Project Experience Interview]"
    )


def count_remaining_question_nodes(state: InterviewSessionState) -> int:
    return sum(1 for round_item in state.rounds for node in round_item.nodes if has_open_node(node))


def has_open_node(node: InterviewTopicNodeState) -> bool:
    return node.status not in {"completed", "skipped"}


def restore_interview_progress_if_needed(state: InterviewSessionState) -> InterviewSessionState:
    if get_active_round(state) and get_active_node(get_active_round(state)):
        return state

    resumable_round = next(
        (
            round_item
            for round_item in state.rounds
            if any(has_open_node(node) for node in round_item.nodes)
        ),
        None,
    )
    if not resumable_round:
        return state

    next_node_id = resumable_round.activeNodeId
    if not next_node_id:
        next_node_id = next(
            (
                node_id
                for node_id in resumable_round.nodeOrder
                if (
                    node := next(
                        (item for item in resumable_round.nodes if item.id == node_id),
                        None,
                    )
                )
                and has_open_node(node)
            ),
            None,
        )
    if not next_node_id:
        return state

    next_node = next((node for node in resumable_round.nodes if node.id == next_node_id), None)
    if not next_node:
        return state

    resumed_node = start_node(next_node) if next_node.status == "pending" else next_node
    resumed_round = resumable_round.model_copy(
        update={
            "status": "in-progress",
            "activeNodeId": next_node_id,
            "nodes": [
                resumed_node if node.id == resumed_node.id else node
                for node in resumable_round.nodes
            ],
        },
        deep=True,
    )
    return update_round(
        state.model_copy(
            update={
                "activeRoundId": resumed_round.id,
                "phase": (
                    "professional-skills-round"
                    if resumed_round.type == "professional-skills"
                    else "project-experience-round"
                ),
            },
            deep=True,
        ),
        resumed_round,
    )


def transition_after_node(
    state: InterviewSessionState,
    round_item: InterviewRoundState,
    completed_node: InterviewTopicNodeState,
) -> InterviewSessionState:
    compressed_node = compress_completed_node(
        summarize_node(completed_node, state.responseLanguage)
    )
    updated_round = update_node(round_item, compressed_node)
    updated_round = updated_round.model_copy(
        update={
            "completedNodeCount": len(
                [node for node in updated_round.nodes if node.status == "completed"]
            )
        },
        deep=True,
    )
    updated_round = move_to_next_node(updated_round)
    next_state = update_round(state, updated_round)

    if updated_round.status == "completed":
        if updated_round.type == "professional-skills":
            project_round = next(
                (
                    round_item
                    for round_item in next_state.rounds
                    if round_item.type == "project-experience"
                ),
                None,
            )
            if (
                not state.setup.settings.skipProjectExperienceRound
                and project_round
                and any(has_open_node(node) for node in project_round.nodes)
            ):
                started_project_round = start_round(
                    project_round.model_copy(
                        update={
                            "status": "pending",
                            "activeNodeId": project_round.activeNodeId
                            or next(
                                (
                                    node_id
                                    for node_id in project_round.nodeOrder
                                    if (
                                        node := next(
                                            (
                                                item
                                                for item in project_round.nodes
                                                if item.id == node_id
                                            ),
                                            None,
                                        )
                                    )
                                    and has_open_node(node)
                                ),
                                None,
                            ),
                        },
                        deep=True,
                    )
                )
                next_state = update_round(next_state, started_project_round)
                return next_state.model_copy(
                    update={
                        "phase": "project-experience-round",
                        "activeRoundId": started_project_round.id,
                    },
                    deep=True,
                )
        return finalize_interview_if_complete(next_state)

    return next_state


def should_keep_following_up(
    round_item: InterviewRoundState,
    node: InterviewTopicNodeState,
    evaluation: AnswerEvaluationResult,
) -> bool:
    if node.followUpCount >= node.maxFollowUps:
        return False

    guaranteed_follow_ups = 2 if round_item.type == "professional-skills" else 1
    if node.followUpCount < guaranteed_follow_ups:
        return True

    if not evaluation.score:
        return node.followUpCount < node.maxFollowUps

    has_open_gaps = bool(evaluation.missingPoints or evaluation.incorrectPoints)
    if not evaluation.shouldCompleteNode:
        return node.followUpCount < node.maxFollowUps

    if round_item.type == "professional-skills":
        return bool(
            node.followUpCount < node.maxFollowUps
            and (evaluation.score.weightedTotal < 8.2 or has_open_gaps)
        )

    return bool(
        node.followUpCount < node.maxFollowUps
        and evaluation.score.weightedTotal < 7.5
        and has_open_gaps
    )


def apply_follow_up(
    node: InterviewTopicNodeState,
    evaluation: AnswerEvaluationResult,
    language: ResponseLanguage,
    enable_flow_test_mode: bool,
) -> InterviewTopicNodeState:
    next_follow_up = next((item for item in node.followUps if item.status == "pending"), None)
    if not next_follow_up:
        return node

    question = build_follow_up_question(
        language=language,
        intent=evaluation.recommendedIntent,
        focus=evaluation.followUpFocus,
        node=node,
        follow_up_index=node.followUpCount + 1,
        enable_flow_test_mode=enable_flow_test_mode,
        generated_question=evaluation.followUpQuestion,
    )
    updated_follow_ups = [
        item.model_copy(
            update={
                "intent": evaluation.recommendedIntent,
                "question": question,
                "status": "asked",
            },
            deep=True,
        )
        if item.id == next_follow_up.id
        else item
        for item in node.followUps
    ]
    return node.model_copy(
        update={
            "status": "awaiting-follow-up-answer",
            "currentTargetType": "follow-up",
            "currentFollowUpId": next_follow_up.id,
            "followUpCount": node.followUpCount + 1,
            "detourResponseCount": 0,
            "followUps": updated_follow_ups,
        },
        deep=True,
    )


def mark_follow_up_answered(
    node: InterviewTopicNodeState,
    answer_attempt_id: str,
) -> InterviewTopicNodeState:
    return node.model_copy(
        update={
            "followUps": [
                follow_up.model_copy(
                    update={"status": "answered", "linkedAnswerId": answer_attempt_id},
                    deep=True,
                )
                if follow_up.id == node.currentFollowUpId
                else follow_up
                for follow_up in node.followUps
            ]
        },
        deep=True,
    )


def append_asked_follow_up_memory(
    state: InterviewSessionState,
    node: InterviewTopicNodeState,
) -> InterviewSessionState:
    follow_up = next(
        (item for item in node.followUps if item.id == node.currentFollowUpId),
        None,
    )
    question = follow_up.question.strip() if follow_up else ""
    if not question or question in state.followUpMemory.askedQuestions:
        return state
    return state.model_copy(
        update={
            "followUpMemory": state.followUpMemory.model_copy(
                update={
                    "askedQuestions": [*state.followUpMemory.askedQuestions, question],
                    "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
                deep=True,
            )
        },
        deep=True,
    )


def summarize_node(
    node: InterviewTopicNodeState,
    language: ResponseLanguage,
) -> InterviewTopicNodeState:
    scored_attempts = [attempt for attempt in node.answerAttempts if attempt.score is not None]
    aggregated_score = _average(
        [attempt.score.weightedTotal for attempt in scored_attempts if attempt.score]
    )
    strengths = _unique_by_normalized_text(
        [item for attempt in scored_attempts for item in attempt.strengths]
    )[:3]
    missing_points = _unique_by_normalized_text(
        [item for attempt in scored_attempts for item in attempt.missingPoints]
    )[:3]
    incorrect_points = _unique_by_normalized_text(
        [item for attempt in scored_attempts for item in attempt.incorrectPoints]
    )[:3]
    improvement_advice = [
        f"补充说明：{item}" if language == "zh" else f"Strengthen your answer on: {item}"
        for item in missing_points
    ]
    evidence = [
        attempt.userMessage[:180] for attempt in node.answerAttempts[-ACTIVE_ATTEMPT_LIMIT:]
    ]

    return node.model_copy(
        update={
            "aggregatedScore": aggregated_score,
            "summary": TopicSummary.model_validate(
                {
                    "strengths": strengths,
                    "weaknesses": incorrect_points,
                    "missingPoints": missing_points,
                    "improvementAdvice": improvement_advice,
                    "evidence": evidence,
                }
            ),
        },
        deep=True,
    )


def compress_completed_node(node: InterviewTopicNodeState) -> InterviewTopicNodeState:
    evidence = [
        attempt.userMessage[:180] for attempt in node.answerAttempts[-ACTIVE_ATTEMPT_LIMIT:]
    ]
    summary = node.summary or {
        "strengths": [],
        "weaknesses": [],
        "missingPoints": [],
        "improvementAdvice": [],
        "evidence": evidence,
    }
    summary_payload = summary.model_dump() if hasattr(summary, "model_dump") else summary
    return node.model_copy(
        update={
            "answerAttempts": node.answerAttempts[-ACTIVE_ATTEMPT_LIMIT:],
            "followUps": [
                follow_up for follow_up in node.followUps if follow_up.status != "pending"
            ],
            "summary": TopicSummary.model_validate({**summary_payload, "evidence": evidence}),
        },
        deep=True,
    )


def finalize_interview_if_complete(state: InterviewSessionState) -> InterviewSessionState:
    resumed_state = restore_interview_progress_if_needed(state)
    if count_remaining_question_nodes(resumed_state) > 0:
        return resumed_state
    return finalize_interview(
        resumed_state.model_copy(update={"phase": "wrap-up", "activeRoundId": None}, deep=True)
    )


def finalize_interview(state: InterviewSessionState) -> InterviewSessionState:
    completed_nodes = [
        node
        for round_item in state.rounds
        for node in round_item.nodes
        if node.status == "completed"
    ]
    return state.model_copy(
        update={
            "phase": "completed",
            "activeRoundId": None,
            "finalReportReady": True,
            "finalReport": render_interview_report_from_template(state, completed_nodes),
        },
        deep=True,
    )


def build_final_interview_state_from_evaluations(
    state: InterviewSessionState,
    evaluations: list[LlmAnswerEvaluationResult],
) -> InterviewSessionState:
    evaluation_by_attempt_id = {evaluation.attemptId: evaluation for evaluation in evaluations}
    rounds: list[InterviewRoundState] = []
    for round_item in state.rounds:
        nodes: list[InterviewTopicNodeState] = []
        for node in round_item.nodes:
            answer_attempts = []
            for attempt in node.answerAttempts:
                evaluation = evaluation_by_attempt_id.get(attempt.id)
                if not evaluation:
                    answer_attempts.append(attempt)
                    continue
                answer_attempts.append(
                    AnswerAttemptState.model_validate(
                        {
                            **attempt.model_dump(),
                            "classification": evaluation.classification,
                            "score": evaluation.score.model_dump(),
                            "strengths": evaluation.strengths,
                            "missingPoints": evaluation.missingPoints,
                            "incorrectPoints": evaluation.incorrectPoints,
                        }
                    )
                )
            next_node = node.model_copy(update={"answerAttempts": answer_attempts}, deep=True)
            nodes.append(
                summarize_node(next_node, state.responseLanguage)
                if next_node.status == "completed"
                else next_node
            )
        rounds.append(round_item.model_copy(update={"nodes": nodes}, deep=True))

    evaluated_state = state.model_copy(update={"rounds": rounds}, deep=True)
    completed_nodes = [
        node
        for round_item in evaluated_state.rounds
        for node in round_item.nodes
        if node.status == "completed"
    ]
    return evaluated_state.model_copy(
        update={
            "phase": "completed",
            "activeRoundId": None,
            "finalReportReady": True,
            "finalReport": render_interview_report_from_template(evaluated_state, completed_nodes),
        },
        deep=True,
    )


def build_pending_questions_guard_reply(state: InterviewSessionState) -> str:
    progress = build_interview_progress_summary(state)
    current_question = get_current_question(state)
    if state.responseLanguage == "zh":
        intro = (
            f"当前还有 {progress.remainingQuestionCount} 个问题未完成，"
            "我会在所有问题结束后再给出面试报告。我们先继续当前题目。"
        )
    else:
        intro = (
            f"There are still {progress.remainingQuestionCount} questions left. "
            "I will give you the interview report only after all questions are finished. "
            "Let's continue with the current question first."
        )
    return "\n\n".join([item for item in [intro, current_question] if item])


def build_next_question_reply(state: InterviewSessionState, correction_summary: str | None) -> str:
    active_round = get_active_round(state)
    current_question = get_current_question(state)
    parts = [correction_summary] if correction_summary else []

    if active_round:
        current_node = get_active_node(active_round)
        is_round_start = (
            current_node
            and current_node.currentTargetType == "main-question"
            and len(current_node.answerAttempts) == 0
        )
        if is_round_start:
            parts.append(get_round_label(active_round.type, state.responseLanguage))

    if current_question:
        parts.append(current_question)
    return "\n\n".join([item for item in parts if item])


def build_correction_summary(
    node: InterviewTopicNodeState,
    language: ResponseLanguage,
) -> str | None:
    if not node.summary:
        missing_points = []
        incorrect_points = []
    elif isinstance(node.summary, dict):
        missing_points = node.summary.get("missingPoints") or []
        incorrect_points = node.summary.get("weaknesses") or []
    else:
        missing_points = node.summary.missingPoints
        incorrect_points = node.summary.weaknesses
    if not missing_points and not incorrect_points:
        return None

    lines = ["回答纠正" if language == "zh" else "Answer Review"]
    for item in missing_points[:2]:
        lines.append(f"- 漏答点：{item}" if language == "zh" else f"- Missing point: {item}")
    for item in incorrect_points[:2]:
        lines.append(f"- 需要修正：{item}" if language == "zh" else f"- Needs correction: {item}")
    return "\n".join(lines)


def build_clarification_reply(state: InterviewSessionState, node: InterviewTopicNodeState) -> str:
    question = get_current_question(state) or node.mainQuestion
    if state.responseLanguage == "zh":
        return (
            f"这道题我主要想了解三点：第一，你是否理解“{node.topic}”背后的核心原理；"
            f"第二，你是否做过相关实践；第三，你能否说明关键取舍和风险。请继续围绕这个问题回答：\n{question}"
        )
    return (
        "For this question, I want to understand three things: "
        f"whether you know the core principles behind {node.topic}, "
        "whether you have practical experience, and whether you can explain "
        "the main trade-offs and risks. "
        f"Please continue with this question:\n{question}"
    )


def build_detour_reply(state: InterviewSessionState, node: InterviewTopicNodeState) -> str:
    current_question = get_current_question(state) or node.mainQuestion
    if node.detourResponseCount > MAX_DETOUR_RESPONSES:
        return (
            f"我先把当前问题收回来。请直接回答这道题：\n{current_question}"
            if state.responseLanguage == "zh"
            else (
                "Let me pull us back to the current question. "
                f"Please answer this question directly:\n{current_question}"
            )
        )
    return (
        f"我先简短回应到这里，但我们还在当前问题上。请继续回答：\n{current_question}"
        if state.responseLanguage == "zh"
        else (
            "I will keep the detour brief, but we still need to finish "
            f"the current question. Please continue here:\n{current_question}"
        )
    )


def build_follow_up_question(
    *,
    language: ResponseLanguage,
    intent: FollowUpIntent,
    focus: list[str],
    node: InterviewTopicNodeState,
    follow_up_index: int,
    enable_flow_test_mode: bool,
    generated_question: str | None,
) -> str:
    if generated_question and generated_question.strip() and not enable_flow_test_mode:
        return generated_question.strip()

    normalized_focus = next((item.strip() for item in focus if item.strip()), node.topic)
    if not enable_flow_test_mode:
        if language == "zh":
            if follow_up_index <= 1:
                return f"请详细说说你提到的“{normalized_focus}”，重点展开你对它的理解。"
            if follow_up_index == 2:
                return (
                    f"请继续围绕“{normalized_focus}”说明它的具体应用场景、"
                    "实现方式，或者其中几个关键区别。"
                )
            return (
                f"继续围绕“{normalized_focus}”往下讲，补充它在实际使用中的"
                "细节、限制、取舍或边界情况。"
            )
        if follow_up_index <= 1:
            return f"Please explain what you mentioned about {normalized_focus} in more detail."
        if follow_up_index == 2:
            return (
                f"Please continue with {normalized_focus} and explain the concrete use cases, "
                "implementation approach, or key distinctions inside it."
            )
        return (
            f"Please keep going on {normalized_focus} and add the practical details, "
            "limitations, trade-offs, or edge cases that matter in real use."
        )

    if language == "zh":
        directive = (
            "这次请直接按线上高压场景作答，明确给出失败案例、监控指标或阈值、"
            "降级或回滚方案，以及为什么不选其他方案。"
            if follow_up_index >= 3
            else (
                "这次请提升到系统设计层面，不要只停留在概念解释，"
                "要说清楚架构决策、异常路径和验证手段。"
            )
            if follow_up_index == 2
            else "不要停留在高层描述，请补充关键实现细节、判断依据或真实约束。"
        )
        if intent == "accuracy":
            return (
                f"你刚才提到了“{normalized_focus}”。请更准确地说明它的实现方式、"
                f"关键约束以及容易出错的地方。{directive}"
            )
        if intent == "experience":
            return (
                f"请结合你真实做过的项目，详细说明你是如何处理“{normalized_focus}”的，"
                f"最终效果如何？{directive}"
            )
        if intent == "breadth":
            return (
                f"除了你刚才提到的内容之外，在“{normalized_focus}”这个方向上，"
                f"你还会优先考虑哪些关键点？{directive}"
            )
        return (
            f"你刚才的回答里提到了“{normalized_focus}”。请继续往下展开，"
            f"重点说明背后的原理、取舍和边界条件。{directive}"
        )

    directive = (
        "Answer it as a production-pressure scenario: include a failure case, "
        "the metrics or thresholds you would watch, the fallback or rollback plan, "
        "and why you would reject the alternatives."
        if follow_up_index >= 3
        else (
            "Raise it to the system-design level this time. Do not stay at the concept level; "
            "explain the architecture decision, failure path, "
            "and how you would validate the design."
        )
        if follow_up_index == 2
        else (
            "Do not stay high level this time; add concrete implementation detail, "
            "evidence, or real constraints."
        )
    )
    if intent == "accuracy":
        return (
            f"You mentioned {normalized_focus}. Please explain it more precisely, "
            "including the implementation details, key constraints, and common failure points. "
            f"{directive}"
        )
    if intent == "experience":
        return (
            f"Use a real project example to explain how you handled {normalized_focus} "
            f"and what outcome you achieved. {directive}"
        )
    if intent == "breadth":
        return (
            "Beyond what you already mentioned, what other key considerations would you "
            f"include when dealing with {normalized_focus}? {directive}"
        )
    return (
        f"You brought up {normalized_focus}. Please go one level deeper and explain "
        f"the underlying principles, trade-offs, and edge cases. {directive}"
    )


def build_flow_test_mock_user_reply(state: InterviewSessionState) -> str:
    focus_label = _active_topic(state) or "当前问题"
    if state.responseLanguage == "zh":
        return f"针对“{focus_label}”，我会先说明核心思路，再补充实现细节、关键风险和项目中的取舍。"
    return (
        f"For {focus_label}, I would explain the core idea first, then add implementation "
        "detail, key risks, and real project trade-offs."
    )


def render_interview_report_from_template(
    state: InterviewSessionState,
    completed_nodes: list[InterviewTopicNodeState],
) -> str:
    overall_score = _average([node.aggregatedScore or 0 for node in completed_nodes]) or 0
    if state.responseLanguage == "zh":
        return "\n".join(
            [
                "## 模拟面试报告",
                "",
                f"**目标岗位**: {state.targetRole}",
                f"**完成题数**: {len(completed_nodes)}",
                f"**综合得分**: {overall_score:.1f}/10",
            ]
        ).strip()
    return "\n".join(
        [
            "## Mock Interview Report",
            "",
            f"**Target Role**: {state.targetRole}",
            f"**Questions Completed**: {len(completed_nodes)}",
            f"**Overall Score**: {overall_score:.1f}/10",
        ]
    ).strip()


def _result_after_transition(
    state: InterviewSessionState,
    correction_summary: str | None,
) -> ProcessAnswerResult:
    if state.finalReportReady:
        return ProcessAnswerResult(state=state, assistantReply=state.finalReport or "")
    return ProcessAnswerResult(
        state=state,
        assistantReply=build_next_question_reply(state, correction_summary),
    )


def _active_topic(state: InterviewSessionState | None) -> str | None:
    if not state:
        return None
    node = get_active_node(get_active_round(state))
    return node.topic if node else None


def _score(value: float) -> AnswerScore:
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


def _is_strong_signal(message: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)


def _create_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _clamp_score(value: float) -> float:
    return max(0, min(10, round(value, 2)))


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return _clamp_score(sum(values) / len(values))


def _unique_by_normalized_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
