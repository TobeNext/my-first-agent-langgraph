from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain.interview_state_machine import (
    AnswerEvaluationResult,
    get_active_node,
    get_active_round,
    get_current_question,
)
from app.integrations.llm_logging import log_llm_error, log_llm_input, log_llm_output
from app.integrations.models import ChatModelLike, create_chat_model
from app.schemas.interview_state import (
    InterviewRoundState,
    InterviewSessionState,
    InterviewTopicNodeState,
)


class FollowUpQuestionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    followUpQuestion: str | None = None


def ensure_generated_follow_up_question(
    *,
    state: InterviewSessionState,
    user_message: str,
    evaluation: AnswerEvaluationResult,
    model: ChatModelLike | None = None,
) -> AnswerEvaluationResult:
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    if not active_round or not active_node:
        return evaluation
    if not _should_generate_follow_up_question(evaluation, active_node):
        return evaluation

    question = generate_follow_up_question(
        state=state,
        active_round=active_round,
        active_node=active_node,
        current_question=get_current_question(state) or active_node.mainQuestion,
        user_message=user_message,
        evaluation=evaluation,
        model=model,
    )
    if not question:
        return evaluation
    return replace(evaluation, followUpQuestion=question)


def generate_follow_up_question(
    *,
    state: InterviewSessionState,
    active_round: InterviewRoundState,
    active_node: InterviewTopicNodeState,
    current_question: str,
    user_message: str,
    evaluation: AnswerEvaluationResult,
    model: ChatModelLike | None = None,
) -> str | None:
    prompt = build_dedicated_follow_up_question_prompt(
        state=state,
        active_round=active_round,
        active_node=active_node,
        current_question=current_question,
        user_message=user_message,
        evaluation=evaluation,
    )
    chat_model = model or create_chat_model()
    metadata = {
        "roundType": active_round.type,
        "nodeId": active_node.id,
        "nodeTopic": active_node.topic,
        "targetType": active_node.currentTargetType,
    }
    log_llm_input(
        thread_id=state.threadId,
        operation="follow-up-question-generation",
        prompt=prompt,
        metadata=metadata,
    )
    try:
        if hasattr(chat_model, "with_structured_output"):
            try:
                structured_model = chat_model.with_structured_output(FollowUpQuestionOutput)
                parsed = structured_model.invoke(prompt)
                question = _normalize_output(parsed)
                log_llm_output(
                    thread_id=state.threadId,
                    operation="follow-up-question-generation",
                    output=parsed,
                    metadata={**metadata, "normalizedQuestion": question},
                )
                return question
            except Exception as exc:
                log_llm_error(
                    thread_id=state.threadId,
                    operation="follow-up-question-generation",
                    error=exc,
                    metadata={**metadata, "stage": "structured-output"},
                )
        raw = chat_model.invoke(prompt)
        question = _parse_raw_output(raw)
        log_llm_output(
            thread_id=state.threadId,
            operation="follow-up-question-generation",
            output=raw,
            metadata={**metadata, "normalizedQuestion": question},
        )
        return question
    except Exception as exc:
        log_llm_error(
            thread_id=state.threadId,
            operation="follow-up-question-generation",
            error=exc,
            metadata=metadata,
        )
        return None


def build_dedicated_follow_up_question_prompt(
    *,
    state: InterviewSessionState,
    active_round: InterviewRoundState,
    active_node: InterviewTopicNodeState,
    current_question: str,
    user_message: str,
    evaluation: AnswerEvaluationResult,
) -> str:
    job_description = state.resumeContext.jobDescription.strip() or "not provided"
    return "\n".join(
        [
            "You are writing the next interviewer follow-up question for a mock interview.",
            "Return JSON only. Do not add markdown.",
            'Return exactly this shape: {"followUpQuestion":"..."}.',
            f"Interview language: {state.responseLanguage}",
            f"Target role: {state.targetRole}",
            f"Round type: {active_round.type}",
            f"Topic: {active_node.topic}",
            f"Current target type: {active_node.currentTargetType}",
            f"Current question: {current_question}",
            f"Main question: {active_node.mainQuestion}",
            f"Next follow-up index: {active_node.followUpCount + 1}",
            f"Job description context: {job_description}",
            "Current question dialogue record:",
            _build_node_conversation_record(active_node=active_node, user_message=user_message),
            f"Answer classification: {evaluation.classification}",
            f"Recommended intent: {evaluation.recommendedIntent}",
            f"Follow-up focus: {' | '.join(evaluation.followUpFocus) or active_node.topic}",
            f"Missing points: {' | '.join(evaluation.missingPoints) or 'none'}",
            f"Incorrect points: {' | '.join(evaluation.incorrectPoints) or 'none'}",
            "Write exactly one short interviewer question that stays on the same topic as "
            "the current question and the candidate answer.",
            "Deepen naturally. Do not jump to a much broader topic.",
            "Use this simple deepening pattern:",
            "- index 1: ask the candidate to explain the mentioned concept in more detail",
            "- index 2: ask for concrete use cases, implementation approach, or internal "
            "distinctions",
            "- index 3 or above: continue drilling into practical details, trade-offs, "
            "limitations, or edge cases that are still directly related",
            "Do not force system design, production pressure, rollback, metrics, or "
            "alternative comparisons unless the candidate already brought them up.",
            "Prefer asking about the specific concept the candidate actually mentioned, "
            "instead of repeating the full original question.",
        ]
    )


def _should_generate_follow_up_question(
    evaluation: AnswerEvaluationResult,
    active_node: InterviewTopicNodeState,
) -> bool:
    if evaluation.followUpQuestion and evaluation.followUpQuestion.strip():
        return False
    if active_node.followUpCount >= active_node.maxFollowUps:
        return False
    return evaluation.classification in {"direct-answer", "partial-answer", "deep-answer"}


def _build_node_conversation_record(
    *,
    active_node: InterviewTopicNodeState,
    user_message: str,
) -> str:
    lines = [f"Interviewer main question: {active_node.mainQuestion}"]
    answer_attempts_by_target_id = {
        attempt.targetId: attempt.userMessage for attempt in active_node.answerAttempts
    }
    main_answer = next(
        (
            attempt.userMessage
            for attempt in active_node.answerAttempts
            if attempt.targetType == "main-question"
        ),
        None,
    )
    if main_answer:
        lines.append(f"Candidate answer #1: {main_answer}")

    for follow_up in active_node.followUps:
        if follow_up.status == "pending" or not follow_up.question.strip():
            continue
        lines.append(f"Interviewer follow-up #{follow_up.index}: {follow_up.question}")
        linked_answer = (
            next(
                (
                    attempt.userMessage
                    for attempt in active_node.answerAttempts
                    if attempt.id == follow_up.linkedAnswerId
                ),
                None,
            )
            if follow_up.linkedAnswerId
            else answer_attempts_by_target_id.get(follow_up.id)
        )
        if linked_answer:
            lines.append(f"Candidate answer #{follow_up.index + 1}: {linked_answer}")

    last_recorded_answer = (
        active_node.answerAttempts[-1].userMessage.strip() if active_node.answerAttempts else None
    )
    if user_message.strip() != last_recorded_answer:
        lines.append(f"Candidate latest answer: {user_message}")

    return "\n".join(lines)


def _normalize_output(value: Any) -> str | None:
    if isinstance(value, FollowUpQuestionOutput):
        return _normalize_nullable_string(value.followUpQuestion)
    if isinstance(value, dict):
        return _normalize_nullable_string(value.get("followUpQuestion"))
    return _parse_raw_output(value)


def _parse_raw_output(value: Any) -> str | None:
    content = _extract_content(value)
    if not content:
        return None
    json_text = _extract_json_object_text(content)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _normalize_nullable_string(parsed.get("followUpQuestion"))


def _extract_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts)
    return None


def _extract_json_object_text(text: str) -> str | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed, flags=re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else trimmed
    start_index = candidate.find("{")
    end_index = candidate.rfind("}")
    if start_index < 0 or end_index <= start_index:
        return None
    return candidate[start_index : end_index + 1]


def _normalize_nullable_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
