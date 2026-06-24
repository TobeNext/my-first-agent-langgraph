from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.integrations.llm_logging import log_llm_error, log_llm_input, log_llm_output
from app.integrations.models import (
    ChatModelLike,
    create_chat_model,
    invoke_json_output_model,
    should_use_native_structured_output,
)
from app.schemas.answer_evaluation import (
    AnswerEvaluationConversationItem,
    LlmAnswerEvaluationResult,
)
from app.schemas.interview_state import (
    AnswerClassification,
    AnswerScore,
    InterviewSessionState,
    InterviewTopicNodeState,
    ResponseLanguage,
    RoundType,
)

ANSWER_EVALUATION_PROMPT_VERSION = "answer-evaluation-v1"


class RawAnswerScore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relevance: float
    accuracy: float
    depth: float
    specificity: float
    clarity: float


class RawAnswerEvaluationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    classification: AnswerClassification
    score: RawAnswerScore
    strengths: list[str]
    missingPoints: list[str]
    incorrectPoints: list[str]
    shouldAskFollowUp: bool
    followUpFocus: list[str]


class AnswerEvaluationContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schemaVersion: int = 1
    evaluationId: str = Field(min_length=1)
    interviewId: str = Field(min_length=1)
    threadId: str = Field(min_length=1)
    resourceId: str | None = None
    nodeId: str = Field(min_length=1)
    roundId: str = Field(min_length=1)
    roundType: RoundType
    attemptId: str = Field(min_length=1)
    targetType: str = Field(min_length=1)
    targetId: str = Field(min_length=1)
    targetRole: str = Field(min_length=1)
    responseLanguage: ResponseLanguage
    question: str = Field(min_length=1)
    mainQuestion: str = Field(min_length=1)
    followUpQuestion: str | None = None
    referenceAnswer: str | None = None
    evaluationPoints: list[str] = Field(default_factory=list)
    candidateAnswer: str = Field(min_length=1)
    nodeConversation: list[AnswerEvaluationConversationItem] = Field(default_factory=list)
    createdAt: str


class StructuredChatModelLike(Protocol):
    def with_structured_output(self, schema: type[Any]) -> ChatModelLike: ...


AnswerEvaluationModelEvaluator = Callable[
    [str, AnswerEvaluationContext],
    RawAnswerEvaluationOutput
    | dict[str, Any]
    | Awaitable[RawAnswerEvaluationOutput | dict[str, Any]],
]


def build_answer_evaluation_contexts_from_state(
    state: InterviewSessionState,
    resource_id: str | None = None,
) -> list[AnswerEvaluationContext]:
    contexts: list[AnswerEvaluationContext] = []
    for round_item in state.rounds:
        for node_id in round_item.nodeOrder:
            node = next((item for item in round_item.nodes if item.id == node_id), None)
            if not node:
                continue
            conversation = _build_node_conversation(node)
            for attempt in node.answerAttempts:
                if attempt.isDetour or attempt.score is None or not attempt.userMessage.strip():
                    continue
                follow_up_question = _follow_up_question(node, attempt.targetId)
                question = (
                    node.mainQuestion
                    if attempt.targetType == "main-question"
                    else follow_up_question
                )
                contexts.append(
                    AnswerEvaluationContext.model_validate(
                        {
                            "schemaVersion": 1,
                            "evaluationId": f"answer-evaluation-{attempt.id}",
                            "interviewId": state.threadId,
                            "threadId": state.threadId,
                            "resourceId": resource_id,
                            "nodeId": node.id,
                            "roundId": round_item.id,
                            "roundType": round_item.type,
                            "attemptId": attempt.id,
                            "targetType": attempt.targetType,
                            "targetId": attempt.targetId,
                            "targetRole": state.targetRole,
                            "responseLanguage": state.responseLanguage,
                            "question": question or node.mainQuestion,
                            "mainQuestion": node.mainQuestion,
                            "followUpQuestion": follow_up_question,
                            "referenceAnswer": node.referenceAnswer,
                            "evaluationPoints": node.evaluationPoints or [],
                            "candidateAnswer": attempt.userMessage,
                            "nodeConversation": _conversation_through_attempt(
                                conversation,
                                attempt.id,
                            ),
                            "createdAt": attempt.createdAt,
                        }
                    )
                )
    return contexts


async def evaluate_answer_context(
    context: AnswerEvaluationContext,
    *,
    evaluator: AnswerEvaluationModelEvaluator | None = None,
    model: ChatModelLike | StructuredChatModelLike | None = None,
    now: Callable[[], str] | None = None,
    evaluator_model: str | None = None,
    prompt_version: str = ANSWER_EVALUATION_PROMPT_VERSION,
) -> LlmAnswerEvaluationResult:
    settings = get_settings()
    prompt = build_answer_evaluation_context_prompt(context)
    raw_evaluation = (
        await _maybe_await(evaluator(prompt, context))
        if evaluator
        else await evaluate_answer_context_with_model(prompt, context, model=model)
    )
    return build_llm_answer_evaluation_result(
        context=context,
        raw_evaluation=raw_evaluation,
        now=now() if now else _utc_now(),
        evaluator_model=evaluator_model or settings.model_name,
        prompt_version=prompt_version,
    )


async def evaluate_answer_contexts(
    contexts: list[AnswerEvaluationContext],
    *,
    evaluator: AnswerEvaluationModelEvaluator | None = None,
    model: ChatModelLike | StructuredChatModelLike | None = None,
    now: Callable[[], str] | None = None,
    evaluator_model: str | None = None,
    prompt_version: str = ANSWER_EVALUATION_PROMPT_VERSION,
) -> list[LlmAnswerEvaluationResult]:
    results: list[LlmAnswerEvaluationResult] = []
    for context in contexts:
        results.append(
            await evaluate_answer_context(
                context,
                evaluator=evaluator,
                model=model,
                now=now,
                evaluator_model=evaluator_model,
                prompt_version=prompt_version,
            )
        )
    return results


def build_answer_evaluation_context_prompt(context: AnswerEvaluationContext) -> str:
    reference_answer = context.referenceAnswer.strip() if context.referenceAnswer else "(none)"
    node_conversation = json.dumps(
        [item.model_dump() for item in context.nodeConversation],
        ensure_ascii=False,
        indent=2,
    )
    return "\n\n".join(
        [
            f"Target role:\n{context.targetRole}",
            f"Round type:\n{context.roundType}",
            f"Question:\n{context.question}",
            f"Main question:\n{context.mainQuestion}",
            f"Reference answer:\n{reference_answer}",
            f"Reference answer points:\n{_format_lines(context.evaluationPoints)}",
            f"Candidate answer:\n{context.candidateAnswer}",
            f"Node conversation:\n{node_conversation}",
        ]
    )


async def evaluate_answer_context_with_model(
    prompt: str,
    context: AnswerEvaluationContext,
    *,
    model: ChatModelLike | StructuredChatModelLike | None = None,
) -> RawAnswerEvaluationOutput:
    chat_model = model or create_chat_model()
    if _is_mock_chat_model(chat_model):
        return _build_mock_answer_evaluation(context)
    evaluator_prompt = _build_evaluator_prompt(prompt)
    metadata = _log_metadata(context)
    log_llm_input(
        thread_id=context.threadId,
        operation="answer-evaluation",
        prompt=evaluator_prompt,
        metadata=metadata,
    )
    try:
        if should_use_native_structured_output(chat_model):
            try:
                structured_model = chat_model.with_structured_output(RawAnswerEvaluationOutput)
                result = structured_model.invoke(evaluator_prompt)
            except Exception as exc:
                log_llm_error(
                    thread_id=context.threadId,
                    operation="answer-evaluation",
                    error=exc,
                    metadata={**metadata, "stage": "structured-output"},
                )
                result = _parse_raw_model_json(
                    invoke_json_output_model(chat_model, evaluator_prompt)
                )
        else:
            result = _parse_raw_model_json(invoke_json_output_model(chat_model, evaluator_prompt))
        parsed = RawAnswerEvaluationOutput.model_validate(result)
        log_llm_output(
            thread_id=context.threadId,
            operation="answer-evaluation",
            output=parsed,
            metadata=metadata,
        )
        return parsed
    except Exception as exc:
        log_llm_error(
            thread_id=context.threadId,
            operation="answer-evaluation",
            error=exc,
            metadata=metadata,
        )
        raise


def build_llm_answer_evaluation_result(
    *,
    context: AnswerEvaluationContext,
    raw_evaluation: RawAnswerEvaluationOutput | dict[str, Any],
    now: str,
    evaluator_model: str,
    prompt_version: str,
) -> LlmAnswerEvaluationResult:
    raw = RawAnswerEvaluationOutput.model_validate(raw_evaluation)
    return LlmAnswerEvaluationResult.model_validate(
        {
            "schemaVersion": 1,
            "taskId": context.evaluationId,
            "interviewId": context.interviewId,
            "threadId": context.threadId,
            "nodeId": context.nodeId,
            "roundId": context.roundId,
            "roundType": context.roundType,
            "attemptId": context.attemptId,
            "classification": raw.classification,
            "score": AnswerScore.model_validate(
                {
                    **raw.score.model_dump(),
                    "weightedTotal": calculate_answer_weighted_total(raw.score),
                }
            ),
            "strengths": raw.strengths,
            "missingPoints": raw.missingPoints,
            "incorrectPoints": raw.incorrectPoints,
            "shouldAskFollowUp": raw.shouldAskFollowUp,
            "followUpFocus": raw.followUpFocus,
            "evaluatorModel": evaluator_model,
            "promptVersion": prompt_version,
            "createdAt": now,
        }
    )


def calculate_answer_weighted_total(score: RawAnswerScore | dict[str, float]) -> float:
    values = score.model_dump() if isinstance(score, RawAnswerScore) else score
    return round(
        values["relevance"] * 0.25
        + values["accuracy"] * 0.25
        + values["depth"] * 0.25
        + values["specificity"] * 0.15
        + values["clarity"] * 0.1,
        2,
    )


def _build_node_conversation(node: InterviewTopicNodeState) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = [
        {
            "role": "interviewer",
            "targetType": "main-question",
            "text": node.mainQuestion,
            "createdAt": node.answerAttempts[0].createdAt if node.answerAttempts else "",
        }
    ]
    for attempt in node.answerAttempts:
        follow_up_question = _follow_up_question(node, attempt.targetId)
        if attempt.targetType == "follow-up" and follow_up_question:
            conversation.append(
                {
                    "role": "interviewer",
                    "targetType": "follow-up",
                    "text": follow_up_question,
                    "createdAt": attempt.createdAt,
                }
            )
        conversation.append(
            {
                "role": "candidate",
                "targetType": attempt.targetType,
                "text": attempt.userMessage,
                "createdAt": attempt.createdAt,
                "attemptId": attempt.id,
            }
        )
    return conversation


def _conversation_through_attempt(
    conversation: list[dict[str, str]],
    attempt_id: str,
) -> list[AnswerEvaluationConversationItem]:
    selected: list[dict[str, str]] = []
    for item in conversation:
        selected.append(item)
        if item.get("attemptId") == attempt_id:
            break
    return [
        AnswerEvaluationConversationItem.model_validate(
            {key: value for key, value in item.items() if key != "attemptId"}
        )
        for item in selected
    ]


def _follow_up_question(node: InterviewTopicNodeState, target_id: str) -> str | None:
    follow_up = next((item for item in node.followUps if item.id == target_id), None)
    return follow_up.question if follow_up and follow_up.question.strip() else None


def _format_lines(values: list[str]) -> str:
    return "\n".join([f"- {value}" for value in values]) if values else "(none)"


def _build_evaluator_prompt(context_prompt: str) -> str:
    return "\n\n".join(
        [
            "You are an answer evaluation subagent for a mock interview.",
            "Return JSON only.",
            "Do not reveal the reference answer.",
            "Use the reference answer as guidance, not as a script.",
            "Equivalent wording counts as covered.",
            "Do not require exact phrasing.",
            "Do not punish a candidate for giving a valid alternative explanation.",
            "Only mark incorrectPoints when the candidate says something technically wrong.",
            "Mark missingPoints for important gaps that matter for the asked question.",
            "Return exactly one JSON object that matches this schema:",
            "{",
            '  "classification": "direct-answer",',
            '  "score": {',
            '    "relevance": 8,',
            '    "accuracy": 8,',
            '    "depth": 7,',
            '    "specificity": 7,',
            '    "clarity": 8',
            "  },",
            '  "strengths": ["..."],',
            '  "missingPoints": ["..."],',
            '  "incorrectPoints": [],',
            '  "shouldAskFollowUp": false,',
            '  "followUpFocus": []',
            "}",
            "Do not put relevance, accuracy, depth, specificity, or clarity at the top level.",
            "score must be an object with all five numeric dimensions from 0 to 10.",
            "classification must be one of: direct-answer, partial-answer, deep-answer, "
            "off-topic, clarification-request, skip-request, stop-request, meta-question.",
            "followUpFocus must be an array of strings, never a single string.",
            "Never include the full reference answer in strengths, missingPoints, "
            "incorrectPoints, or followUpFocus.",
            context_prompt,
        ]
    )


def _parse_raw_model_json(value: Any) -> dict[str, Any]:
    content = _extract_model_content(value)
    if not content:
        raise ValueError("Model returned an empty response.")
    json_text = _extract_json_object_text(content)
    if not json_text:
        raise ValueError("Model response did not contain a JSON object.")
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _extract_model_content(value: Any) -> str | None:
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


def _log_metadata(context: AnswerEvaluationContext) -> dict[str, Any]:
    return {
        "taskId": context.evaluationId,
        "interviewId": context.interviewId,
        "roundId": context.roundId,
        "roundType": context.roundType,
        "nodeId": context.nodeId,
        "attemptId": context.attemptId,
        "targetType": context.targetType,
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_mock_chat_model(model: Any) -> bool:
    return model.__class__.__name__ == "MockChatModel"


def _build_mock_answer_evaluation(context: AnswerEvaluationContext) -> RawAnswerEvaluationOutput:
    has_answer = bool(context.candidateAnswer.strip())
    return RawAnswerEvaluationOutput.model_validate(
        {
            "classification": "direct-answer" if has_answer else "partial-answer",
            "score": {
                "relevance": 7 if has_answer else 4,
                "accuracy": 7 if has_answer else 4,
                "depth": 6 if has_answer else 3,
                "specificity": 6 if has_answer else 3,
                "clarity": 7 if has_answer else 4,
            },
            "strengths": ["回答与问题相关。"] if has_answer else [],
            "missingPoints": [] if has_answer else ["需要补充更具体的项目细节。"],
            "incorrectPoints": [],
            "shouldAskFollowUp": False,
            "followUpFocus": [],
        }
    )


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
