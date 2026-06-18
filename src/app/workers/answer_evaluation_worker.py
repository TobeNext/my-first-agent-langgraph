from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.integrations.llm_logging import log_llm_error, log_llm_input, log_llm_output
from app.integrations.models import ChatModelLike, create_chat_model
from app.integrations.redis_client import create_redis_answer_evaluation_store
from app.schemas.answer_evaluation import AnswerEvaluationTask, LlmAnswerEvaluationResult
from app.schemas.interview_state import AnswerClassification, AnswerScore

ANSWER_EVALUATION_PROMPT_VERSION = "answer-evaluation-v1"
DEFAULT_ANSWER_EVALUATION_MAX_ATTEMPTS = 3
DEFAULT_POLL_INTERVAL_SECONDS = 1.0

WorkerStatus = Literal["succeeded", "retrying", "failed"]


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


class AnswerEvaluationStoreLike(Protocol):
    async def claim_next_task(self) -> AnswerEvaluationTask | None: ...

    async def mark_succeeded(self, result: LlmAnswerEvaluationResult) -> None: ...

    async def mark_failed(self, task_id: str, error: str) -> None: ...

    async def retry_task(self, task_id: str, error: str | None = None) -> None: ...

    async def read_task_status(self, task_id: str) -> Any: ...


AnswerEvaluationModelEvaluator = Callable[
    [str, AnswerEvaluationTask],
    Awaitable[RawAnswerEvaluationOutput | dict[str, Any]],
]


@dataclass(frozen=True)
class AnswerEvaluationWorkerTickResult:
    processed: bool
    taskId: str | None = None
    status: WorkerStatus | None = None
    attempts: int | None = None
    error: str | None = None


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


def build_answer_evaluation_task_prompt(task: AnswerEvaluationTask) -> str:
    reference_answer = task.referenceAnswer.strip() if task.referenceAnswer else "(none)"
    node_conversation = json.dumps(
        [item.model_dump() for item in task.nodeConversation],
        ensure_ascii=False,
        indent=2,
    )
    return "\n\n".join(
        [
            f"Target role:\n{task.targetRole}",
            f"Round type:\n{task.roundType}",
            f"Question:\n{task.question}",
            f"Main question:\n{task.mainQuestion}",
            f"Reference answer:\n{reference_answer}",
            f"Reference answer points:\n{_format_lines(task.evaluationPoints)}",
            f"Candidate answer:\n{task.candidateAnswer}",
            f"Node conversation:\n{node_conversation}",
        ]
    )


async def evaluate_answer_with_model(
    prompt: str,
    task: AnswerEvaluationTask,
    *,
    model: ChatModelLike | None = None,
) -> RawAnswerEvaluationOutput:
    chat_model = model or create_chat_model()
    if _is_mock_chat_model(chat_model):
        return _build_mock_answer_evaluation(task)
    if not hasattr(chat_model, "with_structured_output"):
        raise RuntimeError("Configured chat model does not support structured output.")
    evaluator_prompt = _build_evaluator_prompt(prompt)
    metadata = {
        "taskId": task.taskId,
        "interviewId": task.interviewId,
        "roundId": task.roundId,
        "roundType": task.roundType,
        "nodeId": task.nodeId,
        "attemptId": task.attemptId,
        "targetType": task.targetType,
    }
    log_llm_input(
        thread_id=task.threadId,
        operation="answer-evaluation",
        prompt=evaluator_prompt,
        metadata=metadata,
    )
    structured_model = chat_model.with_structured_output(RawAnswerEvaluationOutput)
    try:
        try:
            result = structured_model.invoke(evaluator_prompt)
        except Exception as exc:
            log_llm_error(
                thread_id=task.threadId,
                operation="answer-evaluation",
                error=exc,
                metadata={**metadata, "stage": "structured-output"},
            )
            result = _parse_raw_model_json(chat_model.invoke(evaluator_prompt))
        parsed = RawAnswerEvaluationOutput.model_validate(result)
        log_llm_output(
            thread_id=task.threadId,
            operation="answer-evaluation",
            output=parsed,
            metadata=metadata,
        )
        return parsed
    except Exception as exc:
        log_llm_error(
            thread_id=task.threadId,
            operation="answer-evaluation",
            error=exc,
            metadata=metadata,
        )
        raise


def build_llm_answer_evaluation_result(
    *,
    task: AnswerEvaluationTask,
    raw_evaluation: RawAnswerEvaluationOutput | dict[str, Any],
    now: str,
    evaluator_model: str,
    prompt_version: str,
) -> LlmAnswerEvaluationResult:
    raw = RawAnswerEvaluationOutput.model_validate(raw_evaluation)
    return LlmAnswerEvaluationResult.model_validate(
        {
            "schemaVersion": 1,
            "taskId": task.taskId,
            "interviewId": task.interviewId,
            "threadId": task.threadId,
            "nodeId": task.nodeId,
            "roundId": task.roundId,
            "roundType": task.roundType,
            "attemptId": task.attemptId,
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


class AnswerEvaluationRunner:
    def __init__(
        self,
        *,
        store: AnswerEvaluationStoreLike,
        evaluator: AnswerEvaluationModelEvaluator | None = None,
        now: Callable[[], str] | None = None,
        evaluator_model: str | None = None,
        prompt_version: str = ANSWER_EVALUATION_PROMPT_VERSION,
        max_attempts: int = DEFAULT_ANSWER_EVALUATION_MAX_ATTEMPTS,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.evaluator = evaluator or (
            lambda prompt, task: evaluate_answer_with_model(prompt, task)
        )
        self.now = now or _utc_now
        self.evaluator_model = evaluator_model or settings.model_name
        self.prompt_version = prompt_version
        self.max_attempts = max_attempts

    async def run_once(self) -> AnswerEvaluationWorkerTickResult:
        task = await self.store.claim_next_task()
        if not task:
            return AnswerEvaluationWorkerTickResult(processed=False)

        prompt = build_answer_evaluation_task_prompt(task)
        try:
            raw_evaluation = await self.evaluator(prompt, task)
            result = build_llm_answer_evaluation_result(
                task=task,
                raw_evaluation=raw_evaluation,
                now=self.now(),
                evaluator_model=self.evaluator_model,
                prompt_version=self.prompt_version,
            )
            await self.store.mark_succeeded(result)
            return AnswerEvaluationWorkerTickResult(
                processed=True,
                taskId=task.taskId,
                status="succeeded",
            )
        except Exception as exc:
            message = _format_evaluation_error(exc)
            status = await self.store.read_task_status(task.taskId)
            attempts = status.attempts if status else 0

            if attempts >= self.max_attempts:
                await self.store.mark_failed(task.taskId, message)
                return AnswerEvaluationWorkerTickResult(
                    processed=True,
                    taskId=task.taskId,
                    status="failed",
                    attempts=attempts,
                    error=message,
                )

            await self.store.retry_task(task.taskId, message)
            return AnswerEvaluationWorkerTickResult(
                processed=True,
                taskId=task.taskId,
                status="retrying",
                attempts=attempts,
                error=message,
            )

    async def run_forever(
        self,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        while True:
            result = await self.run_once()
            if not result.processed:
                await asyncio.sleep(poll_interval_seconds)


def create_default_answer_evaluation_runner() -> AnswerEvaluationRunner:
    return AnswerEvaluationRunner(store=create_redis_answer_evaluation_store())


def _format_lines(values: list[str]) -> str:
    return "\n".join([f"- {value}" for value in values]) if values else "(none)"


def _build_evaluator_prompt(task_prompt: str) -> str:
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
            "Score each dimension from 0 to 10:",
            "- relevance: answer addresses the asked question and stays on topic.",
            "- accuracy: technical correctness compared with reference answer.",
            "- depth: mechanisms, trade-offs, edge cases, reasoning.",
            "- specificity: concrete implementation details, project evidence, constraints.",
            "- clarity: structure, readability, coherence.",
            "Never include the full reference answer in strengths, missingPoints, "
            "incorrectPoints, or followUpFocus.",
            task_prompt,
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


def _format_evaluation_error(error: Exception) -> str:
    return str(error)


def _is_mock_chat_model(model: Any) -> bool:
    return model.__class__.__name__ == "MockChatModel"


def _build_mock_answer_evaluation(task: AnswerEvaluationTask) -> RawAnswerEvaluationOutput:
    has_answer = bool(task.candidateAnswer.strip())
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
