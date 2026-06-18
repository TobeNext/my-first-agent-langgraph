import json
import logging
from typing import Any

from app.integrations.models import MockChatModel
from app.schemas.answer_evaluation import (
    AnswerEvaluationTask,
    AnswerEvaluationTaskStatus,
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)
from app.workers.answer_evaluation_worker import (
    AnswerEvaluationRunner,
    build_answer_evaluation_task_prompt,
    calculate_answer_weighted_total,
    evaluate_answer_with_model,
)

NOW = "2026-06-07T00:00:00.000Z"


def build_task(overrides: dict | None = None) -> AnswerEvaluationTask:
    payload = {
        "schemaVersion": 1,
        "taskId": "task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "nodeId": "node-1",
        "roundId": "round-1",
        "roundType": "professional-skills",
        "attemptId": "attempt-1",
        "targetType": "main-question",
        "targetId": "node-1",
        "targetRole": "Backend Engineer",
        "responseLanguage": "zh",
        "question": "请说明 Spring 事务传播机制。",
        "mainQuestion": "请说明 Spring 事务传播机制。",
        "referenceAnswer": "说明 REQUIRED 和 REQUIRES_NEW；补充异常回滚边界。",
        "evaluationPoints": ["说明 REQUIRED 和 REQUIRES_NEW", "补充异常回滚边界"],
        "candidateAnswer": "我会说明 REQUIRED 和 REQUIRES_NEW 的区别。",
        "nodeConversation": [],
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return AnswerEvaluationTask.model_validate(payload)


class FakeAnswerEvaluationStore:
    def __init__(self, task: AnswerEvaluationTask | None) -> None:
        self.task = task
        self.statuses: list[str] = []
        self.result: LlmAnswerEvaluationResult | None = None
        self.status = (
            AnswerEvaluationTaskStatus.model_validate(
                {
                    "schemaVersion": 1,
                    "taskId": task.taskId,
                    "interviewId": task.interviewId,
                    "attemptId": task.attemptId,
                    "status": "pending",
                    "attempts": 0,
                    "createdAt": NOW,
                }
            )
            if task
            else None
        )
        self.manifest = (
            InterviewEvaluationManifest.model_validate(
                {
                    "schemaVersion": 1,
                    "interviewId": task.interviewId,
                    "threadId": task.threadId,
                    "expectedTaskIds": [task.taskId],
                    "completedTaskIds": [],
                    "failedTaskIds": [],
                    "sealed": False,
                    "updatedAt": NOW,
                }
            )
            if task
            else None
        )

    async def claim_next_task(self) -> AnswerEvaluationTask | None:
        task = self.task
        self.task = None
        if task:
            self.status = self.status.model_copy(
                update={
                    "status": "running",
                    "attempts": self.status.attempts + 1,
                    "startedAt": NOW,
                }
            )
            self.statuses.append("running")
        return task

    async def mark_succeeded(self, result: LlmAnswerEvaluationResult) -> None:
        self.result = result
        self.status = self.status.model_copy(
            update={"status": "succeeded", "completedAt": NOW, "lastError": None}
        )
        self.manifest = self.manifest.model_copy(
            update={
                "completedTaskIds": [result.taskId],
                "failedTaskIds": [],
            }
        )
        self.statuses.append("succeeded")

    async def mark_failed(self, task_id: str, error: str) -> None:
        self.status = self.status.model_copy(
            update={"status": "failed", "completedAt": NOW, "lastError": error}
        )
        self.manifest = self.manifest.model_copy(
            update={"completedTaskIds": [], "failedTaskIds": [task_id]}
        )
        self.statuses.append("failed")

    async def retry_task(self, task_id: str, error: str | None = None) -> None:
        self.status = self.status.model_copy(
            update={"status": "pending", "lastError": error}
        )
        self.task = build_task({"taskId": task_id})
        self.statuses.append("retrying")

    async def read_task_status(self, task_id: str) -> AnswerEvaluationTaskStatus | None:
        return self.status


def raw_evaluation() -> dict:
    return {
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 7,
            "depth": 6,
            "specificity": 5,
            "clarity": 9,
        },
        "strengths": ["覆盖了核心传播类型"],
        "missingPoints": ["异常回滚边界还不够完整"],
        "incorrectPoints": [],
        "shouldAskFollowUp": True,
        "followUpFocus": ["异常回滚边界"],
    }


class StructuredEvaluationModel:
    def with_structured_output(self, schema: type[Any]) -> Any:
        class _Structured:
            def invoke(self, prompt: str) -> Any:
                assert "You are an answer evaluation subagent" in prompt
                return schema.model_validate(raw_evaluation())

        return _Structured()


class StructuredEvaluationFailsRawSucceedsModel:
    def with_structured_output(self, schema: type[Any]) -> Any:
        class _Structured:
            def invoke(self, prompt: str) -> Any:
                raise RuntimeError("response_format unavailable")

        return _Structured()

    def invoke(self, prompt: str) -> str:
        return json.dumps(raw_evaluation(), ensure_ascii=False)


def test_build_answer_evaluation_task_prompt_contains_evaluator_context() -> None:
    prompt = build_answer_evaluation_task_prompt(build_task())

    assert "Reference answer:" in prompt
    assert "说明 REQUIRED 和 REQUIRES_NEW；补充异常回滚边界。" in prompt
    assert "Candidate answer:" in prompt


def test_calculate_answer_weighted_total_uses_fixed_formula() -> None:
    assert (
        calculate_answer_weighted_total(
            {
                "relevance": 8,
                "accuracy": 7,
                "depth": 6,
                "specificity": 5,
                "clarity": 9,
            }
        )
        == 6.9
    )


async def test_runner_claims_evaluates_and_marks_success() -> None:
    store = FakeAnswerEvaluationStore(build_task())
    seen_prompts: list[str] = []

    async def evaluator(prompt: str, task: AnswerEvaluationTask) -> dict:
        seen_prompts.append(prompt)
        return raw_evaluation()

    runner = AnswerEvaluationRunner(
        store=store,
        now=lambda: NOW,
        evaluator_model="mock-model",
        evaluator=evaluator,
    )

    result = await runner.run_once()

    assert result.processed is True
    assert result.status == "succeeded"
    assert store.statuses == ["running", "succeeded"]
    assert seen_prompts and "Reference answer:" in seen_prompts[0]
    assert store.result
    assert store.result.evaluatorModel == "mock-model"
    assert store.result.promptVersion == "answer-evaluation-v1"
    assert store.result.score.weightedTotal == 6.9
    assert "说明 REQUIRED 和 REQUIRES_NEW" not in store.result.model_dump_json()


async def test_runner_retries_until_later_attempt_succeeds() -> None:
    store = FakeAnswerEvaluationStore(build_task())
    calls = 0

    async def evaluator(prompt: str, task: AnswerEvaluationTask) -> dict:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("temporary model failure")
        return raw_evaluation()

    runner = AnswerEvaluationRunner(store=store, now=lambda: NOW, evaluator=evaluator)

    first = await runner.run_once()
    second = await runner.run_once()
    third = await runner.run_once()

    assert first.status == "retrying"
    assert first.attempts == 1
    assert second.status == "retrying"
    assert second.attempts == 2
    assert third.status == "succeeded"
    assert store.statuses == [
        "running",
        "retrying",
        "running",
        "retrying",
        "running",
        "succeeded",
    ]
    assert store.status.status == "succeeded"
    assert store.status.attempts == 3
    assert store.manifest.completedTaskIds == ["task-1"]
    assert store.manifest.failedTaskIds == []


async def test_runner_marks_failed_after_max_attempts() -> None:
    store = FakeAnswerEvaluationStore(build_task())

    async def evaluator(prompt: str, task: AnswerEvaluationTask) -> dict:
        raise RuntimeError("model returned invalid JSON")

    runner = AnswerEvaluationRunner(
        store=store,
        max_attempts=3,
        evaluator=evaluator,
    )

    await runner.run_once()
    await runner.run_once()
    result = await runner.run_once()

    assert result.status == "failed"
    assert result.attempts == 3
    assert result.error == "model returned invalid JSON"
    assert store.status.status == "failed"
    assert store.status.lastError == "model returned invalid JSON"
    assert store.manifest.completedTaskIds == []
    assert store.manifest.failedTaskIds == ["task-1"]


async def test_runner_does_nothing_when_no_task_available() -> None:
    store = FakeAnswerEvaluationStore(None)
    runner = AnswerEvaluationRunner(store=store)

    result = await runner.run_once()

    assert result.processed is False
    assert store.statuses == []


async def test_mock_model_produces_deterministic_worker_evaluation() -> None:
    result = await evaluate_answer_with_model("prompt", build_task(), model=MockChatModel())

    assert result.classification == "direct-answer"
    assert result.score.relevance == 7
    assert result.strengths == ["回答与问题相关。"]


async def test_evaluate_answer_with_model_logs_llm_input_and_output(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.llm")
    task = build_task()

    result = await evaluate_answer_with_model(
        build_answer_evaluation_task_prompt(task),
        task,
        model=StructuredEvaluationModel(),
    )

    assert result.classification == "direct-answer"
    events = [json.loads(record.message) for record in caplog.records]
    assert [event["event"] for event in events] == ["llm.input", "llm.output"]
    assert all(event["threadId"] == "thread-1" for event in events)
    assert events[0]["operation"] == "answer-evaluation"
    assert events[0]["metadata"]["taskId"] == "task-1"
    assert "Candidate answer:" in events[0]["prompt"]
    assert events[1]["output"]["classification"] == "direct-answer"


async def test_evaluate_answer_with_model_falls_back_to_raw_json_when_structured_fails() -> None:
    task = build_task()

    result = await evaluate_answer_with_model(
        build_answer_evaluation_task_prompt(task),
        task,
        model=StructuredEvaluationFailsRawSucceedsModel(),
    )

    assert result.classification == "direct-answer"
    assert result.followUpFocus == ["异常回滚边界"]
