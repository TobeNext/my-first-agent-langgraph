import importlib.util
import json
from pathlib import Path
from typing import Any

from app.domain.report_generation import (
    build_report_generation_prompt,
    generate_report_with_model,
)
from app.integrations.models import MockChatModel
from app.schemas.answer_evaluation import (
    AnswerEvaluationTask,
    InterviewEvaluationManifest,
    LlmAnswerEvaluationResult,
)
from app.schemas.interview_report import (
    InterviewReportRecord,
    InterviewReportWrite,
    ReportGenerationOutput,
    ReportGenerationTask,
    ReportGenerationTaskStatus,
)
from app.workers.report_generation_worker import ReportGenerationRunner

NOW = "2026-06-19T00:00:00.000Z"


def build_task(overrides: dict | None = None) -> ReportGenerationTask:
    payload = {
        "schemaVersion": 1,
        "taskId": "report-task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "resourceId": "resource-1",
        "targetRole": "Backend Engineer",
        "responseLanguage": "zh",
        "evaluationManifestKey": "interview:interview-1:evaluation:manifest",
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return ReportGenerationTask.model_validate(payload)


def build_answer_task(overrides: dict | None = None) -> AnswerEvaluationTask:
    payload = {
        "schemaVersion": 1,
        "taskId": "answer-task-1",
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
        "question": "请说明 RAG 链路。",
        "mainQuestion": "请说明 RAG 链路。",
        "referenceAnswer": "内部参考答案，不应完整外泄。",
        "evaluationPoints": ["覆盖召回", "覆盖重排"],
        "candidateAnswer": "我会先召回再重排。",
        "nodeConversation": [],
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return AnswerEvaluationTask.model_validate(payload)


def build_answer_result(overrides: dict | None = None) -> LlmAnswerEvaluationResult:
    payload = {
        "schemaVersion": 1,
        "taskId": "answer-task-1",
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "nodeId": "node-1",
        "roundId": "round-1",
        "roundType": "professional-skills",
        "attemptId": "attempt-1",
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 8,
            "depth": 7,
            "specificity": 7,
            "clarity": 8,
            "weightedTotal": 7.65,
        },
        "strengths": ["覆盖召回和重排。"],
        "missingPoints": [],
        "incorrectPoints": [],
        "shouldAskFollowUp": False,
        "followUpFocus": [],
        "evaluatorModel": "mock-model",
        "promptVersion": "answer-evaluation-v1",
        "createdAt": NOW,
    }
    payload.update(overrides or {})
    return LlmAnswerEvaluationResult.model_validate(payload)


def build_manifest(**overrides: object) -> InterviewEvaluationManifest:
    payload = {
        "schemaVersion": 1,
        "interviewId": "interview-1",
        "threadId": "thread-1",
        "expectedTaskIds": ["answer-task-1"],
        "completedTaskIds": ["answer-task-1"],
        "failedTaskIds": [],
        "sealed": True,
        "sealedAt": NOW,
        "updatedAt": NOW,
    }
    payload.update(overrides)
    return InterviewEvaluationManifest.model_validate(payload)


def valid_output(overrides: dict | None = None) -> dict[str, Any]:
    payload = {
        "summary": {
            "overallScore": 8,
            "overallComment": "候选人理解核心流程，但需要补充边界场景。",
            "strengths": ["能说明核心链路。"],
            "improvementPriorities": ["补充失败降级和观测指标。"],
        },
        "questionReviews": [
            {
                "questionId": "node-1",
                "attemptId": "attempt-1",
                "targetType": "main-question",
                "question": "请说明 RAG 链路。",
                "score": 8,
                "comment": "回答覆盖召回和重排。",
                "missingPoints": [],
                "improvementAdvice": ["补充失败降级。"],
            }
        ],
        "markdown": "## 模拟面试报告\n\n### 总体评价\n候选人理解核心流程。",
    }
    payload.update(overrides or {})
    return payload


class FakeReportGenerationStore:
    def __init__(self, task: ReportGenerationTask | None) -> None:
        self.task = task
        self.statuses: list[str] = []
        self.succeeded_report_id: str | None = None
        self.markdown_available: bool | None = None
        self.failed_counts: tuple[int | None, int | None, int | None] | None = None
        self.status = (
            ReportGenerationTaskStatus.model_validate(
                {
                    "schemaVersion": 1,
                    "taskId": task.taskId,
                    "interviewId": task.interviewId,
                    "status": "pending",
                    "attempts": 0,
                    "createdAt": NOW,
                }
            )
            if task
            else None
        )

    async def mark_succeeded(
        self,
        task_id: str,
        *,
        report_id: str,
        evaluation_expected_count: int,
        evaluation_completed_count: int,
        evaluation_failed_count: int,
        markdown_available: bool = True,
    ) -> None:
        self.succeeded_report_id = report_id
        self.markdown_available = markdown_available
        self.status = self.status.model_copy(
            update={"status": "succeeded", "completedAt": NOW, "lastError": None}
        )
        self.statuses.append("succeeded")

    async def claim_next_task(self) -> ReportGenerationTask | None:
        task = self.task
        self.task = None
        if task and self.status:
            self.status = self.status.model_copy(
                update={
                    "status": "running",
                    "attempts": self.status.attempts + 1,
                    "startedAt": NOW,
                }
            )
            self.statuses.append("running")
        return task

    async def mark_failed(
        self,
        task_id: str,
        error: str,
        *,
        evaluation_expected_count: int | None = None,
        evaluation_completed_count: int | None = None,
        evaluation_failed_count: int | None = None,
    ) -> None:
        self.failed_counts = (
            evaluation_expected_count,
            evaluation_completed_count,
            evaluation_failed_count,
        )
        self.status = self.status.model_copy(
            update={"status": "failed", "completedAt": NOW, "lastError": error}
        )
        self.statuses.append("failed")

    async def retry_task(self, task_id: str, error: str | None = None) -> None:
        self.status = self.status.model_copy(
            update={"status": "pending", "lastError": error}
        )
        self.task = build_task({"taskId": task_id})
        self.statuses.append("retrying")

    async def read_task_status(self, task_id: str) -> ReportGenerationTaskStatus | None:
        return self.status


class FakeAnswerEvaluationStore:
    def __init__(
        self,
        *,
        manifest: InterviewEvaluationManifest | None = None,
        tasks: list[AnswerEvaluationTask] | None = None,
        results: list[LlmAnswerEvaluationResult] | None = None,
    ) -> None:
        self.manifest = manifest if manifest is not None else build_manifest()
        self.tasks = {task.taskId: task for task in (tasks or [build_answer_task()])}
        self.results = results if results is not None else [build_answer_result()]

    async def read_manifest(self, interview_id: str) -> InterviewEvaluationManifest | None:
        return self.manifest

    async def read_results(self, interview_id: str) -> list[LlmAnswerEvaluationResult]:
        return self.results

    async def read_task(self, task_id: str) -> AnswerEvaluationTask | None:
        return self.tasks.get(task_id)


class FakeReportRepository:
    def __init__(
        self,
        *,
        existing: InterviewReportRecord | None = None,
        fail_write: bool = False,
    ) -> None:
        self.existing = existing
        self.fail_write = fail_write
        self.writes: list[InterviewReportWrite] = []

    def get_report_by_interview_id(self, interview_id: str) -> InterviewReportRecord | None:
        return self.existing

    def write_report(self, report: InterviewReportWrite) -> InterviewReportRecord:
        if self.fail_write:
            raise RuntimeError("DB write failed")
        self.writes.append(report)
        return InterviewReportRecord(
            id=report.id,
            interview_id=report.interview_id,
            thread_id=report.thread_id,
            target_role=report.target_role,
            response_language=report.response_language,
            status=report.status,
            overall_score=report.overall_score,
            markdown=report.markdown,
            structured_json=report.structured_json,
            prompt_version=report.prompt_version,
            model_name=report.model_name,
            source_evaluation_manifest_json=report.source_evaluation_manifest_json,
            created_at=report.created_at,
            updated_at=report.updated_at,
            completed_at=report.completed_at,
        )


def _runner(
    *,
    store: FakeReportGenerationStore,
    evaluation_store: FakeAnswerEvaluationStore | None = None,
    repository: FakeReportRepository | None = None,
    evaluator: Any | None = None,
    max_attempts: int = 3,
) -> ReportGenerationRunner:
    return ReportGenerationRunner(
        store=store,
        evaluation_store=evaluation_store or FakeAnswerEvaluationStore(),
        repository=repository or FakeReportRepository(),
        evaluator=evaluator,
        now=lambda: NOW,
        max_attempts=max_attempts,
    )


def test_build_report_generation_prompt_contains_scoring_rules() -> None:
    prompt = build_report_generation_prompt(
        task=build_task(),
        interview_metadata={"responseLanguage": "zh"},
        evaluation_results=[{"score": {"weightedTotal": 8}}],
        question_answer_context=[
            {
                "targetType": "main-question",
                "referenceAnswer": "internal reference",
                "evaluationPoints": ["覆盖召回", "覆盖重排"],
            }
        ],
    )

    assert "referenceAnswer and evaluationPoints" in prompt
    assert "directness" in prompt
    assert "technical_depth" in prompt
    assert "evidence_specificity" in prompt
    assert "clarity_structure" in prompt
    assert "Do not include full reference answers" in prompt


async def test_mock_model_generates_schema_valid_report_output() -> None:
    output = await generate_report_with_model(
        prompt="prompt",
        task=build_task(),
        model=MockChatModel(),
    )

    assert output.summary.overallScore == 7
    assert output.questionReviews[0].targetType == "main-question"
    assert output.markdown.startswith("## 模拟面试报告")


async def test_report_generation_model_falls_back_to_raw_json_when_structured_fails() -> None:
    class StructuredFailsRawSucceedsModel:
        def with_structured_output(self, schema: type[Any]) -> Any:
            class _Structured:
                def invoke(self, prompt: str) -> Any:
                    raise RuntimeError("response_format unavailable")

            return _Structured()

        def invoke(self, prompt: str) -> str:
            return json.dumps(valid_output(), ensure_ascii=False)

    output = await generate_report_with_model(
        prompt="prompt",
        task=build_task(),
        model=StructuredFailsRawSucceedsModel(),
    )

    assert output.summary.overallScore == 8
    assert output.questionReviews[0].questionId == "node-1"


async def test_report_generation_runner_generates_structured_output_with_mock_evaluator() -> None:
    store = FakeReportGenerationStore(build_task())
    seen_prompts: list[str] = []
    repository = FakeReportRepository()

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        seen_prompts.append(prompt)
        return valid_output()

    runner = _runner(
        store=store,
        repository=repository,
        evaluator=evaluator,
    )

    result = await runner.run_once()

    assert result.processed is True
    assert result.status == "succeeded"
    assert isinstance(result.output, ReportGenerationOutput)
    assert seen_prompts and "Question and answer context:" in seen_prompts[0]
    assert "我会先召回再重排。" in seen_prompts[0]
    assert store.statuses == ["running", "succeeded"]
    assert store.succeeded_report_id == "report-interview-1"
    assert store.markdown_available is True
    assert repository.writes[0].markdown.startswith("## 模拟面试报告")
    assert repository.writes[0].items[0].task_id == "answer-task-1"
    assert repository.writes[0].items[0].candidate_answer == "我会先召回再重排。"


async def test_report_generation_runner_retries_invalid_output_then_succeeds() -> None:
    store = FakeReportGenerationStore(build_task())
    calls = 0

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls < 3:
            return {"markdown": ""}
        return valid_output()

    runner = _runner(store=store, evaluator=evaluator)

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


async def test_report_generation_runner_marks_failed_after_max_attempts() -> None:
    store = FakeReportGenerationStore(build_task())

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        return {"markdown": ""}

    runner = _runner(store=store, evaluator=evaluator, max_attempts=2)

    await runner.run_once()
    result = await runner.run_once()

    assert result.status == "failed"
    assert result.attempts == 2
    assert store.status and store.status.status == "failed"
    assert store.status.lastError
    assert store.statuses == ["running", "retrying", "running", "failed"]


async def test_report_generation_runner_requeues_when_evaluation_not_complete() -> None:
    store = FakeReportGenerationStore(build_task())
    evaluation_store = FakeAnswerEvaluationStore(
        manifest=build_manifest(completedTaskIds=[], sealed=True),
        results=[],
    )
    repository = FakeReportRepository()

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        raise AssertionError("should not generate partial report")

    runner = _runner(
        store=store,
        evaluation_store=evaluation_store,
        repository=repository,
        evaluator=evaluator,
    )

    result = await runner.run_once()

    assert result.status == "retrying"
    assert result.error == "evaluation results are still pending"
    assert repository.writes == []
    assert store.succeeded_report_id is None
    assert store.markdown_available is None


async def test_report_generation_runner_marks_failed_when_evaluation_failed() -> None:
    store = FakeReportGenerationStore(build_task())
    evaluation_store = FakeAnswerEvaluationStore(
        manifest=build_manifest(completedTaskIds=[], failedTaskIds=["answer-task-1"]),
        results=[],
    )
    repository = FakeReportRepository()

    runner = _runner(
        store=store,
        evaluation_store=evaluation_store,
        repository=repository,
    )

    result = await runner.run_once()

    assert result.status == "failed"
    assert "evaluation failed tasks" in (result.error or "")
    assert repository.writes == []
    assert store.statuses == ["running", "failed"]
    assert store.failed_counts == (1, 0, 1)


async def test_report_generation_runner_does_not_mark_ready_when_db_write_fails() -> None:
    store = FakeReportGenerationStore(build_task())
    repository = FakeReportRepository(fail_write=True)

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        return valid_output()

    runner = _runner(
        store=store,
        repository=repository,
        evaluator=evaluator,
        max_attempts=1,
    )

    result = await runner.run_once()

    assert result.status == "failed"
    assert result.error == "DB write failed"
    assert store.succeeded_report_id is None
    assert store.markdown_available is None
    assert store.statuses == ["running", "failed"]


async def test_report_generation_runner_restores_redis_ready_from_existing_db_report() -> None:
    store = FakeReportGenerationStore(build_task())
    existing = InterviewReportRecord(
        id="existing-report-1",
        interview_id="interview-1",
        thread_id="thread-1",
        target_role="Backend Engineer",
        response_language="zh",
        status="succeeded",
        overall_score=8,
        markdown="## Existing",
        structured_json="{}",
        prompt_version="report-generation-v1",
        model_name="mock-model",
        source_evaluation_manifest_json="{}",
        created_at=NOW,
        updated_at=NOW,
        completed_at=NOW,
    )
    repository = FakeReportRepository(existing=existing)

    async def evaluator(prompt: str, task: ReportGenerationTask) -> dict[str, Any]:
        raise AssertionError("should not regenerate existing report")

    runner = _runner(
        store=store,
        repository=repository,
        evaluator=evaluator,
    )

    result = await runner.run_once()

    assert result.status == "succeeded"
    assert repository.writes == []
    assert store.succeeded_report_id == "existing-report-1"
    assert store.markdown_available is True
    assert store.statuses == ["running", "succeeded"]


def test_run_report_generation_worker_script_is_importable() -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_report_generation_worker.py"
    )
    spec = importlib.util.spec_from_file_location("run_report_generation_worker", script_path)

    assert spec and spec.loader
