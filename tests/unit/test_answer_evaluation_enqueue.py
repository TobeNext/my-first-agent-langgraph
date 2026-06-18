from app.domain.answer_evaluation_enqueue import (
    build_answer_evaluation_task,
    enqueue_answer_evaluation_task_best_effort,
)
from app.domain.interview_state_machine import AnswerEvaluationResult, apply_user_reply
from app.schemas.answer_evaluation import AnswerEvaluationTask
from app.schemas.interview_state import AnswerScore
from tests.unit.test_interview_state_machine import _state_fixture

NOW = "2026-06-07T00:00:00.000Z"


class FakeStore:
    def __init__(self) -> None:
        self.tasks: list[AnswerEvaluationTask] = []

    async def enqueue_task(self, task: AnswerEvaluationTask) -> None:
        self.tasks.append(task)


class FailingStore:
    async def enqueue_task(self, task: AnswerEvaluationTask) -> None:
        raise RuntimeError("Redis unavailable")


def _score(value: float = 8) -> AnswerScore:
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


def _evaluation() -> AnswerEvaluationResult:
    return AnswerEvaluationResult(
        classification="direct-answer",
        score=_score(),
        strengths=["回答围绕 RAG 展开"],
        missingPoints=["异常路径还不够完整"],
        incorrectPoints=[],
        recommendedIntent="depth",
        followUpFocus=["异常路径"],
        shouldCompleteNode=False,
    )


def test_build_answer_evaluation_task_from_new_scored_attempt() -> None:
    before_state = _state_fixture(flow_test=False)
    user_message = "我会说明 query rewrite、向量召回和重排。"
    result = apply_user_reply(before_state, user_message, _evaluation())

    task = build_answer_evaluation_task(
        before_state=before_state,
        after_state=result.state,
        user_message=user_message,
        resource_id="resource-1",
        now=lambda: NOW,
        create_task_id=lambda attempt: f"task-{attempt.id}",
    )

    assert task
    assert task.schemaVersion == 1
    assert task.interviewId == before_state.threadId
    assert task.resourceId == "resource-1"
    assert task.roundType == "professional-skills"
    assert task.targetType == "main-question"
    assert task.targetRole == "通用技术岗位"
    assert task.question == "请解释你的 RAG 链路。"
    assert task.mainQuestion == "请解释你的 RAG 链路。"
    assert task.candidateAnswer == user_message
    assert task.createdAt == NOW
    assert task.taskId == f"task-{task.attemptId}"
    assert task.nodeConversation[-1].role == "candidate"
    assert task.nodeConversation[-1].text == user_message


def test_build_answer_evaluation_task_skips_detour_attempts() -> None:
    before_state = _state_fixture(flow_test=False)
    user_message = "这题为什么这么问？"
    result = apply_user_reply(
        before_state,
        user_message,
        AnswerEvaluationResult(
            classification="meta-question",
            score=None,
            strengths=[],
            missingPoints=[],
            incorrectPoints=[],
            recommendedIntent="depth",
            followUpFocus=[],
            shouldCompleteNode=False,
        ),
    )

    assert (
        build_answer_evaluation_task(
            before_state=before_state,
            after_state=result.state,
            user_message=user_message,
        )
        is None
    )


def test_enqueue_answer_evaluation_task_best_effort_uses_injected_store() -> None:
    before_state = _state_fixture(flow_test=False)
    user_message = "我会说明 query rewrite、向量召回和重排。"
    result = apply_user_reply(before_state, user_message, _evaluation())
    store = FakeStore()

    task = enqueue_answer_evaluation_task_best_effort(
        before_state=before_state,
        after_state=result.state,
        user_message=user_message,
        store=store,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert task
    assert store.tasks == [task]


def test_enqueue_answer_evaluation_task_best_effort_keeps_flow_safe_on_store_failure() -> None:
    before_state = _state_fixture(flow_test=False)
    user_message = "我会说明 query rewrite、向量召回和重排。"
    result = apply_user_reply(before_state, user_message, _evaluation())

    task = enqueue_answer_evaluation_task_best_effort(
        before_state=before_state,
        after_state=result.state,
        user_message=user_message,
        store=FailingStore(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    assert task
    assert task.candidateAnswer == user_message
