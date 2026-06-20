from typing import Any

import pytest

from app.domain.answer_evaluation_runtime import (
    build_answer_evaluation_context_prompt,
    build_answer_evaluation_contexts_from_state,
    calculate_answer_weighted_total,
    evaluate_answer_contexts,
)
from app.schemas.interview_state import AnswerAttemptState, AnswerScore
from tests.unit.test_interview_state_machine import _state_fixture

NOW = "2026-06-07T00:00:00.000Z"


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


def _attempt(
    attempt_id: str,
    *,
    target_type: str = "main-question",
    target_id: str = "node-rag",
    user_message: str = "我会先做 query rewrite，再做向量召回和重排。",
    created_at: str = NOW,
    is_detour: bool = False,
) -> AnswerAttemptState:
    return AnswerAttemptState.model_validate(
        {
            "id": attempt_id,
            "targetType": target_type,
            "targetId": target_id,
            "userMessage": user_message,
            "classification": "direct-answer",
            "score": _score().model_dump(),
            "strengths": ["结构清晰"],
            "missingPoints": [],
            "incorrectPoints": [],
            "isDetour": is_detour,
            "createdAt": created_at,
        }
    )


def _state_with_answer_attempts():
    state = _state_fixture(flow_test=False)
    node = state.rounds[0].nodes[0]
    node = node.model_copy(
        update={
            "referenceAnswer": "覆盖 query rewrite、召回、重排和生成。",
            "evaluationPoints": ["说明 query rewrite", "说明重排"],
            "followUps": [
                node.followUps[0].model_copy(
                    update={
                        "question": "如果向量召回结果不稳定，你会如何排查？",
                        "status": "answered",
                        "linkedAnswerId": "attempt-follow-up",
                    }
                ),
                node.followUps[1],
            ],
            "answerAttempts": [
                _attempt("attempt-main"),
                _attempt(
                    "attempt-detour",
                    user_message="这题考察什么？",
                    is_detour=True,
                ),
                _attempt(
                    "attempt-follow-up",
                    target_type="follow-up",
                    target_id="follow-up-1",
                    user_message="我会看 embedding 分布、召回 topK 和重排日志。",
                    created_at="2026-06-07T00:00:01.000Z",
                ),
            ],
        },
        deep=True,
    )
    round_item = state.rounds[0].model_copy(update={"nodes": [node]}, deep=True)
    return state.model_copy(update={"rounds": [round_item, state.rounds[1]]}, deep=True)


def raw_evaluation() -> dict[str, Any]:
    return {
        "classification": "direct-answer",
        "score": {
            "relevance": 8,
            "accuracy": 7,
            "depth": 6,
            "specificity": 5,
            "clarity": 9,
        },
        "strengths": ["覆盖了核心链路"],
        "missingPoints": ["异常路径还不够完整"],
        "incorrectPoints": [],
        "shouldAskFollowUp": False,
        "followUpFocus": [],
    }


def test_build_answer_evaluation_contexts_from_session_attempts() -> None:
    contexts = build_answer_evaluation_contexts_from_state(
        _state_with_answer_attempts(),
        resource_id="resource-1",
    )

    assert [context.attemptId for context in contexts] == [
        "attempt-main",
        "attempt-follow-up",
    ]
    first, second = contexts
    assert first.evaluationId == "answer-evaluation-attempt-main"
    assert first.resourceId == "resource-1"
    assert first.question == "请解释你的 RAG 链路。"
    assert first.referenceAnswer == "覆盖 query rewrite、召回、重排和生成。"
    assert first.evaluationPoints == ["说明 query rewrite", "说明重排"]
    assert first.nodeConversation[-1].text == "我会先做 query rewrite，再做向量召回和重排。"
    assert second.targetType == "follow-up"
    assert second.question == "如果向量召回结果不稳定，你会如何排查？"
    assert second.followUpQuestion == "如果向量召回结果不稳定，你会如何排查？"
    assert [item.role for item in second.nodeConversation] == [
        "interviewer",
        "candidate",
        "candidate",
        "interviewer",
        "candidate",
    ]


def test_build_answer_evaluation_context_prompt_contains_context() -> None:
    context = build_answer_evaluation_contexts_from_state(_state_with_answer_attempts())[0]

    prompt = build_answer_evaluation_context_prompt(context)

    assert "Target role:" in prompt
    assert "Reference answer:" in prompt
    assert "Candidate answer:" in prompt
    assert "说明 query rewrite" in prompt


def test_answer_evaluation_agent_prompt_requires_new_schema() -> None:
    from app.domain.answer_evaluation_runtime import _build_evaluator_prompt

    context = build_answer_evaluation_contexts_from_state(_state_with_answer_attempts())[0]
    prompt = _build_evaluator_prompt(build_answer_evaluation_context_prompt(context))

    assert '"classification": "direct-answer"' in prompt
    assert '"score": {' in prompt
    assert (
        "Do not put relevance, accuracy, depth, specificity, or clarity at the top level."
        in prompt
    )
    assert "followUpFocus must be an array of strings" in prompt


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


async def test_evaluate_answer_contexts_uses_fake_evaluator_for_batch_results() -> None:
    contexts = build_answer_evaluation_contexts_from_state(_state_with_answer_attempts())
    seen_prompts: list[str] = []
    seen_attempts: list[str] = []

    async def evaluator(prompt: str, context) -> dict[str, Any]:
        seen_prompts.append(prompt)
        seen_attempts.append(context.attemptId)
        return raw_evaluation()

    results = await evaluate_answer_contexts(
        contexts,
        evaluator=evaluator,
        now=lambda: NOW,
        evaluator_model="fake-evaluator",
    )

    assert seen_attempts == ["attempt-main", "attempt-follow-up"]
    assert len(seen_prompts) == 2
    assert [result.attemptId for result in results] == seen_attempts
    assert results[0].taskId == "answer-evaluation-attempt-main"
    assert results[0].score.weightedTotal == 6.9
    assert results[0].evaluatorModel == "fake-evaluator"
    assert results[0].promptVersion == "answer-evaluation-v1"


async def test_evaluate_answer_contexts_rejects_flat_score_output() -> None:
    contexts = build_answer_evaluation_contexts_from_state(_state_with_answer_attempts())

    async def evaluator(_prompt: str, _context) -> dict[str, Any]:
        return {
            "relevance": 10,
            "accuracy": 8,
            "depth": 7,
            "specificity": 6,
            "clarity": 9,
            "strengths": ["覆盖了核心链路"],
            "missingPoints": [],
            "incorrectPoints": [],
            "followUpFocus": "可以继续追问调试挑战。",
        }

    with pytest.raises(Exception, match="RawAnswerEvaluationOutput"):
        await evaluate_answer_contexts(
            contexts[:1],
            evaluator=evaluator,
            now=lambda: NOW,
            evaluator_model="legacy-shape-evaluator",
        )
