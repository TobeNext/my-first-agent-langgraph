import json
import logging
from typing import Any

from app.domain.follow_up_generation import (
    build_dedicated_follow_up_question_prompt,
    ensure_generated_follow_up_question,
)
from app.domain.interview_state_machine import (
    AnswerEvaluationResult,
    get_active_node,
    get_active_round,
)
from app.schemas.interview_state import AnswerScore, HistoricalInterviewMemoryState
from tests.unit.test_interview_state_machine import _state_fixture


def _score(value: float = 7.4) -> AnswerScore:
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


def _evaluation(**overrides: object) -> AnswerEvaluationResult:
    values = {
        "classification": "direct-answer",
        "score": _score(),
        "strengths": ["结构清晰"],
        "missingPoints": ["需要补充实现细节"],
        "incorrectPoints": [],
        "recommendedIntent": "depth",
        "followUpFocus": ["query rewrite"],
        "followUpQuestion": None,
        "shouldCompleteNode": False,
    }
    values.update(overrides)
    return AnswerEvaluationResult(**values)  # type: ignore[arg-type]


class StructuredSuccessModel:
    def with_structured_output(self, schema: type[Any]) -> Any:
        class _Structured:
            def invoke(self, prompt: str) -> Any:
                assert "Return JSON only" in prompt
                return schema(followUpQuestion="请具体说明 query rewrite 的触发条件？")

        return _Structured()


class RawResponseModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> str:
        return self.response


class SequenceRawResponseModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts), len(self.responses)) - 1
        return self.responses[index]


class ErrorModel:
    def with_structured_output(self, schema: type[Any]) -> Any:
        raise RuntimeError("model unavailable")


class StructuredFailsRawSucceedsModel:
    def with_structured_output(self, schema: type[Any]) -> Any:
        class _Structured:
            def invoke(self, prompt: str) -> Any:
                raise RuntimeError("response_format unavailable")

        return _Structured()

    def invoke(self, prompt: str) -> str:
        return '{"followUpQuestion":"你能结合项目说明这个取舍是怎么做的吗？"}'


def test_ensure_generated_follow_up_question_uses_structured_model_output() -> None:
    result = ensure_generated_follow_up_question(
        state=_state_fixture(flow_test=False),
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=StructuredSuccessModel(),
    )

    assert result.followUpQuestion == "请具体说明 query rewrite 的触发条件？"


def test_ensure_generated_follow_up_question_logs_llm_input_and_output(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.llm")

    result = ensure_generated_follow_up_question(
        state=_state_fixture(flow_test=False),
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=StructuredSuccessModel(),
    )

    assert result.followUpQuestion == "请具体说明 query rewrite 的触发条件？"
    events = [json.loads(record.message) for record in caplog.records]
    assert [event["event"] for event in events] == ["llm.input", "llm.output"]
    assert all(event["threadId"] == "thread-1" for event in events)
    assert events[0]["operation"] == "follow-up-question-generation"
    assert "Asked follow-up questions in current interview" in events[0]["prompt"]
    assert events[1]["output"]["followUpQuestion"] == "请具体说明 query rewrite 的触发条件？"
    assert events[1]["metadata"]["normalizedQuestion"] == "请具体说明 query rewrite 的触发条件？"


def test_build_follow_up_question_prompt_injects_memory_after_agent_prompt_in_order() -> None:
    state = _state_fixture(flow_test=False)
    active_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                ),
                state.rounds[0].nodes[0].followUps[1].model_copy(
                    update={
                        "question": "重排阶段如何处理召回噪声？",
                        "status": "answered",
                    }
                ),
            ],
            "answerAttempts": [
                {
                    "id": "attempt-1",
                    "targetType": "main-question",
                    "targetId": "node-rag",
                    "userMessage": "候选人回答原文不能出现在追问记忆 prompt。",
                    "classification": "direct-answer",
                    "score": None,
                    "strengths": [],
                    "missingPoints": [],
                    "incorrectPoints": [],
                    "isDetour": False,
                    "createdAt": "2026-06-19T00:00:00Z",
                }
            ],
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "resumeContext": state.resumeContext.model_copy(
                update={
                    "professionalSkills": "TypeScript, RAG, Agent Memory",
                    "jobDescription": "负责构建带记忆管理的 Agent 面试系统。",
                }
            ),
            "historicalMemory": HistoricalInterviewMemoryState(
                hasMemory=True,
                sourceInterviewIds=["interview-old"],
                weaknesses=["RAG 失败降级覆盖不足"],
                missingPoints=["缺少指标阈值"],
                improvementAdvice=["补充监控和回滚策略"],
                reinforcementQuestionHints=["追问失败时如何降级"],
            ),
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [active_node]}, deep=True),
                state.rounds[1],
            ],
        },
        deep=True,
    )
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_round is not None
    assert active_node is not None

    prompt = build_dedicated_follow_up_question_prompt(
        state=state,
        active_round=active_round,
        active_node=active_node,
        current_question=active_node.mainQuestion,
        user_message="候选人回答原文不能出现在追问记忆 prompt。",
        evaluation=_evaluation(),
    )

    ordered_labels = [
        "Return exactly this shape",
        "User historical interview reports",
        "User resume information",
        "Job description information",
        "Historical interview memory",
        "Previous weak areas and improvement targets",
        "Asked follow-up questions in current interview",
        "Current main question",
    ]
    positions = [prompt.index(label) for label in ordered_labels]
    assert positions == sorted(positions)
    assert "TypeScript, RAG, Agent Memory" in prompt
    assert "负责构建带记忆管理的 Agent 面试系统。" in prompt
    assert "RAG 失败降级覆盖不足" in prompt
    assert "缺少指标阈值" in prompt
    assert "Use historical interview memory only when it is relevant" in prompt
    assert 'Do not ask a generic "last time you did poorly" question.' in prompt
    assert "你如何判断 query rewrite 是否需要触发？" in prompt
    assert "重排阶段如何处理召回噪声？" in prompt
    assert "候选人回答原文" not in prompt
    assert "Current question dialogue record" not in prompt


def test_ensure_generated_follow_up_question_parses_fenced_json_from_raw_model() -> None:
    result = ensure_generated_follow_up_question(
        state=_state_fixture(flow_test=False),
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=RawResponseModel('```json\n{"followUpQuestion":"能展开讲讲召回候选如何重排吗？"}\n```'),
    )

    assert result.followUpQuestion == "能展开讲讲召回候选如何重排吗？"


def test_ensure_generated_follow_up_question_rejects_duplicate_llm_output() -> None:
    state = _state_fixture(flow_test=False)
    active_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                ),
                state.rounds[0].nodes[0].followUps[1],
            ],
            "followUpCount": 1,
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [active_node]}, deep=True),
                state.rounds[1],
            ]
        },
        deep=True,
    )
    baseline = _evaluation()

    result = ensure_generated_follow_up_question(
        state=state,
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=baseline,
        model=RawResponseModel('{"followUpQuestion":"你如何判断 query rewrite 是否需要触发?"}'),
    )

    assert result == baseline


def test_ensure_generated_follow_up_question_retries_once_after_duplicate() -> None:
    state = _state_fixture(flow_test=False)
    active_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                ),
                state.rounds[0].nodes[0].followUps[1],
            ],
            "followUpCount": 1,
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [active_node]}, deep=True),
                state.rounds[1],
            ]
        },
        deep=True,
    )
    model = SequenceRawResponseModel(
        [
            '{"followUpQuestion":"你如何判断 query rewrite 是否需要触发?"}',
            '{"followUpQuestion":"那 query rewrite 失败时你会如何回退？"}',
        ]
    )

    result = ensure_generated_follow_up_question(
        state=state,
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=model,
    )

    assert result.followUpQuestion == "那 query rewrite 失败时你会如何回退？"
    assert len(model.prompts) == 2
    assert "Rejected duplicate question" in model.prompts[1]
    assert "Choose a different uncovered angle" in model.prompts[1]


def test_ensure_generated_follow_up_question_rejects_two_duplicate_attempts(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.llm")
    state = _state_fixture(flow_test=False)
    active_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                ),
                state.rounds[0].nodes[0].followUps[1],
            ],
            "followUpCount": 1,
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [active_node]}, deep=True),
                state.rounds[1],
            ]
        },
        deep=True,
    )
    baseline = _evaluation()

    result = ensure_generated_follow_up_question(
        state=state,
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=baseline,
        model=SequenceRawResponseModel(
            [
                '{"followUpQuestion":"你如何判断 query rewrite 是否需要触发?"}',
                '{"followUpQuestion":"你如何判断 query rewrite 是否需要触发的？"}',
            ]
        ),
    )

    assert result == baseline
    events = [json.loads(record.message) for record in caplog.records]
    outputs = [event for event in events if event["event"] == "llm.output"]
    assert [event["metadata"]["attemptIndex"] for event in outputs] == [1, 2]
    assert all(event["metadata"]["duplicateRejected"] is True for event in outputs)


def test_ensure_generated_follow_up_question_falls_back_to_raw_json_when_structured_fails() -> None:
    result = ensure_generated_follow_up_question(
        state=_state_fixture(flow_test=False),
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=StructuredFailsRawSucceedsModel(),
    )

    assert result.followUpQuestion == "你能结合项目说明这个取舍是怎么做的吗？"


def test_ensure_generated_follow_up_question_keeps_fallback_on_empty_invalid_or_error() -> None:
    baseline = _evaluation()

    for model in [
        RawResponseModel('{"followUpQuestion":"   "}'),
        RawResponseModel("not json"),
        ErrorModel(),
    ]:
        result = ensure_generated_follow_up_question(
            state=_state_fixture(flow_test=False),
            user_message="我会先做 query rewrite，再召回候选。",
            evaluation=baseline,
            model=model,
        )
        assert result == baseline


def test_ensure_generated_follow_up_question_skips_existing_question_and_detours() -> None:
    existing = _evaluation(followUpQuestion="已有追问")
    detour = _evaluation(classification="meta-question", score=None)

    assert (
        ensure_generated_follow_up_question(
            state=_state_fixture(flow_test=False),
            user_message="我想了解评分标准",
            evaluation=existing,
            model=StructuredSuccessModel(),
        )
        == existing
    )
    assert (
        ensure_generated_follow_up_question(
            state=_state_fixture(flow_test=False),
            user_message="我想了解评分标准",
            evaluation=detour,
            model=StructuredSuccessModel(),
        )
        == detour
    )
