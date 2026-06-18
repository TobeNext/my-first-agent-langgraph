import json
import logging
from typing import Any

from app.domain.follow_up_generation import ensure_generated_follow_up_question
from app.domain.interview_state_machine import AnswerEvaluationResult
from app.schemas.interview_state import AnswerScore
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
    assert "Current question dialogue record" in events[0]["prompt"]
    assert events[1]["output"]["followUpQuestion"] == "请具体说明 query rewrite 的触发条件？"
    assert events[1]["metadata"]["normalizedQuestion"] == "请具体说明 query rewrite 的触发条件？"


def test_ensure_generated_follow_up_question_parses_fenced_json_from_raw_model() -> None:
    result = ensure_generated_follow_up_question(
        state=_state_fixture(flow_test=False),
        user_message="我会先做 query rewrite，再召回候选。",
        evaluation=_evaluation(),
        model=RawResponseModel('```json\n{"followUpQuestion":"能展开讲讲召回候选如何重排吗？"}\n```'),
    )

    assert result.followUpQuestion == "能展开讲讲召回候选如何重排吗？"


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
