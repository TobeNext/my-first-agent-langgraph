from types import SimpleNamespace

import app.graphs.nodes.process_user_reply as process_user_reply_module
from app.graphs.nodes.process_user_reply import process_user_reply_node
from app.schemas.interview_state import AnswerAttemptState, InterviewSessionState
from tests.unit.test_interview_state_machine import _score, _state_fixture

REPORT_GENERATING_REPLY = "面试已结束，报告生成中。生成进度和最终报告可在右上角通知中查看。"


def completed_state_with_attempt() -> InterviewSessionState:
    state = _state_fixture(flow_test=False)
    attempt = AnswerAttemptState.model_validate(
        {
            "id": "attempt-1",
            "targetType": "main-question",
            "targetId": "node-rag",
            "userMessage": "我会先做召回再重排。",
            "classification": "direct-answer",
            "score": _score(5).model_dump(),
            "strengths": ["本地规则优势"],
            "missingPoints": [],
            "incorrectPoints": [],
            "isDetour": False,
            "createdAt": "2026-06-15T00:00:00Z",
        }
    )
    completed_node = state.rounds[0].nodes[0].model_copy(
        update={
            "status": "completed",
            "answerAttempts": [attempt],
            "currentFollowUpId": None,
            "currentTargetType": "main-question",
        },
        deep=True,
    )
    completed_round = state.rounds[0].model_copy(
        update={"status": "completed", "completedNodeCount": 1, "nodes": [completed_node]},
        deep=True,
    )
    return state.model_copy(
        update={
            "phase": "completed",
            "activeRoundId": None,
            "finalReportReady": True,
            "finalReport": "local report",
            "rounds": [completed_round, state.rounds[1].model_copy(update={"status": "skipped"})],
        },
        deep=True,
    )


def test_process_user_reply_moves_completed_state_to_wrap_up_for_inline_report(
    monkeypatch,
) -> None:
    completed_state = completed_state_with_attempt()

    def apply_user_reply(*args, **kwargs):
        return SimpleNamespace(state=completed_state, assistantReply="local report")

    monkeypatch.setattr(process_user_reply_module, "apply_user_reply", apply_user_reply)

    result = process_user_reply_node(
        {
            "session": _state_fixture(flow_test=False).model_dump(),
            "raw_user_message": "最后一题回答",
            "resource_id": "resource-1",
        }
    )

    state = InterviewSessionState.model_validate(result["session"])

    assert state.phase == "wrap-up"
    assert state.activeRoundId is None
    assert state.finalReportReady is False
    assert state.finalReport is None
    assert result["assistant_reply"] == REPORT_GENERATING_REPLY
    assert result["final_report_ready"] is False
