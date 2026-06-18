import json

from app.sse import mastra_compatible_stream


def _data_lines(chunks: list[bytes]) -> list[str]:
    raw = b"".join(chunks).decode("utf-8")
    return [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]


def test_mastra_sse_stream_shape_matches_frontend_parser_contract() -> None:
    snapshot = {
        "assistantReply": "hello world",
        "flowTestMockUserReply": None,
        "phase": "professional-skills-round",
        "activeRoundType": "professional-skills",
        "activeNodeTopic": "RAG",
        "finalReportReady": False,
        "progress": {
            "totalQuestionCount": 1,
            "completedQuestionCount": 0,
            "remainingQuestionCount": 1,
            "currentQuestionIndex": 1,
            "currentRoundType": "professional-skills",
            "currentRoundLabel": "专业技能",
            "currentStage": "main-question",
            "currentFollowUpIndex": None,
            "currentQuestionText": "hello world",
            "currentNodeTopic": "RAG",
        },
    }

    lines = _data_lines(list(mastra_compatible_stream("hello world", snapshot)))
    events = [json.loads(line) for line in lines if line != "[DONE]"]

    assert events[0]["type"] == "text-delta"
    assert events[-1]["type"] == "tool-result"
    assert events[-1]["payload"]["toolName"] == "interviewStateManagerTool"
    assert events[-1]["payload"]["result"] == snapshot
    assert lines[-1] == "[DONE]"
