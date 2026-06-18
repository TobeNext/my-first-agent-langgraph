import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings
from app.graphs.interview_graph import (
    build_interview_graph,
    invoke_interview_graph,
    snapshot_from_graph_state,
)
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3]
    / "my-first-agent"
    / "PLAN"
    / "fixtures"
    / "contracts"
)
GOLDEN_FIXTURE_NAMES = [
    "unit00-basic-start.json",
    "unit00-start-with-jd.json",
    "unit00-flow-test-skip.json",
]


class EmptyQuestionStore:
    def search(self, *, vector, top_k, round_type):
        return type("Result", (), {"questions": []})()


@pytest.fixture(autouse=True)
def offline_question_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.domain.question_retriever.MilvusQuestionStore",
        lambda: EmptyQuestionStore(),
    )


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _request_from_fixture(fixture: dict[str, Any]) -> MastraStreamRequest:
    return MastraStreamRequest.model_validate(fixture["runtimeRequest"])


def _reply_request(fixture: dict[str, Any], reply: str) -> MastraStreamRequest:
    runtime_request = fixture["runtimeRequest"]
    return MastraStreamRequest.model_validate(
        {
            "messages": [{"role": "user", "content": reply}],
            "memory": runtime_request["memory"],
            "maxSteps": runtime_request.get("maxSteps", 5),
        }
    )


def _assert_subset(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            if key in {"activeNodeTopicContainsAny", "flowTestMockUserReplyCanBePresent"}:
                continue
            _assert_subset(actual[key], value)
        return

    assert actual == expected


def _assert_expected_snapshot(snapshot: InterviewStateSnapshot, expected: dict[str, Any]) -> None:
    payload = snapshot.model_dump()
    _assert_subset(payload, expected)

    expected_topics = expected.get("activeNodeTopicContainsAny")
    if expected_topics:
        active_topic = snapshot.activeNodeTopic or ""
        question_text = snapshot.progress.currentQuestionText or ""
        assert any(topic in active_topic or topic in question_text for topic in expected_topics)


@pytest.mark.parametrize("fixture_name", GOLDEN_FIXTURE_NAMES)
def test_unit00_golden_transcript_start_snapshots(
    fixture_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()
    fixture = _load_fixture(fixture_name)
    context = SqliteSaver.from_conn_string(str(tmp_path / f"{fixture['name']}.db"))
    saver = context.__enter__()
    try:
        graph = build_interview_graph(checkpointer=saver)
        state = invoke_interview_graph(_request_from_fixture(fixture), graph=graph)
        snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))
    finally:
        context.__exit__(None, None, None)

    _assert_expected_snapshot(snapshot, fixture["expectedSnapshotSummary"])


@pytest.mark.parametrize("fixture_name", GOLDEN_FIXTURE_NAMES)
def test_unit00_golden_transcript_first_reply_uses_deterministic_baseline(
    fixture_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()
    fixture = _load_fixture(fixture_name)
    context = SqliteSaver.from_conn_string(str(tmp_path / f"{fixture['name']}-reply.db"))
    saver = context.__enter__()
    try:
        graph = build_interview_graph(checkpointer=saver)
        invoke_interview_graph(_request_from_fixture(fixture), graph=graph)
        state = invoke_interview_graph(
            _reply_request(fixture, fixture["userReplies"][0]),
            graph=graph,
        )
        snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))
    finally:
        context.__exit__(None, None, None)

    assert snapshot.finalReportReady is False
    assert snapshot.progress.currentStage == "follow-up"
    assert snapshot.progress.currentFollowUpIndex == 1
    assert snapshot.progress.currentQuestionText
    assert snapshot.flowTestMockUserReply is None
