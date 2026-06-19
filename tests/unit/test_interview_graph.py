from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import app.graphs.interview_graph as interview_graph_module
import app.graphs.nodes.process_user_reply as process_user_reply_module
from app.config import get_settings
from app.graphs.interview_graph import (
    assistant_reply_from_graph_state,
    build_interview_graph,
    invoke_interview_graph,
    snapshot_from_graph_state,
)
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot
from app.schemas.interview_state import InterviewSessionState


def _request(thread_id: str, message: str) -> MastraStreamRequest:
    return MastraStreamRequest.model_validate(
        {
            "messages": [{"role": "user", "content": message}],
            "memory": {
                "thread": thread_id,
                "resource": f"frontend-interview-{thread_id}",
            },
            "maxSteps": 5,
        }
    )


def _graph(db_path: Path):
    context = SqliteSaver.from_conn_string(str(db_path))
    saver = context.__enter__()
    return build_interview_graph(checkpointer=saver), context


def _isolate_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()


def _session_fixture(thread_id: str = "thread-1", *, flow_test: bool = False) -> InterviewSessionState:
    return InterviewSessionState.model_validate(
        {
            "version": 1,
            "threadId": thread_id,
            "targetRole": "通用技术岗位",
            "company": None,
            "responseLanguage": "zh",
            "phase": "professional-skills-round",
            "activeRoundId": "round-professional",
            "finalReportReady": False,
            "finalReport": None,
            "setup": {
                "selectedDirection": "通用技术岗位",
                "directionSource": "derived",
                "settings": {
                    "reviewIncorrectOrMissingPoints": True,
                    "skipProfessionalSkillsRound": False,
                    "skipProjectExperienceRound": False,
                    "enableFlowTestMode": flow_test,
                    "professionalQuestionMode": "custom-count",
                    "professionalQuestionCount": 1,
                    "projectQuestionCount": 1,
                },
            },
            "resumeContext": {
                "professionalSkills": "TypeScript\nRAG",
                "projectExperience": "AI 面试 Agent 状态机改造",
                "jobDescription": "",
                "resumeParsed": True,
            },
            "lastCorrectionSummary": None,
            "rounds": [
                {
                    "id": "round-professional",
                    "type": "professional-skills",
                    "status": "in-progress",
                    "plannedNodeCount": 1,
                    "completedNodeCount": 0,
                    "activeNodeId": "node-rag",
                    "nodeOrder": ["node-rag"],
                    "nodes": [
                        {
                            "id": "node-rag",
                            "topic": "RAG",
                            "source": "knowledge-base",
                            "mainQuestion": "请解释你的 RAG 链路。",
                            "status": "awaiting-main-answer",
                            "currentTargetType": "main-question",
                            "currentFollowUpId": None,
                            "followUpCount": 0,
                            "maxFollowUps": 3,
                            "detourResponseCount": 0,
                            "earlyCompletionReason": None,
                            "followUps": [
                                {
                                    "id": "follow-up-1",
                                    "index": 1,
                                    "intent": "depth",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                },
                                {
                                    "id": "follow-up-2",
                                    "index": 2,
                                    "intent": "accuracy",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                },
                            ],
                            "answerAttempts": [],
                            "aggregatedScore": None,
                            "summary": None,
                        }
                    ],
                },
                {
                    "id": "round-project",
                    "type": "project-experience",
                    "status": "pending",
                    "plannedNodeCount": 1,
                    "completedNodeCount": 0,
                    "activeNodeId": "node-project",
                    "nodeOrder": ["node-project"],
                    "nodes": [
                        {
                            "id": "node-project",
                            "topic": "状态机改造",
                            "source": "resume",
                            "mainQuestion": "请介绍你的状态机项目。",
                            "status": "pending",
                            "currentTargetType": "main-question",
                            "currentFollowUpId": None,
                            "followUpCount": 0,
                            "maxFollowUps": 2,
                            "detourResponseCount": 0,
                            "earlyCompletionReason": None,
                            "followUps": [
                                {
                                    "id": "project-follow-up-1",
                                    "index": 1,
                                    "intent": "depth",
                                    "question": "",
                                    "status": "pending",
                                    "linkedAnswerId": None,
                                }
                            ],
                            "answerAttempts": [],
                            "aggregatedScore": None,
                            "summary": None,
                        }
                    ],
                },
            ],
        }
    )


def _mock_graph_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def initialize_interview_from_kickoff(thread_id: str, raw_kickoff_message: str):
        session = _session_fixture(thread_id)
        return SimpleNamespace(
            state=session,
            assistantReply=session.rounds[0].nodes[0].mainQuestion,
            resources=SimpleNamespace(recallTraces=[], generationTrace=[]),
        )

    monkeypatch.setattr(
        interview_graph_module,
        "initialize_interview_from_kickoff",
        initialize_interview_from_kickoff,
    )


def test_graph_start_returns_legal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    graph, context = _graph(tmp_path / "start.db")
    try:
        state = invoke_interview_graph(_request("thread-start", "开始面试"), graph=graph)
    finally:
        context.__exit__(None, None, None)

    snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))

    assert snapshot.phase == "professional-skills-round"
    assert snapshot.activeRoundType == "professional-skills"
    assert snapshot.progress.currentStage == "main-question"
    assert assistant_reply_from_graph_state(state) == snapshot.assistantReply


def test_graph_continue_uses_checkpointed_session_and_processes_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    graph, context = _graph(tmp_path / "continue.db")
    try:
        start_state = invoke_interview_graph(_request("thread-continue", "开始面试"), graph=graph)
        next_state = invoke_interview_graph(
            _request("thread-continue", "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    start_session = InterviewSessionState.model_validate(start_state["session"])
    next_session = InterviewSessionState.model_validate(next_state["session"])
    next_snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(next_state))

    assert len(start_session.rounds[0].nodes[0].answerAttempts) == 0
    assert len(next_session.rounds[0].nodes[0].answerAttempts) == 1
    assert next_snapshot.progress.currentStage == "follow-up"
    assert next_snapshot.progress.currentFollowUpIndex == 1


def test_graph_continue_uses_generated_follow_up_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    generated_question = "请说明 LLM 生成的追问如何落进状态机？"

    def generated_follow_up(**kwargs):
        return replace(kwargs["evaluation"], followUpQuestion=generated_question)

    monkeypatch.setattr(
        process_user_reply_module,
        "ensure_generated_follow_up_question",
        generated_follow_up,
    )
    thread_id = "thread-generated-follow-up"
    graph, context = _graph(tmp_path / "generated-follow-up.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"session": _session_fixture(thread_id).model_dump()},
        )
        next_state = invoke_interview_graph(
            _request(thread_id, "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    next_snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(next_state))

    assert next_snapshot.progress.currentStage == "follow-up"
    assert next_snapshot.progress.currentQuestionText == generated_question
    assert assistant_reply_from_graph_state(next_state) == generated_question


def test_graph_continue_enqueues_answer_evaluation_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    calls: list[dict] = []

    def enqueue_spy(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        process_user_reply_module,
        "enqueue_answer_evaluation_task_best_effort",
        enqueue_spy,
    )
    graph, context = _graph(tmp_path / "enqueue.db")
    try:
        invoke_interview_graph(_request("thread-enqueue", "开始面试"), graph=graph)
        invoke_interview_graph(
            _request("thread-enqueue", "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    assert len(calls) == 1
    assert calls[0]["before_state"].threadId == "thread-enqueue"
    assert calls[0]["after_state"].threadId == "thread-enqueue"
    assert calls[0]["user_message"] == "我会先召回候选，再重排和生成答案。"


def test_graph_flow_test_skip_does_not_enqueue_answer_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    calls: list[dict] = []

    def enqueue_spy(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        process_user_reply_module,
        "enqueue_answer_evaluation_task_best_effort",
        enqueue_spy,
    )
    graph, context = _graph(tmp_path / "flow-test-skip-enqueue.db")
    try:
        start_state = invoke_interview_graph(_request("thread-flow-skip", "开始面试"), graph=graph)
        session = InterviewSessionState.model_validate(start_state["session"])
        session.setup.settings.enableFlowTestMode = True
        graph.update_state(
            {"configurable": {"thread_id": "thread-flow-skip"}},
            {"session": session.model_dump()},
        )
        invoke_interview_graph(
            _request("thread-flow-skip", "[FLOW_TEST_SKIP]"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    assert calls == []


def test_graph_checkpoints_are_isolated_by_thread_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    graph, context = _graph(tmp_path / "isolated.db")
    try:
        invoke_interview_graph(_request("thread-a", "开始面试"), graph=graph)
        invoke_interview_graph(_request("thread-a", "我会回答第一题。"), graph=graph)
        thread_b_state = invoke_interview_graph(_request("thread-b", "开始面试"), graph=graph)
    finally:
        context.__exit__(None, None, None)

    thread_b_session = InterviewSessionState.model_validate(thread_b_state["session"])

    assert len(thread_b_session.rounds[0].nodes[0].answerAttempts) == 0
    assert thread_b_session.threadId == "thread-b"
