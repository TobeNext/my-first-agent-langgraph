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
    run_report_generation_for_thread,
    should_start_background_report_generation,
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


def _session_fixture(
    thread_id: str = "thread-1",
    *,
    flow_test: bool = False,
) -> InterviewSessionState:
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
    def initialize_interview_from_kickoff(
        thread_id: str,
        raw_kickoff_message: str,
        resume_jd_match_analysis: dict | None = None,
        historical_memory: dict | None = None,
        professional_question_plan: list[dict] | None = None,
        retrieved_professional_questions: list[dict] | None = None,
        retrieved_project_questions: list[dict] | None = None,
        recall_traces: list[dict] | None = None,
        generated_professional_questions: list[dict] | None = None,
        generated_project_questions: list[dict] | None = None,
        generation_trace: list[dict] | None = None,
        judged_professional_questions: list[dict] | None = None,
        judged_project_questions: list[dict] | None = None,
        judge_trace: list[dict] | None = None,
    ):
        session = _session_fixture(thread_id)
        assert resume_jd_match_analysis is not None
        assert historical_memory is not None
        assert professional_question_plan is not None
        assert retrieved_professional_questions is not None
        assert retrieved_project_questions is not None
        assert recall_traces is not None
        assert generated_professional_questions is not None
        assert generated_project_questions is not None
        assert generation_trace is not None
        assert judged_professional_questions is not None
        assert judged_project_questions is not None
        assert judge_trace is not None
        return SimpleNamespace(
            state=session,
            assistantReply=session.rounds[0].nodes[0].mainQuestion,
            resources=SimpleNamespace(recallTraces=[], generationTrace=[], judgeTrace=[]),
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
    assert state["initialization_input"]["threadId"] == "thread-start"
    assert state["initialization_input"]["hasStructuredStart"] is False
    assert state["initialization_input"]["protocol"] == "reply"
    assert state["resume_jd_match_analysis"]["isJobMatched"] is True
    assert state["historical_memory"]["hasMemory"] is False
    assert len(state["professional_question_plan"]) == 1
    assert isinstance(state["retrieved_professional_questions"], list)
    assert isinstance(state["retrieved_project_questions"], list)
    assert isinstance(state["generated_professional_questions"], list)
    assert isinstance(state["generated_project_questions"], list)
    assert isinstance(state["judged_professional_questions"], list)
    assert isinstance(state["judged_project_questions"], list)


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


def test_graph_continue_does_not_prepare_initialization_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    thread_id = "thread-skip-prepare"
    graph, context = _graph(tmp_path / "skip-prepare.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"session": _session_fixture(thread_id).model_dump()},
        )
        state = invoke_interview_graph(
            _request(thread_id, "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    assert "initialization_input" not in state
    assert InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))


def test_graph_carries_initialization_intermediate_state_without_snapshot_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    thread_id = "thread-initialization-state"
    graph, context = _graph(tmp_path / "initialization-state.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "thread_id": thread_id,
                "session": _session_fixture(thread_id).model_dump(),
                "initialization_input": {"threadId": thread_id},
                "initialization_resources": {"normalizedProfessionalSkills": ["RAG"]},
                "professional_question_plan": [{"primarySkill": "RAG"}],
                "historical_memory": {"hasMemory": False},
                "resume_jd_match_analysis": {"resumeJdMatch": ["RAG"]},
                "judge_trace": [{"status": "accepted"}],
            },
        )
        state = invoke_interview_graph(
            _request(thread_id, "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    snapshot = snapshot_from_graph_state(state)

    assert state["initialization_input"] == {"threadId": thread_id}
    assert state["initialization_resources"] == {"normalizedProfessionalSkills": ["RAG"]}
    assert state["professional_question_plan"] == [{"primarySkill": "RAG"}]
    assert state["historical_memory"] == {"hasMemory": False}
    assert state["resume_jd_match_analysis"] == {"resumeJdMatch": ["RAG"]}
    assert state["judge_trace"] == [{"status": "accepted"}]
    assert "initialization_input" not in snapshot
    assert "professional_question_plan" not in snapshot


def test_graph_start_does_not_trigger_report_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    calls: list[str] = []

    def evaluate_spy(state):
        calls.append("evaluate")
        return {}

    monkeypatch.setattr(interview_graph_module, "run_evaluate_answers_node", evaluate_spy)
    graph, context = _graph(tmp_path / "start-no-report.db")
    try:
        invoke_interview_graph(_request("thread-start-no-report", "开始面试"), graph=graph)
    finally:
        context.__exit__(None, None, None)

    assert calls == []


def test_graph_normal_reply_does_not_trigger_report_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    calls: list[str] = []

    def evaluate_spy(state):
        calls.append("evaluate")
        return {}

    monkeypatch.setattr(interview_graph_module, "run_evaluate_answers_node", evaluate_spy)
    graph, context = _graph(tmp_path / "reply-no-report.db")
    try:
        invoke_interview_graph(_request("thread-reply-no-report", "开始面试"), graph=graph)
        invoke_interview_graph(
            _request("thread-reply-no-report", "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    assert calls == []


def test_graph_wrap_up_returns_immediately_without_report_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    calls: list[str] = []
    thread_id = "thread-report-chain"

    def apply_reply_transition(state):
        session = _session_fixture(thread_id).model_copy(
            update={
                "phase": "wrap-up",
                "activeRoundId": None,
                "finalReportReady": False,
                "finalReport": None,
            },
            deep=True,
        )
        return {
            "session": session.model_dump(),
            "assistant_reply": "报告生成中",
            "final_report_ready": False,
        }

    monkeypatch.setattr(
        interview_graph_module,
        "run_apply_reply_transition_node",
        apply_reply_transition,
    )

    graph, context = _graph(tmp_path / "report-chain.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"session": _session_fixture(thread_id).model_dump()},
        )
        state = invoke_interview_graph(_request(thread_id, "最后一题回答"), graph=graph)
    finally:
        context.__exit__(None, None, None)

    session = InterviewSessionState.model_validate(state["session"])
    snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))

    assert calls == []
    assert session.phase == "wrap-up"
    assert session.finalReportReady is False
    assert snapshot.phase == "wrap-up"
    assert snapshot.finalReportReady is False
    assert assistant_reply_from_graph_state(state) == "报告生成中"
    assert should_start_background_report_generation(state) is True


def test_background_report_generation_runner_updates_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = "thread-background-report"
    calls: list[str] = []

    def evaluate_answers(state):
        calls.append("evaluate")
        return {"evaluation_results": [{"attemptId": "attempt-1"}], "report_status": "evaluated"}

    def generate_report(state):
        calls.append("generate")
        assert state["evaluation_results"] == [{"attemptId": "attempt-1"}]
        return {"report_output": {"markdown": "report"}, "report_status": "generated"}

    def persist_report(state):
        calls.append("persist")
        session = InterviewSessionState.model_validate(state["session"]).model_copy(
            update={
                "phase": "completed",
                "activeRoundId": None,
                "finalReportReady": True,
                "finalReport": "报告已生成",
            },
            deep=True,
        )
        return {
            "session": session.model_dump(),
            "assistant_reply": "报告已生成",
            "final_report_ready": True,
            "report_id": "report-thread-background-report",
            "report_markdown_available": True,
            "report_status": "succeeded",
        }

    monkeypatch.setattr(interview_graph_module, "run_evaluate_answers_node", evaluate_answers)
    monkeypatch.setattr(interview_graph_module, "run_generate_report_node", generate_report)
    monkeypatch.setattr(interview_graph_module, "run_persist_report_node", persist_report)

    graph, context = _graph(tmp_path / "background-report.db")
    try:
        wrap_up_session = _session_fixture(thread_id).model_copy(
            update={
                "phase": "wrap-up",
                "activeRoundId": None,
                "finalReportReady": False,
                "finalReport": None,
            },
            deep=True,
        )
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"session": wrap_up_session.model_dump(), "thread_id": thread_id},
        )
        state = run_report_generation_for_thread(thread_id, graph=graph)
        checkpoint_state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    finally:
        context.__exit__(None, None, None)

    session = InterviewSessionState.model_validate(state["session"])
    checkpoint_session = InterviewSessionState.model_validate(checkpoint_state["session"])

    assert calls == ["evaluate", "generate", "persist"]
    assert state["report_status"] == "succeeded"
    assert session.finalReportReady is True
    assert checkpoint_session.finalReportReady is True


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


def test_graph_continue_processes_reply_without_inline_report_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    graph, context = _graph(tmp_path / "reply.db")
    try:
        invoke_interview_graph(_request("thread-reply", "开始面试"), graph=graph)
        state = invoke_interview_graph(
            _request("thread-reply", "我会先召回候选，再重排和生成答案。"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    session = InterviewSessionState.model_validate(state["session"])

    assert len(session.rounds[0].nodes[0].answerAttempts) == 1


def test_graph_flow_test_skip_still_advances_without_inline_report_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    _mock_graph_initialization(monkeypatch)
    graph, context = _graph(tmp_path / "flow-test-skip.db")
    try:
        start_state = invoke_interview_graph(_request("thread-flow-skip", "开始面试"), graph=graph)
        session = InterviewSessionState.model_validate(start_state["session"])
        session.setup.settings.enableFlowTestMode = True
        graph.update_state(
            {"configurable": {"thread_id": "thread-flow-skip"}},
            {"session": session.model_dump()},
        )
        state = invoke_interview_graph(
            _request("thread-flow-skip", "[FLOW_TEST_SKIP]"),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    next_session = InterviewSessionState.model_validate(state["session"])

    assert len(next_session.rounds[0].nodes[0].answerAttempts) == 1


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
