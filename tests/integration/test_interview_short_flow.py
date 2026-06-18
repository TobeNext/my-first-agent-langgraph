import json
from pathlib import Path

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


def _start_payload(thread_id: str) -> str:
    return json.dumps(
        {
            "requestKind": "interview-start",
            "protocolVersion": "2026-05-structured-start-v1",
            "startInterview": True,
            "threadId": thread_id,
            "resumeMarkdown": "### 专业技能\n- RAG\n- Python\n\n### 项目经历\n- AI 面试系统迁移",
            "jobDescriptionMarkdown": "### 岗位职责\n- 负责 RAG 和 Agent 状态机",
            "settings": {
                "reviewIncorrectOrMissingPoints": True,
                "skipProfessionalSkillsRound": False,
                "skipProjectExperienceRound": True,
                "enableFlowTestMode": False,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 1,
                "projectQuestionCount": 0,
            },
            "resumeSections": {
                "professionalSkills": "- RAG\n- Python",
                "projectExperience": "- AI 面试系统迁移",
            },
        },
        ensure_ascii=False,
    )


def _graph(db_path: Path):
    context = SqliteSaver.from_conn_string(str(db_path))
    saver = context.__enter__()
    return build_interview_graph(checkpointer=saver), context


def _isolate_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()


def test_complete_short_interview_flow_waits_for_async_evaluations_before_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_artifacts(tmp_path, monkeypatch)
    thread_id = "short-flow-thread"
    graph, context = _graph(tmp_path / "short-flow.db")
    try:
        state = invoke_interview_graph(_request(thread_id, _start_payload(thread_id)), graph=graph)
        start_snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))

        assert start_snapshot.progress.totalQuestionCount == 1
        assert start_snapshot.progress.currentStage == "main-question"

        for message in [
            "我会先做向量召回，再做候选重排，并记录 trace。",
            "追问一我会说明状态 checkpoint 和恢复策略。",
            "追问二我会补充失败降级、回滚和观测指标。",
            "追问三我会给出真实项目中的取舍和边界。",
        ]:
            state = invoke_interview_graph(_request(thread_id, message), graph=graph)
    finally:
        context.__exit__(None, None, None)

    final_session = InterviewSessionState.model_validate(state["session"])
    final_snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))

    assert final_session.finalReportReady is False
    assert final_session.phase == "wrap-up"
    assert final_snapshot.progress.currentStage == "completed"
    assert final_snapshot.progress.completedQuestionCount == 1
    assert final_snapshot.assistantReply
    assert "等待异步评分完成" in final_snapshot.assistantReply

    outcome_path = Path(str(state["outcome_file_path"]))
    rag_sample_path = Path(str(state["rag_recall_sample_file_path"]))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    rag_sample = json.loads(rag_sample_path.read_text(encoding="utf-8"))

    assert outcome["threadId"] == thread_id
    assert outcome["session"]["finalReportReady"] is False
    assert outcome["candidateImprovement"]["completedQuestionCount"] == 1
    assert outcome["candidateImprovement"]["report"]["finalReport"] is None
    assert rag_sample["threadId"] == thread_id
    assert "postInterviewAnswerPerformance" in rag_sample["recalls"][0]
    assert rag_sample["interviewSnapshot"]["finalReportReady"] is False
    assert rag_sample["interviewSnapshot"]["answerPerformances"][0]["answerAttemptCount"] >= 1
