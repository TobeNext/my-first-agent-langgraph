import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings
from app.domain.interview_initialization_pipeline import initialize_interview_from_kickoff
from app.domain.interview_memory_summary import InterviewMemorySummaryOutput
from app.graphs.interview_graph import (
    assistant_reply_from_graph_state,
    build_interview_graph,
    invoke_interview_graph,
    run_report_generation_for_thread,
    should_start_background_report_generation,
    snapshot_from_graph_state,
)
from app.integrations.report_repository import InterviewReportRepository
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot
from app.schemas.interview_state import InterviewSessionState
from tests.unit.test_interview_state_machine import _state_fixture


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


def _last_question_session(thread_id: str) -> InterviewSessionState:
    state = _state_fixture(flow_test=False).model_copy(update={"threadId": thread_id}, deep=True)
    settings = state.setup.settings.model_copy(
        update={
            "skipProjectExperienceRound": True,
            "reviewIncorrectOrMissingPoints": False,
        },
        deep=True,
    )
    setup = state.setup.model_copy(update={"settings": settings}, deep=True)
    active_round = state.rounds[0]
    active_node = active_round.nodes[0].model_copy(
        update={
            "followUpCount": 1,
            "maxFollowUps": 1,
            "referenceAnswer": "覆盖 query rewrite、召回、重排、生成和失败降级。",
            "evaluationPoints": ["说明 query rewrite", "说明重排", "说明失败降级"],
        },
        deep=True,
    )
    active_round = active_round.model_copy(update={"nodes": [active_node]}, deep=True)
    project_round = state.rounds[1]
    skipped_project_node = project_round.nodes[0].model_copy(update={"status": "skipped"})
    skipped_project_round = project_round.model_copy(
        update={
            "status": "skipped",
            "activeNodeId": None,
            "nodes": [skipped_project_node],
        },
        deep=True,
    )
    return state.model_copy(
        update={"setup": setup, "rounds": [active_round, skipped_project_round]}
    )


def _start_payload(thread_id: str, *, user_id: str) -> str:
    return json.dumps(
        {
            "requestKind": "interview-start",
            "protocolVersion": "2026-05-structured-start-v1",
            "startInterview": True,
            "threadId": thread_id,
            "userId": user_id,
            "resumeMarkdown": "### 专业技能\n- RAG 检索\n- Python\n\n### 项目经历\n- AI 面试系统",
            "jobDescriptionMarkdown": "### 岗位职责\n- 负责 RAG 检索、失败降级和监控指标",
            "settings": {
                "reviewIncorrectOrMissingPoints": True,
                "skipProfessionalSkillsRound": False,
                "skipProjectExperienceRound": True,
                "enableFlowTestMode": False,
                "enableHistoricalMemory": True,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 1,
                "projectQuestionCount": 0,
            },
            "resumeSections": {
                "professionalSkills": "- RAG 检索\n- Python",
                "projectExperience": "- AI 面试系统",
            },
        },
        ensure_ascii=False,
    )


def test_interview_graph_starts_background_report_without_worker_or_redis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    thread_id = "thread-inline-report-integration"
    report_db = tmp_path / "reports.db"
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("REPORT_DATABASE_URL", f"sqlite:///{report_db}")
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()

    graph, context = _graph(tmp_path / "checkpoints.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "thread_id": thread_id,
                "resource_id": f"frontend-interview-{thread_id}",
                "session": _last_question_session(thread_id).model_dump(mode="json"),
            },
        )
        state = invoke_interview_graph(
            _request(
                thread_id,
                "我会先做 query rewrite，再召回 topK，重排后生成答案，并补充失败降级。",
            ),
            graph=graph,
        )
        report_state = run_report_generation_for_thread(thread_id, graph=graph)
    finally:
        context.__exit__(None, None, None)

    settings = get_settings()
    repository = InterviewReportRepository(database_url=f"sqlite:///{report_db}")
    stored = repository.get_report_by_interview_id(thread_id)
    items = repository.list_items(stored.id if stored else "")
    session = InterviewSessionState.model_validate(state["session"])
    report_session = InterviewSessionState.model_validate(report_state["session"])
    snapshot = InterviewStateSnapshot.model_validate(snapshot_from_graph_state(state))

    assert not hasattr(settings, "redis_url")
    assert should_start_background_report_generation(state) is True
    assert session.phase == "wrap-up"
    assert session.finalReportReady is False
    assert snapshot.phase == "wrap-up"
    assert snapshot.finalReportReady is False
    assert assistant_reply_from_graph_state(state) == (
        "面试已结束，报告生成中。生成进度和最终报告可在右上角通知中查看。"
    )
    assert report_state["report_status"] == "succeeded"
    assert report_state["report_markdown_available"] is True
    assert report_session.phase == "completed"
    assert report_session.finalReportReady is True
    assert stored is not None
    assert stored.status == "succeeded"
    assert stored.markdown.startswith("## 模拟面试报告")
    assert items
    assert report_state["evaluation_contexts"][0]["candidateAnswer"]


def test_report_memory_persists_and_reinforces_next_interview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_thread_id = "thread-memory-first"
    second_thread_id = "thread-memory-second"
    report_db = tmp_path / "reports.db"
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("REPORT_DATABASE_URL", f"sqlite:///{report_db}")
    monkeypatch.setenv("INTERVIEW_MEMORY_USER_ID", "user-a")
    monkeypatch.setenv("OUTCOME_ROOT", str(tmp_path / "Interview outcome"))
    monkeypatch.setenv("RAG_LOG_ROOT", str(tmp_path / "RAG LOG INFO"))
    get_settings.cache_clear()

    async def deterministic_memory_summary(**_kwargs) -> InterviewMemorySummaryOutput:
        return InterviewMemorySummaryOutput(
            weaknessSummary=["RAG 失败降级覆盖不足"],
            missingPoints=["缺少失败降级", "缺少指标阈值"],
            improvementAdvice=["补充失败降级和监控指标"],
            reinforcementQuestionHints=["追问失败时如何降级"],
            normalizedWeaknessKeys=["rag-failure-degradation"],
            improvedAreas=["RAG 链路解释"],
            embeddingText="RAG 检索 失败降级 指标阈值",
        )

    monkeypatch.setattr(
        "app.graphs.nodes.report_generation.generate_interview_memory_summary_with_model",
        deterministic_memory_summary,
    )

    graph, context = _graph(tmp_path / "checkpoints.db")
    try:
        graph.update_state(
            {"configurable": {"thread_id": first_thread_id}},
            {
                "thread_id": first_thread_id,
                "resource_id": f"frontend-interview-{first_thread_id}",
                "session": _last_question_session(first_thread_id).model_dump(mode="json"),
            },
        )
        state = invoke_interview_graph(
            _request(
                first_thread_id,
                "我会先做 query rewrite，再召回 topK，重排后生成答案，但暂时没讲失败降级。",
            ),
            graph=graph,
        )
        report_state = run_report_generation_for_thread(first_thread_id, graph=graph)

        second_state = invoke_interview_graph(
            _request(second_thread_id, _start_payload(second_thread_id, user_id="user-a")),
            graph=graph,
        )
        isolated_state = invoke_interview_graph(
            _request(
                "thread-memory-other",
                _start_payload("thread-memory-other", user_id="user-b"),
            ),
            graph=graph,
        )
    finally:
        context.__exit__(None, None, None)

    repository = InterviewReportRepository(database_url=f"sqlite:///{report_db}")
    memories = repository.list_user_memories("user-a")
    second_session = InterviewSessionState.model_validate(second_state["session"])
    isolated_session = InterviewSessionState.model_validate(isolated_state["session"])
    initialized = initialize_interview_from_kickoff(
        thread_id="thread-memory-resource-check",
        raw_kickoff_message=_start_payload("thread-memory-resource-check", user_id="user-a"),
        memory_repository=repository,
    )
    reinforced_plans = [
        plan
        for plan in initialized.resources.professionalQuestionPlan
        if plan.reinforcementIntent == "review-weakness"
    ]

    assert should_start_background_report_generation(state) is True
    assert report_state["report_status"] == "succeeded"
    assert report_state["memory_status"] == "succeeded"
    assert len(memories) == 1
    assert memories[0].source_interview_id == first_thread_id
    assert second_session.historicalMemory.hasMemory is True
    assert second_session.historicalMemory.sourceInterviewIds == [first_thread_id]
    assert second_session.historicalMemory.missingPoints == ["缺少失败降级", "缺少指标阈值"]
    assert isolated_session.historicalMemory.hasMemory is False
    assert len(reinforced_plans) == 1
    assert reinforced_plans[0].historicalWeaknessSignals

    get_settings.cache_clear()
