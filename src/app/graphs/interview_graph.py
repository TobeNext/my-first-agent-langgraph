from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from app.domain.interview_initialization_pipeline import initialize_interview_from_kickoff
from app.domain.interview_outcome import (
    create_interview_outcome_snapshot,
)
from app.domain.interview_state_machine import (
    build_interview_progress_summary,
)
from app.domain.rag_recall_sample import (
    write_initialization_rag_recall_sample,
)
from app.graphs.nodes.process_user_reply import (
    process_user_reply_node as run_process_user_reply_node,
)
from app.integrations.checkpoint_store import get_sqlite_checkpointer
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot
from app.schemas.interview_state import InterviewSessionState

GraphAction = Literal["initialize-session", "process-user-reply"]


class InterviewGraphState(TypedDict, total=False):
    thread_id: str
    resource_id: str | None
    raw_user_message: str
    action: GraphAction
    session: dict[str, Any] | None
    assistant_reply: str | None
    snapshot: dict[str, Any] | None
    final_report_ready: bool
    outcome_file_path: str | None
    rag_recall_sample_file_path: str | None
    recall_traces: list[dict[str, Any]]
    generation_trace: list[dict[str, Any]]


def invoke_interview_graph(
    request: MastraStreamRequest,
    *,
    graph: Any | None = None,
) -> InterviewGraphState:
    runtime_graph = graph or get_interview_graph()
    return runtime_graph.invoke(
        {
            "thread_id": request.thread_id,
            "resource_id": request.resource_id,
            "raw_user_message": request.last_user_message(),
        },
        config=thread_config(request.thread_id),
    )


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


@lru_cache
def get_interview_graph() -> Any:
    return build_interview_graph(checkpointer=get_sqlite_checkpointer())


def build_interview_graph(checkpointer: Any) -> Any:
    builder = StateGraph(InterviewGraphState)
    builder.add_node("route_action", route_action)
    builder.add_node("initialize_session", initialize_session_node)
    builder.add_node("process_user_reply", process_user_reply_node)
    builder.add_node("emit_snapshot", emit_snapshot_node)

    builder.add_edge(START, "route_action")
    builder.add_conditional_edges(
        "route_action",
        select_action_node,
        {
            "initialize_session": "initialize_session",
            "process_user_reply": "process_user_reply",
        },
    )
    builder.add_edge("initialize_session", "emit_snapshot")
    builder.add_edge("process_user_reply", "emit_snapshot")
    builder.add_edge("emit_snapshot", END)

    return builder.compile(checkpointer=checkpointer)


def route_action(state: InterviewGraphState) -> InterviewGraphState:
    action: GraphAction = "process-user-reply" if state.get("session") else "initialize-session"
    return {"action": action}


def select_action_node(state: InterviewGraphState) -> str:
    return "process_user_reply" if state["action"] == "process-user-reply" else "initialize_session"


def initialize_session_node(state: InterviewGraphState) -> InterviewGraphState:
    initialized = initialize_interview_from_kickoff(
        thread_id=state["thread_id"],
        raw_kickoff_message=state.get("raw_user_message") or "",
    )
    outcome_file_path, rag_sample_file_path = _create_initial_artifacts(initialized)
    return {
        "session": initialized.state.model_dump(),
        "assistant_reply": initialized.assistantReply,
        "final_report_ready": initialized.state.finalReportReady,
        "outcome_file_path": outcome_file_path,
        "rag_recall_sample_file_path": rag_sample_file_path,
        "recall_traces": [_serialize_trace(trace) for trace in initialized.resources.recallTraces],
        "generation_trace": [
            _serialize_trace(trace) for trace in initialized.resources.generationTrace
        ],
    }


def process_user_reply_node(state: InterviewGraphState) -> InterviewGraphState:
    if not state.get("session"):
        return initialize_session_node(state)
    return run_process_user_reply_node(state)


def emit_snapshot_node(state: InterviewGraphState) -> InterviewGraphState:
    session_payload = state.get("session")
    session = InterviewSessionState.model_validate(session_payload)
    active_round = next(
        (round_item for round_item in session.rounds if round_item.id == session.activeRoundId),
        None,
    )
    active_node = (
        next((node for node in active_round.nodes if node.id == active_round.activeNodeId), None)
        if active_round
        else None
    )
    assistant_reply = state.get("assistant_reply") or session.finalReport or ""
    snapshot = InterviewStateSnapshot.model_validate(
        {
            "assistantReply": assistant_reply,
            "flowTestMockUserReply": None,
            "phase": session.phase,
            "activeRoundType": active_round.type if active_round else None,
            "activeNodeTopic": active_node.topic if active_node else None,
            "finalReportReady": session.finalReportReady,
            "progress": build_interview_progress_summary(session).model_dump(),
        }
    )
    return {
        "assistant_reply": assistant_reply,
        "snapshot": snapshot.model_dump(),
        "final_report_ready": session.finalReportReady,
    }


def snapshot_from_graph_state(state: InterviewGraphState) -> dict[str, Any]:
    snapshot = state.get("snapshot")
    if not snapshot:
        raise ValueError("Interview graph did not emit a snapshot.")
    return cast(dict[str, Any], snapshot)


def assistant_reply_from_graph_state(state: InterviewGraphState) -> str:
    return state.get("assistant_reply") or snapshot_from_graph_state(state)["assistantReply"]


def _create_initial_artifacts(initialized: Any) -> tuple[str | None, str | None]:
    outcome_file_path = None
    rag_sample_file_path = None
    try:
        outcome_file_path = create_interview_outcome_snapshot(
            thread_id=initialized.state.threadId,
            state=initialized.state,
            recall_traces=initialized.resources.recallTraces,
            generation_trace=initialized.resources.generationTrace,
        )
    except Exception:
        outcome_file_path = None
    try:
        rag_sample_file_path = write_initialization_rag_recall_sample(
            thread_id=initialized.state.threadId,
            target_role=initialized.state.targetRole,
            recall_traces=initialized.resources.recallTraces,
            state=initialized.state,
        )
    except Exception:
        rag_sample_file_path = None
    return outcome_file_path, rag_sample_file_path


def _serialize_trace(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)
