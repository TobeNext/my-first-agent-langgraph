from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import get_settings
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
from app.graphs.nodes.initialization import (
    analyze_resume_jd_match_node as run_analyze_resume_jd_match_node,
)
from app.graphs.nodes.initialization import (
    generate_question_set_node as run_generate_question_set_node,
)
from app.graphs.nodes.initialization import (
    judge_question_set_node as run_judge_question_set_node,
)
from app.graphs.nodes.initialization import (
    plan_questions_node as run_plan_questions_node,
)
from app.graphs.nodes.initialization import (
    prepare_initialization_input_node as run_prepare_initialization_input_node,
)
from app.graphs.nodes.initialization import (
    retrieve_historical_memory_node as run_retrieve_historical_memory_node,
)
from app.graphs.nodes.initialization import (
    retrieve_questions_node as run_retrieve_questions_node,
)
from app.graphs.nodes.process_user_reply import (
    apply_reply_transition_node as run_apply_reply_transition_node,
)
from app.graphs.nodes.process_user_reply import (
    classify_user_reply_node as run_classify_user_reply_node,
)
from app.graphs.nodes.process_user_reply import (
    maybe_generate_follow_up_node as run_maybe_generate_follow_up_node,
)
from app.graphs.nodes.process_user_reply import (
    write_answer_artifacts_node as run_write_answer_artifacts_node,
)
from app.graphs.nodes.report_generation import (
    evaluate_answers_node as run_evaluate_answers_node,
)
from app.graphs.nodes.report_generation import (
    generate_report_node as run_generate_report_node,
)
from app.graphs.nodes.report_generation import (
    persist_report_node as run_persist_report_node,
)
from app.graphs.nodes.report_generation import (
    persist_user_memory_node as run_persist_user_memory_node,
)
from app.integrations.checkpoint_store import get_sqlite_checkpointer
from app.langsmith_tracing import langsmith_graph_context
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot
from app.schemas.interview_state import InterviewSessionState
from app.telemetry import interview_protocol_from_message

GraphAction = Literal["initialize-session", "process-user-reply"]


class InterviewGraphState(TypedDict, total=False):
    thread_id: str
    resource_id: str | None
    raw_user_message: str
    action: GraphAction
    stored_user_message: str
    answer_evaluation: dict[str, Any]
    session: dict[str, Any] | None
    assistant_reply: str | None
    snapshot: dict[str, Any] | None
    final_report_ready: bool
    outcome_file_path: str | None
    rag_recall_sample_file_path: str | None
    initialization_input: dict[str, Any] | None
    initialization_resources: dict[str, Any] | None
    professional_question_plan: list[dict[str, Any]]
    historical_memory: dict[str, Any] | None
    resume_jd_match_analysis: dict[str, Any] | None
    retrieved_professional_questions: list[dict[str, Any]]
    retrieved_project_questions: list[dict[str, Any]]
    recall_traces: list[dict[str, Any]]
    generated_professional_questions: list[dict[str, Any]]
    generated_project_questions: list[dict[str, Any]]
    generation_trace: list[dict[str, Any]]
    judged_professional_questions: list[dict[str, Any]]
    judged_project_questions: list[dict[str, Any]]
    judge_trace: list[dict[str, Any]]
    evaluation_contexts: list[dict[str, Any]]
    evaluation_results: list[dict[str, Any]]
    report_output: dict[str, Any] | None
    report_id: str | None
    report_status: str | None
    report_error: str | None
    report_completed_at: str | None
    report_markdown_available: bool
    memory_status: str | None
    memory_error: str | None


def invoke_interview_graph(
    request: MastraStreamRequest,
    *,
    graph: Any | None = None,
) -> InterviewGraphState:
    runtime_graph = graph or get_interview_graph()
    raw_user_message = request.last_user_message()
    settings = get_settings()
    with _get_tracer().start_as_current_span(
        "langgraph.invoke_interview_graph",
        attributes={
            "interview.thread_id": request.thread_id,
            "interview.resource_id": request.resource_id,
            "interview.protocol": interview_protocol_from_message(raw_user_message),
        },
    ) as span:
        try:
            with langsmith_graph_context(settings=settings, thread_id=request.thread_id):
                return runtime_graph.invoke(
                    {
                        "thread_id": request.thread_id,
                        "resource_id": request.resource_id,
                        "raw_user_message": raw_user_message,
                    },
                    config=thread_config(request.thread_id),
                )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


@lru_cache
def get_interview_graph() -> Any:
    return build_interview_graph(checkpointer=get_sqlite_checkpointer())


@lru_cache
def get_report_generation_graph() -> Any:
    return build_report_generation_graph()


def build_interview_graph(checkpointer: Any) -> Any:
    builder = StateGraph(InterviewGraphState)
    builder.add_node("route_action", route_action)
    builder.add_node("prepare_initialization_input", prepare_initialization_input_node)
    builder.add_node("analyze_resume_jd_match", analyze_resume_jd_match_node)
    builder.add_node("retrieve_historical_memory", retrieve_historical_memory_node)
    builder.add_node("plan_questions", plan_questions_node)
    builder.add_node("retrieve_questions", retrieve_questions_node)
    builder.add_node("generate_question_set", generate_question_set_node)
    builder.add_node("judge_question_set", judge_question_set_node)
    builder.add_node("build_session_state", build_session_state_node)
    builder.add_node("write_initialization_artifacts", write_initialization_artifacts_node)
    builder.add_node("classify_user_reply", classify_user_reply_node)
    builder.add_node("maybe_generate_follow_up", maybe_generate_follow_up_node)
    builder.add_node("apply_reply_transition", apply_reply_transition_node)
    builder.add_node("write_answer_artifacts", write_answer_artifacts_node)
    builder.add_node("emit_snapshot", emit_snapshot_node)

    builder.add_edge(START, "route_action")
    builder.add_conditional_edges(
        "route_action",
        select_action_node,
        {
            "initialize_session": "prepare_initialization_input",
            "process_user_reply": "classify_user_reply",
        },
    )
    builder.add_edge("prepare_initialization_input", "analyze_resume_jd_match")
    builder.add_edge("analyze_resume_jd_match", "retrieve_historical_memory")
    builder.add_edge("retrieve_historical_memory", "plan_questions")
    builder.add_edge("plan_questions", "retrieve_questions")
    builder.add_edge("retrieve_questions", "generate_question_set")
    builder.add_edge("generate_question_set", "judge_question_set")
    builder.add_edge("judge_question_set", "build_session_state")
    builder.add_edge("build_session_state", "write_initialization_artifacts")
    builder.add_edge("write_initialization_artifacts", "emit_snapshot")
    builder.add_edge("classify_user_reply", "maybe_generate_follow_up")
    builder.add_edge("maybe_generate_follow_up", "apply_reply_transition")
    builder.add_edge("apply_reply_transition", "write_answer_artifacts")
    builder.add_edge("write_answer_artifacts", "emit_snapshot")
    builder.add_edge("emit_snapshot", END)

    return builder.compile(checkpointer=checkpointer)


def build_report_generation_graph() -> Any:
    builder = StateGraph(InterviewGraphState)
    builder.add_node("evaluate_answers", evaluate_answers_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("persist_report", persist_report_node)
    builder.add_node("persist_user_memory", persist_user_memory_node)
    builder.add_node("emit_report_snapshot", emit_snapshot_node)

    builder.add_edge(START, "evaluate_answers")
    builder.add_edge("evaluate_answers", "generate_report")
    builder.add_edge("generate_report", "persist_report")
    builder.add_edge("persist_report", "persist_user_memory")
    builder.add_edge("persist_user_memory", "emit_report_snapshot")
    builder.add_edge("emit_report_snapshot", END)

    return builder.compile()


def route_action(state: InterviewGraphState) -> InterviewGraphState:
    action: GraphAction = "process-user-reply" if state.get("session") else "initialize-session"
    return {"action": action}


def select_action_node(state: InterviewGraphState) -> str:
    return "process_user_reply" if state["action"] == "process-user-reply" else "initialize_session"


def prepare_initialization_input_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.prepare_initialization_input",
        attributes=_node_span_attributes(state, "prepare_initialization_input"),
    ) as span:
        try:
            result = run_prepare_initialization_input_node(state)
            initialization_input = result.get("initialization_input") or {}
            span.set_attribute(
                "interview.has_structured_start",
                bool(initialization_input.get("hasStructuredStart")),
            )
            span.set_attribute(
                "interview.has_job_description",
                bool(initialization_input.get("hasJobDescription")),
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def analyze_resume_jd_match_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.analyze_resume_jd_match",
        attributes=_node_span_attributes(state, "analyze_resume_jd_match"),
    ) as span:
        try:
            result = run_analyze_resume_jd_match_node(state)
            analysis = result.get("resume_jd_match_analysis") or {}
            matches = analysis.get("resumeJdMatch") if isinstance(analysis, dict) else []
            span.set_attribute("resume_jd_match.match_count", len(matches or []))
            span.set_attribute(
                "resume_jd_match.is_job_matched",
                bool(analysis.get("isJobMatched", True)) if isinstance(analysis, dict) else True,
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def retrieve_historical_memory_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.retrieve_historical_memory",
        attributes=_node_span_attributes(state, "retrieve_historical_memory"),
    ) as span:
        try:
            result = run_retrieve_historical_memory_node(state)
            memory = result.get("historical_memory") or {}
            span.set_attribute(
                "interview.historical_memory.has_memory",
                bool(memory.get("hasMemory")) if isinstance(memory, dict) else False,
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def plan_questions_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.plan_questions",
        attributes=_node_span_attributes(state, "plan_questions"),
    ) as span:
        try:
            result = run_plan_questions_node(state)
            plan = result.get("professional_question_plan") or []
            span.set_attribute("interview.professional_question_plan.count", len(plan))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def retrieve_questions_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.retrieve_questions",
        attributes=_node_span_attributes(state, "retrieve_questions"),
    ) as span:
        try:
            result = run_retrieve_questions_node(state)
            span.set_attribute(
                "rag.retrieved_professional_question_count",
                len(result.get("retrieved_professional_questions") or []),
            )
            span.set_attribute(
                "rag.retrieved_project_question_count",
                len(result.get("retrieved_project_questions") or []),
            )
            span.set_attribute("rag.recall_trace_count", len(result.get("recall_traces") or []))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def generate_question_set_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.generate_question_set",
        attributes=_node_span_attributes(state, "generate_question_set"),
    ) as span:
        try:
            result = run_generate_question_set_node(state)
            span.set_attribute(
                "question_generation.professional_count",
                len(result.get("generated_professional_questions") or []),
            )
            span.set_attribute(
                "question_generation.project_count",
                len(result.get("generated_project_questions") or []),
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def judge_question_set_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.judge_question_set",
        attributes=_node_span_attributes(state, "judge_question_set"),
    ) as span:
        try:
            result = run_judge_question_set_node(state)
            span.set_attribute(
                "question_judge.professional_count",
                len(result.get("judged_professional_questions") or []),
            )
            span.set_attribute(
                "question_judge.project_count",
                len(result.get("judged_project_questions") or []),
            )
            span.set_attribute("question_judge.trace_count", len(result.get("judge_trace") or []))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def build_session_state_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.build_session_state",
        attributes=_node_span_attributes(state, "build_session_state"),
    ) as span:
        try:
            initialized = initialize_interview_from_kickoff(
                thread_id=state["thread_id"],
                raw_kickoff_message=state.get("raw_user_message") or "",
                resume_jd_match_analysis=state.get("resume_jd_match_analysis"),
                historical_memory=state.get("historical_memory"),
                professional_question_plan=state.get("professional_question_plan"),
                retrieved_professional_questions=state.get("retrieved_professional_questions"),
                retrieved_project_questions=state.get("retrieved_project_questions"),
                recall_traces=state.get("recall_traces"),
                generated_professional_questions=state.get("generated_professional_questions"),
                generated_project_questions=state.get("generated_project_questions"),
                generation_trace=state.get("generation_trace"),
                judged_professional_questions=state.get("judged_professional_questions"),
                judged_project_questions=state.get("judged_project_questions"),
                judge_trace=state.get("judge_trace"),
            )
            span.set_attribute("interview.phase", initialized.state.phase)
            span.set_attribute("interview.round_count", len(initialized.state.rounds))
            return {
                "session": initialized.state.model_dump(),
                "assistant_reply": initialized.assistantReply,
                "final_report_ready": initialized.state.finalReportReady,
                "recall_traces": [
                    _serialize_trace(trace) for trace in initialized.resources.recallTraces
                ],
                "generation_trace": [
                    _serialize_trace(trace) for trace in initialized.resources.generationTrace
                ],
                "judge_trace": [
                    _serialize_trace(trace) for trace in initialized.resources.judgeTrace
                ],
            }
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def write_initialization_artifacts_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.write_initialization_artifacts",
        attributes=_node_span_attributes(state, "write_initialization_artifacts"),
    ) as span:
        outcome_file_path, rag_sample_file_path = _create_initial_artifacts_from_graph_state(state)
        span.set_attribute("artifact.outcome_written", bool(outcome_file_path))
        span.set_attribute("artifact.rag_sample_written", bool(rag_sample_file_path))
        return {
            "outcome_file_path": outcome_file_path,
            "rag_recall_sample_file_path": rag_sample_file_path,
        }


def classify_user_reply_node(state: InterviewGraphState) -> InterviewGraphState:
    if not state.get("session"):
        return prepare_initialization_input_node(state)
    with _get_tracer().start_as_current_span(
        "langgraph.node.classify_user_reply",
        attributes=_node_span_attributes(state, "classify_user_reply"),
    ) as span:
        try:
            result = run_classify_user_reply_node(state)
            evaluation = result.get("answer_evaluation") or {}
            if isinstance(evaluation, dict):
                span.set_attribute(
                    "answer.classification",
                    str(evaluation.get("classification") or "unknown"),
                )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def maybe_generate_follow_up_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.maybe_generate_follow_up",
        attributes=_node_span_attributes(state, "maybe_generate_follow_up"),
    ) as span:
        try:
            before = state.get("answer_evaluation") or {}
            result = run_maybe_generate_follow_up_node(state)
            after = result.get("answer_evaluation") or {}
            span.set_attribute(
                "follow_up.generated",
                bool(
                    isinstance(after, dict)
                    and after.get("followUpQuestion")
                    and (
                        not isinstance(before, dict)
                        or after.get("followUpQuestion") != before.get("followUpQuestion")
                    )
                ),
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def apply_reply_transition_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.apply_reply_transition",
        attributes=_node_span_attributes(state, "apply_reply_transition"),
    ) as span:
        try:
            result = run_apply_reply_transition_node(state)
            span.set_attribute("interview.final_report_ready", result["final_report_ready"])
            session = InterviewSessionState.model_validate(result["session"])
            span.set_attribute("interview.phase", session.phase)
            span.set_attribute("interview.round_count", len(session.rounds))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def write_answer_artifacts_node(state: InterviewGraphState) -> InterviewGraphState:
    with _get_tracer().start_as_current_span(
        "langgraph.node.write_answer_artifacts",
        attributes=_node_span_attributes(state, "write_answer_artifacts"),
    ) as span:
        try:
            result = run_write_answer_artifacts_node(state)
            span.set_attribute("artifact.outcome_present", bool(state.get("outcome_file_path")))
            span.set_attribute(
                "artifact.rag_sample_present",
                bool(state.get("rag_recall_sample_file_path")),
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


def evaluate_answers_node(state: InterviewGraphState) -> InterviewGraphState:
    return run_evaluate_answers_node(state)


def generate_report_node(state: InterviewGraphState) -> InterviewGraphState:
    return run_generate_report_node(state)


def persist_report_node(state: InterviewGraphState) -> InterviewGraphState:
    return run_persist_report_node(state)


def persist_user_memory_node(state: InterviewGraphState) -> InterviewGraphState:
    return run_persist_user_memory_node(state)


def should_start_background_report_generation(state: InterviewGraphState) -> bool:
    session_payload = state.get("session")
    if not session_payload:
        return False
    session = InterviewSessionState.model_validate(session_payload)
    return session.phase == "wrap-up" and not session.finalReportReady


def run_report_generation_for_thread(
    thread_id: str,
    *,
    graph: Any | None = None,
    report_graph: Any | None = None,
) -> InterviewGraphState:
    settings = get_settings()
    runtime_graph = graph or get_interview_graph()
    snapshot = runtime_graph.get_state(thread_config(thread_id))
    state: InterviewGraphState = dict(snapshot.values)
    if not should_start_background_report_generation(state):
        return state

    with _get_tracer().start_as_current_span(
        "langgraph.invoke_report_generation_graph",
        attributes={
            "interview.thread_id": thread_id,
        },
    ) as span:
        try:
            with langsmith_graph_context(settings=settings, thread_id=thread_id):
                report_state = (report_graph or get_report_generation_graph()).invoke(
                    state,
                    config=thread_config(thread_id),
                )
                runtime_graph.update_state(thread_config(thread_id), report_state)
                return report_state
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise


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
            generation_trace=initialized.resources.generationTrace,
            judge_trace=initialized.resources.judgeTrace,
        )
    except Exception:
        rag_sample_file_path = None
    return outcome_file_path, rag_sample_file_path


def _create_initial_artifacts_from_graph_state(
    state: InterviewGraphState,
) -> tuple[str | None, str | None]:
    session = InterviewSessionState.model_validate(state.get("session"))
    recall_traces = list(state.get("recall_traces") or [])
    generation_trace = list(state.get("generation_trace") or [])
    judge_trace = list(state.get("judge_trace") or [])
    outcome_file_path = None
    rag_sample_file_path = None
    try:
        outcome_file_path = create_interview_outcome_snapshot(
            thread_id=session.threadId,
            state=session,
            recall_traces=recall_traces,
            generation_trace=generation_trace,
        )
    except Exception:
        outcome_file_path = None
    try:
        rag_sample_file_path = write_initialization_rag_recall_sample(
            thread_id=session.threadId,
            target_role=session.targetRole,
            recall_traces=recall_traces,
            state=session,
            generation_trace=generation_trace,
            judge_trace=judge_trace,
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


def _node_span_attributes(state: InterviewGraphState, node_name: str) -> dict[str, str]:
    attributes = {
        "langgraph.node": node_name,
        "interview.thread_id": state["thread_id"],
    }
    resource_id = state.get("resource_id")
    if resource_id:
        attributes["interview.resource_id"] = resource_id
    return attributes


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer("interview-python-agent")
